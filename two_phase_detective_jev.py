"""Two-phase Jev evaluation pipeline for long detective novels.

Phase 1 — chunk scoring:
  The novel is split into ~CHUNK_SIZE character chunks at line boundaries.
  Every chunk is scored by sending it to Jev with a structured state:

      [BACKGROUND CONTEXT]
      {novel opening + tail of high-scoring passages seen so far}

      [CURRENT PASSAGE]
      {the chunk being scored}

  For each question, a Jev `score` request is sent (5 ordered levels,
  0 = unrelated → 4 = direct answer/key revelation). Each chunk's total
  is the sum of per-question scores (float, max = 4 * num_questions).
  Chunks scoring >= PRIOR_INCLUSION_THRESHOLD are added to the rolling
  prior so later chunks are never read cold.

Phase 2 — final evaluation:
  Chunks are ranked by total score descending. The highest-scoring chunks
  are selected greedily until MAX_FINAL_STATE characters are filled, then
  re-sorted into original narrative order. A short meta-header explains
  to Jev that the passages are non-contiguous excerpts. The assembled
  state is sent in a single Jev `choice` request for the answers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent / "DetectiveQA"
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

CHUNK_SIZE = 25_000
NOVEL_OPENING_CHARS = 2_000
ROLLING_PRIOR_CHARS = 4_000
MAX_FINAL_STATE = 100_000

# Each string below describes one relevance level, from least to most relevant.
# Jev returns a continuous float across these levels (0.0 = first level, 4.0 = last level), 
# plus per-level probabilities and confidence.
SCORE_LEVELS: list[str] = [
    "Unrelated — nothing in the passage helps answer this question",
    "Weakly related background, atmosphere, or filler",
    "Mentions characters, places, or events tied to the question",
    "Contains a specific clue, alibi detail, action, or piece of evidence relevant to the question",
    "Contains a direct answer, key revelation, confession, or definitive evidence for the question",
]
SCORE_MAX = len(SCORE_LEVELS) - 1  # 4.0
# A chunk's total = sum of per-question scores. Chunks whose total
# reaches the threshold below contribute to rolling prior context for later chunks.
PRIOR_INCLUSION_THRESHOLD = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase Jev evaluation for long DetectiveQA novels."
    )
    p.add_argument("--overwrite", action="store_true", help="Re-run already completed novels.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Characters per chunk in phase 1.")
    p.add_argument("--all", action="store_true", help="Run every novel in the dataset.")
    p.add_argument("--novel-id", type=int, default=None, help="Single novel id.")
    p.add_argument(
        "--annotation",
        choices=("AIsup_anno", "human_anno"),
        default="AIsup_anno",
        help="Annotation split to use with --novel-id (default: AIsup_anno).",
    )
    return p.parse_args()


def load_novel(root: Path, novel_id: int) -> tuple[Path, str]:
    matches = sorted((root / "novel_data_en").glob(f"{novel_id}-*.txt"))
    if not matches:
        raise FileNotFoundError(f"No novel file for id {novel_id} under {root / 'novel_data_en'}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple novel files matched {novel_id}: {matches}")
    path = matches[0]
    return path, path.read_text(encoding="utf-8-sig")


def annotation_files(root: Path, split: str) -> list[tuple[int, Path]]:
    directory = root / "anno_data_en" / split
    if not directory.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {directory}")
    files: list[tuple[int, Path]] = []
    for path in directory.glob("*.json"):
        try:
            novel_id = int(path.stem)
        except ValueError:
            continue
        files.append((novel_id, path))
    if not files:
        raise FileNotFoundError(f"No numbered JSON files found in {directory}")
    return sorted(files)


def call_jev(
    endpoint: str,
    api_key: str,
    timeout: float,
    model: str,
    state: str,
    questions: dict[str, dict[str, Any]],
    max_retries: int = 3,
) -> tuple[dict[str, Any], float, int]:
    try:
        import requests
        from requests.exceptions import ConnectionError, SSLError, Timeout
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: python -m pip install requests") from exc
    payload = {"state": state, "model": model, "questions": questions}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    started = time.perf_counter()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0:
            delay = 2 ** attempt
            print(f"    retrying in {delay}s (attempt {attempt + 1}/{max_retries})...", flush=True)
            time.sleep(delay)
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout
            )
            elapsed = time.perf_counter() - started
            try:
                body = response.json()
            except ValueError:
                body = {"raw_text": response.text}
            if not response.ok:
                raise RuntimeError(
                    f"Jev API returned HTTP {response.status_code}: "
                    f"{json.dumps(body, ensure_ascii=False)[:2000]}"
                )
            if not isinstance(body, dict):
                raise RuntimeError("Jev API returned a non-object JSON response")
            return body, elapsed, response.status_code
        except (SSLError, ConnectionError, Timeout) as exc:
            last_exc = exc
            print(f"    transient error: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}") from last_exc


def chunk_novel(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    raw = text.split("\n\n")
    if len(raw) == 1:
        raw = text.split("\n")

    units: list[str] = []
    for unit in raw:
        if len(unit) <= chunk_size:
            units.append(unit)
        else:
            for i in range(0, len(unit), chunk_size):
                units.append(unit[i: i + chunk_size])

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = len(unit) + 1
        if current and current_len + unit_len > chunk_size:
            chunks.append("\n".join(current))
            current = [unit]
            current_len = unit_len
        else:
            current.append(unit)
            current_len += unit_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_scoring_questions(mc_questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build Jev `score` questions — one per question — to grade passage relevance."""
    questions: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(mc_questions, start=1):
        qid = f"q{i:03d}"
        questions[qid] = {
            "type": "score",
            "instructions": (
                "Rate how relevant the CURRENT PASSAGE is for answering this "
                "specific detective-story question. Use the ordered levels from "
                "0 (unrelated) to the highest level (contains a direct answer or "
                "key revelation). Question: "
                f"{str(item.get('question', '')).strip()}"
            ),
            "criteria": SCORE_LEVELS,
        }
    return questions


def build_mc_questions(
    annotation: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    questions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for i, item in enumerate(annotation.get("questions", []), start=1):
        qid = f"q{i:03d}"
        options = item.get("options")
        if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
            raise ValueError(f"Question {qid} does not have exactly A/B/C/D options")
        questions[qid] = {
            "type": "choice",
            "instructions": (
                "Choose the correct answer to this multiple-choice question. "
                f"{str(item.get('question', '')).strip()}"
            ),
            "criteria": {k: str(v).strip() for k, v in options.items()},
        }
        records.append({
            "id": qid,
            "index": i,
            "question": item.get("question"),
            "options": options,
            "gold_answer": item.get("answer"),
            "answer_position": item.get("answer_position"),
        })
    if not questions:
        raise ValueError("Annotation contains no questions")
    return questions, records


def phase1_score(
    *,
    novel_text: str,
    mc_questions: list[dict[str, Any]],
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float,
    position: str,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    """Score every chunk with Jev `score` questions.

    Each MC question gets a per-passage relevance score in [0, SCORE_MAX]. The
    chunk's total is the sum of per-question scores (float). Higher = more relevant.
    Returns (chunks, total_scores, chunk_results).
    """
    chunks = chunk_novel(novel_text, chunk_size=chunk_size)
    opening = novel_text[:NOVEL_OPENING_CHARS]
    scoring_qs = build_scoring_questions(mc_questions)
    scores: list[float] = []
    chunk_results: list[dict[str, Any]] = []
    prior_texts: list[str] = []
    max_total = SCORE_MAX * len(scoring_qs)

    for idx, chunk in enumerate(chunks):
        rolling_prior = "".join(prior_texts)[-ROLLING_PRIOR_CHARS:] if prior_texts else ""
        state = (
            "[BACKGROUND CONTEXT - story so far]\n"
            + opening
            + ("\n\n" + rolling_prior if rolling_prior else "")
            + "\n\n[CURRENT PASSAGE - assess relevance to the questions below]\n"
            + chunk
        )
        print(
            f"  {position} phase 1 — chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)",
            flush=True,
        )
        body, elapsed, _ = call_jev(
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            model=model,
            state=state,
            questions=scoring_qs,
        )
        answers = body.get("answers", {})
        per_question: dict[str, dict[str, Any]] = {}
        total = 0.0
        for qid in scoring_qs:
            ans = answers.get(qid)
            if isinstance(ans, dict):
                s_val = ans.get("score")
                per_question[qid] = {
                    "score": s_val,
                    "confidence": ans.get("confidence"),
                    "probabilities": ans.get("probabilities"),
                }
                if isinstance(s_val, (int, float)):
                    total += float(s_val)
            else:
                per_question[qid] = {"score": None, "confidence": None, "probabilities": None}
        scores.append(total)

        # Rolling prior grows with chunks scoring above threshold, so later chunks
        # have real narrative context (characters, events) — not just the opening.
        if total >= PRIOR_INCLUSION_THRESHOLD:
            prior_texts.append(chunk)

        print(f"    -> score {total:.2f}/{max_total:.1f}", flush=True)
        chunk_results.append({
            "chunk_index": idx,
            "chunk_chars": len(chunk),
            "score": total,
            "per_question": per_question,
            "elapsed_seconds": elapsed,
        })

    return chunks, scores, chunk_results


def select_chunks(
    chunks: list[str],
    scores: list[float],
    max_chars: int = MAX_FINAL_STATE,
) -> list[str]:
    """Greedy select highest-scoring chunks that fit in max_chars, restore original order."""
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    selected: list[int] = []
    total = 0
    for i in order:
        if total + len(chunks[i]) > max_chars:
            continue
        selected.append(i)
        total += len(chunks[i])
    selected.sort()
    return [chunks[i] for i in selected]


def phase2_evaluate(
    *,
    selected_texts: list[str],
    annotation: dict[str, Any],
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float,
    position: str,
) -> dict[str, Any]:
    meta = (
        "The following are selected excerpts from a detective novel, "
        "arranged in original narrative order. Excerpts are separated by "
        "\"---\" and may not be contiguous — text between excerpts has been "
        "omitted."
    )
    final_state = meta + "\n\n" + "\n\n---\n\n".join(selected_texts)
    mc_questions, records = build_mc_questions(annotation)
    print(
        f"  {position} phase 2 — {len(selected_texts)} chunks selected, "
        f"{len(final_state)} chars",
        flush=True,
    )
    body, elapsed, status_code = call_jev(
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        model=model,
        state=final_state,
        questions=mc_questions,
    )
    answers = body.get("answers", {})
    if not isinstance(answers, dict):
        raise RuntimeError("Jev response has no object-valued 'answers' field")
    correct = 0
    for record in records:
        raw = answers.get(record["id"])
        if isinstance(raw, dict):
            choice = raw.get("choice") or raw.get("answer")
            confidence = raw.get("confidence")
            probabilities = raw.get("probabilities")
        else:
            choice, confidence, probabilities = raw, None, None
        record.update({
            "predicted_answer": choice,
            "confidence": confidence,
            "probabilities": probabilities,
            "prediction_raw": raw,
        })
        record["correct"] = choice == record["gold_answer"]
        correct += int(record["correct"])

    return {
        "state_chars": len(final_state),
        "question_count": len(records),
        "correct_count": correct,
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "http_status": status_code,
        "usage": body.get("usage"),
        "questions": records,
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return v.get("status") == "complete" and isinstance(v.get("phase2"), dict)


def evaluate_novel(
    *,
    root: Path,
    novel_id: int,
    annotation_path: Path,
    annotation_split: str,
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float,
    position: str,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    novel_path, novel_text = load_novel(root, novel_id)
    data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError(f"Unexpected annotation structure in {annotation_path}")
    annotation = data[0]
    mc_questions = annotation.get("questions", [])
    if not mc_questions:
        raise ValueError(f"No questions in {annotation_path}")

    chunks, scores, chunk_results = phase1_score(
        novel_text=novel_text,
        mc_questions=mc_questions,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        position=position,
        chunk_size=chunk_size,
    )

    selected_texts = select_chunks(chunks, scores)
    if not selected_texts:
        selected_texts = [novel_text[:MAX_FINAL_STATE]]

    p2 = phase2_evaluate(
        selected_texts=selected_texts,
        annotation=annotation,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        position=position,
    )

    selected_chars = sum(len(t) for t in selected_texts)
    print(
        f"  {position} complete — "
        f"{p2['correct_count']}/{p2['question_count']} ({p2['accuracy']:.2%}), "
        f"phase1_calls={len(chunks)}, selected={len(selected_texts)}/{len(chunks)} chunks",
        flush=True,
    )

    return {
        "status": "complete",
        "model": model,
        "novel_id": novel_id,
        "novel_file": str(novel_path),
        "annotation_file": str(annotation_path),
        "annotation_split": annotation_split,
        "endpoint": endpoint,
        "original_chars": len(novel_text),
        "phase1": {
            "total_chunks": len(chunks),
            "selected_chunks": len(selected_texts),
            "total_chars": len(novel_text),
            "selected_chars": selected_chars,
            "chunk_results": chunk_results,
        },
        "phase2": p2,
    }


def main() -> int:
    args = parse_args()
    if not args.novel_id and not args.all:
        print("Specify --novel-id or --all.", file=sys.stderr)
        return 2
    api_key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if not api_key:
        print("Set TYPESAFE_API_KEY (or JEV_API_KEY) before running.", file=sys.stderr)
        return 2

    root = DEFAULT_ROOT.resolve()
    output_dir = (root.parent / "jev_results_twophase").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.novel_id:
        pairs: list[tuple[int, Path]] = [
            (args.novel_id, root / "anno_data_en" / args.annotation / f"{args.novel_id}.json")
        ]
    else:
        splits = ["AIsup_anno", "human_anno"] if args.all else [args.annotation]
        pairs = []
        for split in splits:
            pairs.extend(annotation_files(root, split))
        if not pairs:
            raise ValueError("No annotation files found")

    errors = 0
    for idx, (novel_id, anno_path) in enumerate(pairs, start=1):
        annotation_split = anno_path.parent.name
        result_path = output_dir / f"novel_{novel_id:03d}_{annotation_split}.json"
        position = f"[{idx}/{len(pairs)}] novel {novel_id}:"
        if not args.overwrite and completed(result_path):
            print(f"{position} skipped (already complete)", flush=True)
            continue
        print(f"{position} starting", flush=True)
        started = time.perf_counter()
        try:
            result = evaluate_novel(
                root=root,
                novel_id=novel_id,
                annotation_path=anno_path,
                annotation_split=annotation_split,
                model=DEFAULT_MODEL,
                endpoint=DEFAULT_ENDPOINT,
                api_key=api_key,
                timeout=600.0,
                position=position,
                chunk_size=args.chunk_size,
            )
            save_json(result_path, result)
        except Exception as exc:
            errors += 1
            elapsed = time.perf_counter() - started
            print(
                f"  {position} error after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
