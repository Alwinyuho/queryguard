"""Measure execution accuracy on BIRD Mini-Dev (SQLite), before and after self-repair.

Generation runs in batched rounds: every question gets a first attempt, then only
failed attempts (error, guard block, empty result) get repaired, up to MAX_REPAIRS.
Repair never looks at the gold answer, so it is fair to use at inference time.

Usage: python scripts/run_eval.py --schema full|linked [--limit N]
Output: runs/<schema>.jsonl and reports/<schema>.json
"""

import argparse
import json
import math
import pickle
import random
import time
from collections import Counter, defaultdict

import httpx

from queryguard import config
from queryguard.agent import (
    QueryAgent,
    check,
    extract_sql,
    initial_messages,
    needs_repair,
    repair_messages,
)
from queryguard.executor import execute, results_match
from queryguard.llm import LocalLLM

GOLD_TIMEOUT_S = 120.0
CHUNK = 25
SAMPLE_SEED = 13


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    """95% confidence interval for a proportion; reliable even near 0% or 100%."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_questions() -> list[dict]:
    path = config.RUNS_DIR / "questions.jsonl"
    if not path.exists():
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            httpx.get(config.QUESTIONS_URL, follow_redirects=True, timeout=60).text,
            encoding="utf-8",
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_gold(questions: list[dict]) -> dict[int, list[tuple] | None]:
    path = config.RUNS_DIR / "gold.pkl"
    gold = pickle.loads(path.read_bytes()) if path.exists() else {}
    missing = [q for q in questions if q["question_id"] not in gold]
    for q in missing:
        result = execute(
            config.database_path(q["db_id"]), q["SQL"], timeout_s=GOLD_TIMEOUT_S, max_rows=10**7
        )
        gold[q["question_id"]] = result.rows if result.ok else None
    if missing:
        path.write_bytes(pickle.dumps(gold))
    return gold


def accuracy_table(records: list[dict], key: str) -> dict:
    by_level = defaultdict(list)
    for r in records:
        by_level[r["difficulty"]].append(r[key])
        by_level["overall"].append(r[key])
    return {
        level: {
            "n": len(v),
            "accuracy": sum(v) / len(v),
            "ci95": wilson_interval(sum(v), len(v)),
        }
        for level, v in by_level.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=["full", "linked"], default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="random subset size")
    args = parser.parse_args()

    questions = load_questions()
    if args.sample:
        # Fixed seed, so every configuration is scored on the same questions.
        questions = random.Random(SAMPLE_SEED).sample(questions, args.sample)
    questions = questions[: args.limit]
    gold = load_gold(questions)
    unusable = [q["question_id"] for q in questions if gold[q["question_id"]] is None]
    questions = [q for q in questions if gold[q["question_id"]] is not None]
    print(
        f"{len(questions)} questions, {len(unusable)} skipped because gold SQL failed", flush=True
    )

    out_path = config.RUNS_DIR / f"{args.schema}.jsonl"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                done[record["question_id"]] = record
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"resuming: {len(done)} already scored, {len(todo)} to go", flush=True)

    llm = LocalLLM()
    agent = QueryAgent(llm, use_linking=args.schema == "linked")
    for chunk_start in range(0, len(todo), CHUNK):
        chunk = todo[chunk_start : chunk_start + CHUNK]
        began = time.perf_counter()
        records = run_chunk(chunk, gold, llm, agent)
        seconds = (time.perf_counter() - began) / len(chunk)
        with out_path.open("a", encoding="utf-8") as handle:
            for record in records:
                record["seconds"] = seconds
                done[record["question_id"]] = record
                handle.write(json.dumps(record, default=str) + "\n")
        print(f"{len(done)}/{len(questions)} scored, {seconds:.1f}s per question", flush=True)

    records = [done[q["question_id"]] for q in questions]
    write_report(records, args.schema, llm.name, unusable)


def run_chunk(chunk, gold, llm, agent) -> list[dict]:
    """First attempt for every question, then repair rounds only for failed attempts."""
    conversations = [
        initial_messages(
            q["question"],
            q["evidence"],
            agent.schema_text(q["db_id"], q["question"], q["evidence"]),
        )
        for q in chunk
    ]
    state = [{"steps": [], "final": None, "first": None} for _ in chunk]
    pending = list(range(len(chunk)))
    for attempt in range(config.MAX_REPAIRS + 1):
        replies = llm.generate_batch([conversations[i] for i in pending])
        still_pending = []
        for i, reply in zip(pending, replies, strict=True):
            q = chunk[i]
            step, result = check(q["db_id"], extract_sql(reply), attempt)
            rows = result.rows if result and result.ok else None
            state[i]["steps"].append(
                {"sql": step.sql, "status": step.status, "detail": step.detail}
            )
            if attempt == 0:
                state[i]["first"] = rows
            if state[i]["final"] is None or step.status in {"ok", "empty"}:
                state[i]["final"] = rows
            if needs_repair(step) and attempt < config.MAX_REPAIRS:
                conversations[i] = repair_messages(conversations[i], reply, step)
                still_pending.append(i)
        pending = still_pending
        if not pending:
            break

    records = []
    for q, s in zip(chunk, state, strict=True):
        expected = gold[q["question_id"]]
        records.append(
            {
                "question_id": q["question_id"],
                "db_id": q["db_id"],
                "difficulty": q["difficulty"],
                "correct_first_attempt": s["first"] is not None
                and results_match(s["first"], expected),
                "correct_with_repair": s["final"] is not None
                and results_match(s["final"], expected),
                "attempts": len(s["steps"]),
                "steps": s["steps"],
            }
        )
    return records


def write_report(records, schema, model_name, unusable) -> None:
    first_status = Counter(r["steps"][0]["status"] for r in records)
    final_status = Counter(
        next(
            (st["status"] for st in reversed(r["steps"]) if st["status"] in {"ok", "empty"}),
            r["steps"][-1]["status"],
        )
        for r in records
    )
    report = {
        "model": model_name,
        "schema": schema,
        "n": len(records),
        "skipped_gold_failed": unusable,
        "first_attempt": accuracy_table(records, "correct_first_attempt"),
        "with_repair": accuracy_table(records, "correct_with_repair"),
        "first_attempt_status": dict(first_status),
        "final_status": dict(final_status),
        "mean_llm_calls": sum(r["attempts"] for r in records) / len(records),
        "seconds_per_question": sum(r["seconds"] for r in records) / len(records),
        "hardware": "RTX 5050 Laptop GPU 8 GB, batched greedy decoding",
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORT_DIR / f"{schema}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "skipped_gold_failed"}, indent=2))


if __name__ == "__main__":
    main()
