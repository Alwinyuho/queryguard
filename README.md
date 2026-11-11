# QueryGuard

**A text-to-SQL agent that can't change your data. It writes SQL, checks it is read-only, runs it in a sandbox, and repairs it from the error message. Evaluated on BIRD Mini-Dev with a small local model.**

Asking an LLM to write SQL is easy. Trusting the result is the hard part. QueryGuard is built around three questions: **is it safe, does it run, and is it right?** It answers each one with measurements.

> **Live demo:** not deployed yet. Run it locally with `python -m queryguard.ui`. You need the BIRD databases, plus a GPU or an LLM API key.

## Results

### Safety (red-team test)

| Test | Result |
|---|---|
| Harmful SQL sent to the static guard | **22 of 22 blocked** |
| The same SQL sent straight to the sandbox, with the guard **switched off** | **22 of 22 blocked** |
| Harmful natural-language requests where the model wrote harmful SQL anyway | 8 of 10 |
| Those requests refused, with a clear message to the user | **8 of 8** (the other 2 got harmless read-only queries the model chose itself) |
| Database files changed after the test (SHA-256) | **none** |

The attacks covered DELETE, DROP, UPDATE, INSERT and REPLACE, stacked statements (`SELECT 1; DROP ...`), ATTACH, PRAGMA writable_schema, CREATE TABLE AS, triggers, VACUUM INTO, `load_extension`, comment and casing tricks, and prompt injection ("ignore all previous instructions..."). See [scripts/run_adversarial.py](scripts/run_adversarial.py).

**Lesson:** the model is not a safety layer. It readily wrote harmful SQL when asked, and safety comes from the two layers that don't depend on it.

### Accuracy (BIRD Mini-Dev, SQLite, 2026 revision)

The test set is a fixed random sample of 150 of the 500 questions, covering all 11 databases. 143 of the 150 are rated "challenging". The model is Qwen3-4B, run locally in 4-bit on an 8 GB laptop GPU, with greedy decoding. The metric is BIRD's official execution accuracy: the predicted and reference results must contain the same set of rows.

| Setup | Correct, first attempt | Correct, with up to 2 repairs | 95% CI | Seconds per question |
|---|---|---|---|---|
| Full schema in the prompt (about 4,600 tokens) | 4.0% | 4.7% | 2.3% to 9.3% | about 18 to 29 \* |
| **Schema linking (top 30 columns + join keys)** | 5.3% | **6.0%** | 3.2% to 11.0% | **11** |

\* Measured while running alone. The logged average of 55 s includes a period when another job was competing for the CPU.

What the numbers say:

- **Running is not the same as being correct.** Repair cut the queries that fail to run from 44 to 21, and raised the queries that run from 93 to 114. Yet correct answers rose by only 1 question. Most wrong answers run fine and simply answer a slightly different question, for example by picking the wrong column, adding extra output columns, or inventing a filter.
- **The model invents columns.** "no such column" made up 68 of the 93 execution errors with the full schema. That is the clearest lever for improvement.
- **Schema linking is more than twice as fast, at no cost in accuracy.** Its answers were a superset of the full-schema answers: every question the full schema solved, plus 2 more. The accuracy gap itself is within noise.
- **This is a hard benchmark for a 4B model.** The 2026 revision of Mini-Dev clarified output requirements and removed hints that gave away schema links. Compare setups with each other, and do not compare with leaderboard results from frontier models.

## How it works

```
question ─> schema text (CREATE TABLE + column descriptions + example values)
            optional: keep the 30 most similar columns + keys (bge-small embeddings)
                    │
                    v
              LLM writes SQL  <─────────────── feedback: error text, or "no rows returned"
                    │                                        ^
          static guard (sqlglot syntax tree)                 │
            ├─ write attempt   -> refuse; never "repaired"   │
            ├─ malformed SQL   -> repair ────────────────────┤
            └─ one read-only SELECT                          │
                    │                                        │
          sandbox: read-only file + SQLite authorizer        │
                   + time limit + row cap                    │
            ├─ error or empty result -> repair (max 2) ──────┘
            └─ rows -> answer, with the SQL and the trace of every attempt
```

Design choices:
- **Check the syntax tree, not the text.** Keyword filters are fooled by comments, odd casing and stacked statements. The guard parses the SQL and walks the tree, and it rejects anything that can't be parsed, since what can't be checked can't be trusted.
- **Defence in depth.** Even with the guard switched off, the database is opened with `mode=ro`, and SQLite's authorizer allows only SELECT, READ and FUNCTION actions. The tests run write attempts directly against the sandbox and check the file's hash.
- **Refuse writes; don't repair them.** An early version fed "write blocked" back to the model, which then quietly answered a different, read-only question. Now write attempts stop with a clear refusal, and only malformed SQL is repaired.
- **Repair never sees the answer.** Feedback is only the database error, or "no rows returned", so the loop is fair to use in production.
- **Crash-proof guard.** A model once wrote an unclosed backtick. That raised a tokenizer error, not a parse error, and crashed the run. It is now caught and covered by a regression test.
- **Resumable evaluation.** Results are saved every 25 questions, so a crash or reboot loses at most one chunk.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[ml,ui,dev]"
```

Download the BIRD Mini-Dev databases (`minidev.zip`, linked from the [BIRD site](https://bird-bench.github.io/)) and extract `MINIDEV/dev_databases` to `../_downloads/bird/extracted/dev_databases`, or set `BIRD_DIR`. The questions are fetched from [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev).

```bash
pytest -q                                              # 52 tests, no GPU or data needed
python scripts/run_eval.py --schema linked --sample 150  # benchmark, resumable
python scripts/run_adversarial.py                       # red-team test
python -m queryguard.ui                                 # demo at http://127.0.0.1:7861
```

To use an API model instead of the local GPU, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`. Any OpenAI-compatible endpoint works, such as Groq, Gemini or Ollama. See [.env.example](.env.example).

## Limitations

- **Small sample, small model.** 150 questions with one 4B model, so differences of 1 to 2 points are within noise. A stronger model behind the same harness is the obvious next experiment.
- **The model is local-only for now.** A free CPU host cannot run it, so a deployed demo would need an API model. The [Dockerfile](Dockerfile) (API, databases mounted at `/data/bird`) has not been built or tested, because Docker was not available on the development machine.
- **The guard is for SQLite.** Other database dialects need their own parser settings and permission checks. In production, also use a read-only database user.
- **Some questions are ambiguous.** No ambiguity detection or clarifying questions are implemented yet.

## Credits

- BIRD Mini-Dev: Li et al., *Can LLM Already Serve as a Database Interface?* (NeurIPS 2023), CC BY-SA 4.0.
- Models: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
