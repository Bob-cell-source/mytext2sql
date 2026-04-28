# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python project; all main entrypoints live at the repo root. `final_code.py` runs the end-to-end SQL generation pipeline and writes logs to `./logs/`. `generate_sft_dpo_data.py` builds training artifacts under paths such as `./output/training_data/`. `train_sft.py`, `train_dpo.py`, and `testdpo.py` cover SFT training, DPO training, and adapter inference. `sql_exe.py` contains the MySQL/StarRocks execution helper. Expected input files are referenced under `./data/` (for example `final_dataset.json` and `schema_with.json`).

## Build, Test, and Development Commands
Use a Python 3.10+ virtual environment and install the libraries imported by the scripts (`transformers`, `trl`, `peft`, `datasets`, `openai`, `pymysql`, `numpy`).

```bash
python final_code.py
python generate_sft_dpo_data.py --input ./data/final_dataset.json --schema ./data/schema_with.json --outdir ./output/training_data
python train_sft.py --train ./output/training_data/sft_final.jsonl --out ./model
python train_dpo.py --sft_adapter ./model --train ./output/training_data/dpo_pairs.jsonl --out ./model_dpo
python testdpo.py
```

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, snake_case for functions/variables, and UPPER_CASE for constants such as prompt templates. Keep new modules script-friendly and avoid introducing package-only patterns unless the repo is being reorganized. Match the current file style when editing; several scripts use CRLF line endings. Prefer small helper functions over deeply nested logic.

## Testing Guidelines
There is no formal `pytest` suite yet. Validate changes with targeted script runs and smoke tests against small datasets. For model pipeline changes, verify generated files in `./output/training_data/`, check `./logs/`, and run `python testdpo.py` against a known sample. If you add reusable logic, include a lightweight test script or adopt `pytest` in the same change.

## Commit & Pull Request Guidelines
History is minimal (`init`), so keep commits short, imperative, and scoped, for example `train: fix DPO prompt mapping`. PRs should describe the changed workflow, required env vars, sample commands used for verification, and any output-path or schema assumptions. Include log snippets or JSON examples when behavior changes; screenshots are usually unnecessary.

## Security & Configuration Tips
Do not commit real API keys, DB passwords, model weights, or generated data dumps. Prefer environment variables over hardcoded secrets: `OPENAI_API_KEY` or `DASHSCOPE_API_KEY`, `LLM_MODEL`, `DASHSCOPE_BASE_URL`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and `DB_PORT`.
