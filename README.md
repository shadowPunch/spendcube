# spendcube

Turns a company's raw purchase files and its category tree into a spend cube: every
purchase line counted once, cleaned, given a supplier key, an item key and a category,
and every excluded line given a reason code.

A new company needs a new YAML config, not new code.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Task data is confidential and is not in this repository. Put each company's files in
its own folder, for example `data/company_A/` and `data/company_B/` (both git-ignored).

### LLM access (optional, recommended)

Classification uses an LLM for the items that carry most of the spend. Without one the
tool still runs on its local fallback, but accuracy on an unseen tree is lower.
Put keys in a `.env` file at the repository root (git-ignored).

| Backend | How | `.env` |
|---|---|---|
| `codex` (default) | AgentRouter through the [Codex CLI](https://github.com/openai/codex) (`codex` on `PATH`). AgentRouter only accepts known agent clients, so the tool runs `codex exec` read-only in an empty folder with its own `CODEX_HOME`. | `AGENTROUTER_API_KEY=...` |
| `openai` | Any OpenAI-compatible endpoint. Set `SPENDCUBE_LLM=openai`, optionally `SPENDCUBE_LLM_MODEL` and the `base_url` in `configs/defaults.yaml`. | `OPENAI_API_KEY=...` |
| `none` | Local fallback only. Set `SPENDCUBE_LLM=none`. | none |

LLM answers are cached in `.cache/llm/`, so a re-run is repeatable and makes no new calls.
If the gateway fails three times in a row, the run stops calling it and the local model
labels the rest; the run summary says so.

### Experiment tracking

Runs are logged to Weights & Biases (project `spendcube`): config, settings, counts,
accuracy, LLM calls and tokens, runtime. Only configs and metrics are sent, never data rows.
Set `SPENDCUBE_WANDB=0` to run untracked.

## The one command

```bash
uv run spendcube run --config configs/A.yaml --data-dir data/company_A --out outputs/<name>__A.csv
uv run spendcube run --config configs/B.yaml --data-dir data/company_B --out outputs/<name>__B.csv
```

**A new delivery of a known company** uses the same config, plus the folder that holds the
company's labelled delivery, so the tool learns from it:

```bash
uv run spendcube run --config configs/A.yaml --data-dir data/<new-delivery> --reference-dir data/company_A --out outputs/<name>__<new-delivery>.csv
```

**A new company** needs only a config:

1. Look at its files: `uv run spendcube profile data/<company>/<file>.csv`.
2. Copy `configs/TEMPLATE.yaml` (every option, commented) to `configs/<company>.yaml` and map
   the columns. The config handles other separators and encodings, Excel files, decimal
   commas, debit/credit flags, tax amounts or rates, FX rates quoted either way, and trees
   given with code columns or as one path column.
3. If the file has no invoice line-number column, set `duplicate_columns` (or
   `exact_duplicates: false`); otherwise two genuine identical lines on one invoice look
   like one line exported twice.
4. Run it as above, without `--reference-dir`.

### Other commands

```bash
uv run spendcube run ... --hide-labels      # zero-label path: ignore the labels file
uv run spendcube holdout --config configs/A.yaml --data-dir data/company_A   # accuracy on unseen items
uv run python experiments/effort.py --config configs/A.yaml --data-dir data/company_A  # LLM reasoning effort
env -u PYTHONPATH uv run --group dev pytest # tests (unset PYTHONPATH if ROS or similar injects pytest plugins)
```

Besides the output file, each run writes `runs/<out-name>_details.csv` (git-ignored) with
the decision route, duplicate partner, data-quality flags and FX rate for every record.

## Outputs

| File | Produced by (W&B run) | Notes |
|---|---|---|
| `outputs/demo__A.csv` | `A-final-v4` | A's labels as examples |
| `outputs/demo__B.csv` | `B-final-v1` | B's seed labels as examples; the LLM labelled 4,000 items (the current default labels about 2,800) |

`demo` is a placeholder for the author's name.

## Repository layout

```
configs/       company configs (A, B), TEMPLATE for a new company, shared defaults and terms
spendcube/     the package (modules below)
tests/         unit tests and an end-to-end test on a made-up company
experiments/   measurements behind design choices (reasoning effort)
outputs/       output files for A and B
```

## How it works

```
config + files
  -> ingest       map every source onto one canonical record layout
  -> normalise    amounts, VAT removal, dates, currency conversion
  -> suppliers    one key per real supplier
  -> count once   collapse re-exports; match the two systems one-to-one
  -> decide       CANCELLED / NOT_APPROVED / DUPLICATE / NON_PROCUREMENT
  -> items        one key per real item
  -> classify     known examples -> LLM -> local model
  -> write        output file + contract checks
```

| Module | Purpose |
|---|---|
| `config.py` | Company config schema (pydantic), validated on load |
| `ingest.py` | Reads files as text and maps columns onto the canonical record |
| `normalise.py` | Amount, tax-rate and date parsing; FX conversion with a plausibility check |
| `text.py` | Supplier-name and description normalisation, OCR folding, invoice-number core |
| `dedupe.py` | Union-find grouping of suppliers and items |
| `match.py` | Re-export collapse and tiered one-to-one cross-system matching |
| `decide.py` | Exclusion rules in a fixed order |
| `taxonomy.py` | Category tree read at run time (any depth up to 4) |
| `classify.py` | Signature building and the three classification routes |
| `llm.py` | Codex and OpenAI-compatible backends with cache, retries, circuit breaker and token accounting |
| `evaluate.py` | Accuracy, grouping and exclusion scores; synthetic matching test |
| `output.py` | Output columns and contract checks |
| `pipeline.py` | Orchestration, `run` and `holdout` |
| `profile.py` | `spendcube profile`: column summary of an unfamiliar file |
| `tracking.py` | W&B wrapper with an off switch |

Settings shared by all companies (thresholds, LLM batch size, uninformative descriptions)
live in `configs/defaults.yaml`; generic non-procurement terms in
`configs/non_procurement_terms.yaml`. Neither holds category names.

### Key design decisions

- **Exact amounts for matching.** Tool lines match ERP lines only when the net amount agrees
  to the cent (one cent of rounding allowed). Near-amounts on the same PO are separate
  deliveries in this data.
- **Re-exports need the same invoice number.** Identical lines under different invoice
  numbers are repeat purchases (median 79 days apart), not duplicates.
- **One-way item links.** A matched invoice line takes its goods receipt's item but cannot
  merge two items through its noisy text.
- **Classify signatures, not lines.** A signature is one supplier group plus one item group;
  about 6,000 for A and 13,000 for B. The LLM labels the ones that make up 99% of describable
  spend, highest spend first; a local model trained on its answers (and on any labels)
  covers the long tail.
- **Codex for AgentRouter.** A short instructions file and disabled agent tools cut Codex's
  fixed overhead from about 17k to 10k input tokens per call; the gateway caches nothing, so
  items go 200 per call to spread that overhead.
- **The LLM answers with leaf names, not numbered IDs.** With IDs like `C02` the model
  sometimes wrote an ID 20 places off; names are schema-checked and mapped back in code.

See `DECISIONS.md` for what the data showed and what was decided, and `APPROACH.md` for
the one-page method summary with measured accuracy, runtime and LLM usage.
