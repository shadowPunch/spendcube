# Approach

## Pipeline

One config-driven pipeline; each stage is a small module (see README).

1. **Ingest.** A YAML config maps each source file onto one canonical record (supplier,
   description, item code, amount, currency, tax rate, dates, status, PO, invoice) plus
   named context columns. A new company needs a new config only.
2. **Normalise.** Tolerant parsers for amounts, VAT notations and several date formats;
   failures are flagged. Gross amounts lose VAT at the stated rate (15% when blank).
   Currency: reporting-currency lines as they are; otherwise the line rate when it is
   within 2× of the FX table, else the table for the invoice month (posting month when
   the invoice date is implausible).
3. **Count once.** Exact duplicate rows are dropped. Re-exports in the invoice tool share
   supplier, invoice-number core, amount and quantity. Tool lines then match ERP lines
   one-to-one in three tiers: PO + line + quantity + amount; PO + amount; supplier +
   amount + similar text. Only approved tool lines take part. Unmatched tool lines are
   spend that exists only in the tool.
4. **Decide.** Status → duplicate → company rules (e.g. B's intercompany accounts) →
   generic multilingual term list (taxes, intercompany, financing, payroll, with
   exceptions such as tax *advice*) → the LLM's confident non-procurement flag.

## Matching and de-duplication

Suppliers and items are grouped with union-find. Suppliers are linked through shared
vendor IDs and a normalised name (upper case, no punctuation, no legal form, no spaces,
which merges OCR splits such as `PRAH AVEN`), then a strict fuzzy pass. Items are linked
by non-generic item code and by normalised description (decimal comma → point, OCR
digit/letter folding), then a fuzzy pass inside each supplier that requires every number
to agree (`1500MM` never merges with `1800MM`). Catch-all codes are detected from the data.
A matched invoice line inherits its goods receipt's item one-way.

## Classifying into an unseen tree

The tree is read at run time; the LLM answers with leaf names constrained by a JSON schema,
so only names from the tree file can be written. The unit is the **item signature**
(supplier group + item group): A has 5,706 and B 12,743, against 42k and 38k lines.

- **Known:** signatures with labelled twins take their majority label.
- **LLM:** the remaining signatures that make up 99% of describable spend go to the LLM,
  200 per call, with the whole tree, a short list of labelled examples when labels exist,
  and each batch's suppliers with their main items by spend (names are anonymised, so what
  a supplier sells is the best clue for vague lines). Batches go highest spend first, so if
  the LLM stops partway, the high-spend items already have its answers.
- **Local:** a TF-IDF (character n-grams of the description, words of supplier and
  account context) + logistic-regression model trained on the known and LLM answers
  labels the long tail. With no labels and no LLM, a lexical match to leaf names seeds it.

## Measured accuracy

| Setting | L1 | L2 | L3 | L3 by spend |
|---|---|---|---|---|
| A, labels hidden, LLM at 95% coverage (37,029 ERP lines) | 86.2% | 73.7% | 63.1% | 69.5% |
| A, unseen items, labels + local model, no LLM (8,253 lines) | 99.3% | 79.8% | 76.9% | 66.0% |
| B, unseen seed items, seeds + local model, no LLM (527 lines) | 91.3% | 90.9% | 85.8% | 94.1% |
| B, seeds hidden, LLM (2,550 seed lines; 4,000 LLM items) | 91.3% | 90.6% | 84.5% | 90.5% |
| B, same, simulated at 95% coverage (1,583 LLM items) | | | 70.9% | 89.9% |
| B, same, simulated at 99% coverage, the default (2,843 LLM items) | | | 78.7% | 90.3% |
| A, new delivery: last 20% of A as new data, the rest as labelled reference (7,406 lines) | 94.8% | 92.4% | 90.6% | 92.1% |
| B, new delivery: last 20% of B as new data, the rest as labelled reference (seed lines) | 96.5% | 96.1% | 93.8% | 98.8% |

A level counts only if every level above it is also right. Most of A's zero-label L3
error is a labelling convention that the tree's wording does not reveal: A's labels use the
general leaf named like its parent (*Roof Bolters > Roof Bolters*) where the LLM picks the
specific sibling (*RB Maintenance Spares*, used for 3 labelled lines). A prompt rule asking
for general leaves did not change this; with labels, the known and local routes learn it.

Other measurements on A:
- **Supplier and item grouping:** pairwise F1 0.98 and 0.99 against A's golden keys. This uses
  A's key conventions: the client keeps suppliers without a vendor ID, and catalogue items,
  under one key each (DECISIONS.md). Without its catch-all groups the F1 scores are 1.00 and 0.98.
- **Synthetic matching test:** of 3,000 ERP lines copied with tool-style noise, 99.7% were
  found, all on the right receipt or an identical one.
- **Chance test:** with tool amounts shuffled, under 1% of lines still match.
- **Exclusions on B:** agree with all 3,000 seeds (training data for the rules).

## Runtime and LLM usage

- **Runtime:** A (65,549 records) with labels hidden and the LLM: 5 minutes; without the
  LLM, about 1 minute. Peak memory is under 0.7 GB.
- **LLM:** GPT Sol 5.6 through AgentRouter via the Codex CLI, reasoning effort `low`. A
  200-item call uses about 25k input and 15k output tokens at the median (53 calls), most
  of the output being reasoning. Codex adds about 10k input tokens of its own to every call
  and the gateway caches nothing, hence 200 items per call.
- **Per 1,000 lines:** with no labels the LLM labels about 2,800 items: about 15 calls for
  B's 41k lines, **about 14k tokens per 1,000 lines** (A: 14 calls, 8k). A new delivery of a
  known company needed one call, since most of its items already have labels. Responses are
  cached on disk, so a re-run makes no new calls.
- **Without a key** the tool runs locally, but an unseen tree then drops to 24.5% L3 on B's
  seeds.
- **Reasoning effort:** `minimal` was rejected (52% seed agreement against 99.9% for `low`).
  More reasoning did not help either: on A's 200 largest items (71% of its spend), labels
  hidden, L3 by spend was 77.2% at `low`, 78.3% at `medium` and 77.8% at `high`, and `high`
  took 221 s against 67 s. At one setting, reasoning per call ranges from 1k to 26k tokens,
  so the gaps are within one call's variation. `low` is kept (`experiments/effort.py`).

## AI tools used

Claude Code helped write this code. GPT Sol 5.6 (via AgentRouter and the Codex CLI)
classifies items at run time.
