# Decisions

What the data showed and what the tool does about it. Numbers come from profiling the
kit files. Where the brief was silent, the assumption is stated.

## Company A: two systems, ZAR

| Finding | Evidence | Decision |
|---|---|---|
| Amounts line up exactly across systems | 85% of tool lines have a net-of-VAT amount equal to an ERP line value | Match on net amount to the cent (±1 cent for rounding when VAT is removed) |
| PO references are unreliable | 18% of tool lines have no PO; 24% of the POs given are not in the ERP file; four formats (`4510…`, `PO#…`, `PO …`, `…/10.0`) | Three matching tiers: PO + line + quantity + amount; PO + amount; supplier + amount + similar text |
| Near-amounts on one PO are different deliveries | Same PO and text, 33.62 t vs 33.4 t | Never match on approximate amounts |
| Re-exports carry the same invoice number | Invoice `100059` appears four times with OCR-varied text | Re-export = same supplier, invoice-number core, amount and quantity, text similarity ≥ 60 |
| Identical lines under different invoice numbers are repeat purchases | Such twins span a median of 79 days; the ERP itself has 2,965 identical repeat receipts | Count them; they are not duplicates |
| Invoice numbers are formatted many ways | `INV/2023/759857`, `TAX INV 759857`, `IN0759857`, `SI-2023-759857` | Compare the last digit run without leading zeros |
| VAT notation varies | `15`, `0.15`, `15%`, `0`, blank | Parse all; blank means 15% (brief); `0` means zero-rated, so net = gross |
| Dates come in three formats, 65% blank | `2024-03-27`, `27/03/2024`, `27-Mar-24` | Parse in that order; dd/mm (values such as `31/12/2022` rule out mm/dd) |
| ERP `document_currency` sometimes says EUR or CHF | 16 lines | `value_zar` is already ZAR (brief); the column is ignored |
| ERP has negative lines | 434 lines, −56M ZAR; 278 have an exact positive twin on the same PO line | Kept as negative spend so reversals net to zero |
| Supplier names are typed many ways | 702 variants in the tool against 193 in the ERP | Upper case, drop punctuation and legal forms, remove all spaces; then strict fuzzy (ratio ≥ 94, length ≥ 8, digits equal) |
| One material code is a catch-all | `M1000032` carries hundreds of unrelated free-text descriptions | A code with ≥ 5 different descriptions and none above half, or with ≥ 20 suppliers, is not used to group items |
| The same description sometimes has two codes | The golden item key merges them in 16 such cases and splits them in 14 | Same normalised description = same item |
| Cancelled or unapproved invoices could match a receipt | 503 tool lines are cancelled, rejected or on hold | Status is checked first, and only approved or paid tool lines take part in matching, so they never use up a receipt |

### A's golden keys leave out everything outside the client's master data

One `supplier_key` in `labels_A.csv` covers 20,891 lines and 122 unrelated suppliers: exactly
the suppliers that never have a vendor ID (20,492 lines; 399 exceptions). One `item_key`
covers 2,661 lines: the catalogue and free-text lines that carry a placeholder material code
(2,659 of them; every one of its 192 codes is always inside it). Lines with no material code
have no item key at all.

So the client's golden lists hold only master-data suppliers and materials. To match them,
A's config switches on two `key_conventions`: suppliers never seen with a vendor ID share one
key, and catalogue / free-text items share one key. A supplier seen with an ID in the
reference delivery keeps its own key, because a new delivery can carry no IDs at all (the
simulated one below did). The conventions change only the output keys;
classification still uses the real supplier and item groups. Other companies keep one key per
real supplier and item, as the brief asks.

| Pairwise score against A's golden keys | Supplier F1 | Item F1 |
|---|---|---|
| One key per real supplier / item | 0.18 | 0.56 |
| With A's key conventions | 0.98 | 0.99 |
| Without the catch-all groups (either way) | 1.00 | 0.98 |
| New delivery (last 20% of A), with conventions | 0.99 | 0.98 |

## Company B: eight ERPs, 25 currencies, USD

| Finding | Evidence | Decision |
|---|---|---|
| USD lines carry a misleading rate | 1,279 USD lines have a rate near 17 | Reporting-currency lines ignore the rate (brief) |
| Some line rates belong to another currency | 59 EUR and CHF lines have a rate near 18 where about 0.9 is right | A line rate more than 2× off the FX table for that month is replaced by the table rate (61 lines) |
| Some currencies never carry a rate | CAD, HKD, MMK, THB, RCN: 3,200 lines, plus 143 blank rates elsewhere | FX table for the invoice month (brief), nearest month if that one is missing |
| Some invoice dates are impossible | 2018 to 2027 while posting dates are all 2023–24 | A date more than 366 days from the posting date is flagged (149 lines) and the posting month is used for FX |
| No status column | | Every line is approved spend unless another rule excludes it |
| Intercompany accounts | All 364 seed lines on `IC05`, `IC06`, `I/C Payable`, `IC Payables`… are excluded | B config rule on `gl_account_description` |
| Tax lines | `State & Local Taxes -`, `Sales Tax Expense -`, `GST Input Tax -` (86 seed lines, all excluded) | Generic term list; tax advice and tax services are not hits |
| Credits and reversals are counted | 20 of 25 negative seed lines and all net-to-zero pairs in the seed are included | Kept, as negative spend |
| Exact duplicate rows | 613 rows identical to another in every column except `line_id`, mostly in one ERP | `DUPLICATE`, first one counted (same reasoning as the brief's rule for A's tool) |
| 41% of descriptions are blank | | Such lines are grouped per supplier and account and classified from supplier and account context |
| No supplier names | Vendor IDs only | `supplier_key` is the vendor ID; the same ID in two ERPs is treated as one supplier |
| Seed labels never cover blank descriptions | All 167 blank-description seed lines are excluded ones | Blank lines are not scored on "clear-cut lines"; they still get a category from their supplier |

Exclusions agree with all 3,000 seed labels. The B rules were written after reading the
seeds, so this is a training score; a new company is the real test of the generic list.

## Classification

| Finding | Evidence | Decision |
|---|---|---|
| Items repeat across many lines | A: 41,984 included lines but 5,706 supplier+item signatures; B: 37,935 and 12,743 | Classify signatures; copy the answer to every line |
| Spend is concentrated | 1,405 A signatures carry 95% of spend | The LLM labels signatures up to 99% of describable spend; a local model trained on its answers does the tail. 99% against 95%: on B +7.8 pts L3 by lines and +0.4 by spend; on A 69.1% L3 by spend both |
| Numbered leaf IDs cause errors | Half of one batch came back with IDs shifted by 20–22 places | The LLM answers with unique leaf names, schema-checked and mapped back to IDs |
| Low reasoning is needed | `minimal` effort: 52% seed agreement on B's top 200 items; `low`: 99.9% | Reasoning effort `low` |
| More reasoning does not pay | A's top 200 items, labels hidden: L3 by spend 77.2% `low`, 78.3% `medium`, 77.8% `high` (one call each). Reasoning per call ranges 1k–26k tokens at the same setting | Kept `low`: the gaps are within one call's variation |
| A's labels prefer general leaves | *Roof Bolters > Roof Bolters* for lines the LLM files under *RB Maintenance Spares* (3 labelled lines) | Not tuned for: it is a client convention the tree does not state. Labels, when present, teach it through the known and local routes |
| Supplier names are anonymised | Names like `PRAHAVEN (PTY) LTD` say nothing | Each LLM batch lists its suppliers' main items by spend |
| Blank descriptions | 41% of B's lines | Not sent to the LLM; the local model labels them from supplier and account context |
| The LLM can fail | The gateway once refused every call partway through a run | After 3 failures in a row the run stops calling the LLM and the local model labels everything left. Batches go highest spend first, so what the LLM did answer is the high-spend part |

## A simulated new company

B's lines were re-laid out as a new client might send them (semicolon-separated, decimal
commas, day/month/year dates, FX quoted as USD per unit, the tree as one path column) and run
from a config written from `configs/TEMPLATE.yaml`, with no code change.

| Finding | Evidence | Decision |
|---|---|---|
| Exact-duplicate detection depends on the columns present | The re-laid-out file had no invoice line number, so 3,230 genuine lines looked identical and were marked `DUPLICATE` | `duplicate_columns` and `exact_duplicates` in the source config; check for a line-number column before trusting the default |
| The free fallback is weak on an unseen tree | No LLM and no labels: 24.5% L3 agreement with B's seeds | An unseen tree needs the LLM; the run warns when LLM calls stop |

## General assumptions

- **Negative and zero amounts are included.** No exclusion code fits them.
- **Exclusion order:** status, then duplicate, then company rules, then generic terms. Status
  is a stated property of a record; duplication is inferred.
- **Excluded rows keep their supplier and item keys**, spend is 0, categories blank, and
  confidence is the confidence in the exclusion.
- **The LLM can flag non-procurement lines.** Its flag excludes a line only at confidence ≥ 0.9.
- **What leaves the machine:** description text, supplier name, account context and the
  category tree go to the LLM gateway. No amounts, IDs or dates are sent.
