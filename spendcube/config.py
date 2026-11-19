"""Company configuration: schema, loading and validation.

A company is fully described by one YAML file. Paths inside it are relative to
the data directory given on the command line, so the same config works on new
deliveries of the same layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Every source is mapped onto these columns. Anything else the classifier or the
# exclusion rules should see is listed under `context` and kept by its own name.
CANONICAL_FIELDS = (
    "record_id",
    "supplier_name",
    "supplier_id",
    "description",
    "item_code",
    "quantity",
    "unit",
    "amount",
    "currency",
    "tax_rate",
    "tax_amount",
    "fx_rate",
    "date",
    "fallback_date",
    "status",
    "po_ref",
    "po_line",
    "invoice_ref",
)

EXCLUSION_CODES = ("CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatusConfig(_Strict):
    approved: list[str]
    cancelled: list[str] = []


class CreditConfig(_Strict):
    column: str  # e.g. a debit/credit indicator
    values: list[str]  # rows with these values get a negative amount


RateDirection = Literal["units_per_reporting", "reporting_per_unit"]


class SourceConfig(_Strict):
    file: str
    sep: str = ","
    encoding: str = "utf-8-sig"
    sheet: str | int | None = None  # for .xlsx / .xls files
    id_prefix: str = ""
    # canonical field -> source column; a list takes the first non-empty value
    # (for record_id, a list is joined with '-' to build an ID)
    columns: dict[str, str | list[str]]
    context: list[str] = []
    amount_is: Literal["net", "gross"] = "net"
    default_tax_rate: float = 0.0  # used when the tax rate is blank (gross amounts only)
    currency: str | None = None  # constant currency when the file has no currency column
    rate_direction: RateDirection = "units_per_reporting"  # how the line FX rate is quoted
    credit: CreditConfig | None = None
    date_formats: list[str] = ["%Y-%m-%d"]
    decimal: Literal[".", ","] = "."
    status: StatusConfig | None = None  # None: the file holds approved spend only
    # Rows equal on these columns are one record exported twice. Default: every column but the ID.
    # Narrow files without a line number need an explicit list (or exact_duplicates: false),
    # otherwise two genuine identical lines on one invoice would look like a re-export.
    duplicate_columns: list[str] | None = None
    exact_duplicates: bool = True

    def source_columns(self) -> set[str]:
        cols = {c for v in self.columns.values() for c in ([v] if isinstance(v, str) else v)}
        extra = set(self.duplicate_columns or []) | ({self.credit.column} if self.credit else set())
        return cols | set(self.context) | extra

    @model_validator(mode="after")
    def _check_columns(self) -> SourceConfig:
        unknown = set(self.columns) - set(CANONICAL_FIELDS)
        if unknown:
            raise ValueError(f"unknown canonical fields {sorted(unknown)}; allowed: {CANONICAL_FIELDS}")
        for required in ("record_id", "amount"):
            if required not in self.columns:
                raise ValueError(f"column mapping must include '{required}'")
        clash = set(self.context) & set(CANONICAL_FIELDS)
        if clash:
            raise ValueError(f"context columns clash with canonical names: {sorted(clash)}")
        if self.status and "status" not in self.columns:
            raise ValueError("status rules given but no 'status' column mapped")
        return self


class MatchingConfig(_Strict):
    primary: str
    secondary: str


class FxConfig(_Strict):
    file: str
    currency_column: str = "currency"
    month_column: str = "month"
    rate_column: str
    rate_direction: RateDirection = "units_per_reporting"


class RuleConfig(_Strict):
    column: str  # a canonical field or a context column
    pattern: str  # case-insensitive regular expression
    reason: Literal["NON_PROCUREMENT", "CANCELLED", "NOT_APPROVED"] = "NON_PROCUREMENT"


class MatchRule(_Strict):
    column: str
    pattern: str  # case-insensitive regular expression


class KeyConventions(_Strict):
    """How the client's golden keys treat records outside its master data, learned from its labels.

    The internal supplier and item groups (used for classification) are unchanged; only the
    output keys follow the convention.
    """
    shared_supplier_without_id: bool = False  # suppliers never seen with a supplier ID share one key
    shared_item: MatchRule | None = None  # item groups whose records mostly match share one key


class LabelsConfig(_Strict):
    file: str
    id_column: str = "record_id"
    id_prefix: str = ""
    category_columns: list[str]
    supplier_key_column: str | None = None
    item_key_column: str | None = None
    include_column: str | None = None
    reason_column: str | None = None


class CompanyConfig(_Strict):
    company: str
    reporting_currency: str
    taxonomy: str
    taxonomy_columns: list[str] | None = None  # name columns in level order, if the file has others
    taxonomy_separator: str | None = None  # for a tree given as one path column, e.g. ' > '
    sources: dict[str, SourceConfig]
    matching: MatchingConfig | None = None
    fx: FxConfig | None = None
    labels: LabelsConfig | None = None
    exclusion_rules: list[RuleConfig] = []
    key_conventions: KeyConventions | None = None
    # Context columns worth showing to the classifier (account text, material type, ...).
    classification_context: list[str] = []

    @model_validator(mode="after")
    def _check_refs(self) -> CompanyConfig:
        if self.matching:
            for name in (self.matching.primary, self.matching.secondary):
                if name not in self.sources:
                    raise ValueError(f"matching refers to unknown source '{name}'")
        context = {c for s in self.sources.values() for c in s.context}
        missing = set(self.classification_context) - context
        if missing:
            raise ValueError(f"classification_context not listed as source context: {sorted(missing)}")
        return self


def load_config(path: str | Path) -> CompanyConfig:
    with open(path, encoding="utf-8") as fh:
        return CompanyConfig.model_validate(yaml.safe_load(fh))


def load_yaml(name: str) -> dict:
    """Load a shared settings file from the configs directory."""
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
