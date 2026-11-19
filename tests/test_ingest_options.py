"""Config options for file layouts the kit did not show, all exercised through one run."""

import pandas as pd
import pytest
import yaml

from spendcube.pipeline import run
from spendcube.taxonomy import Taxonomy

LINES = """doc;pos;vendor;text;gross;vat;dc;ccy;usd_per_ccy;date
D1;1;Acme;Cleaning service;1.150,00;150,00;D;EUR;1,10;15.02.2024
D1;2;Acme;Cleaning service credit;115,00;15,00;C;EUR;1,10;15.02.2024
D2;1;Bolt;Hex bolts M8;50,00;0,00;D;USD;;20.02.2024
D2;1;Bolt;Hex bolts M10;60,00;0,00;D;USD;;20.02.2024
"""
TREE = """L1 code,L1 name,L2 code,L2 name
10,Facilities,1010,Cleaning
20,MRO,2010,Fasteners
"""


@pytest.fixture
def company(tmp_path, monkeypatch):
    (tmp_path / "lines.csv").write_text(LINES, encoding="utf-8")
    (tmp_path / "tree.csv").write_text(TREE)
    cfg = {
        "company": "Y", "reporting_currency": "USD", "taxonomy": "tree.csv",
        "taxonomy_columns": ["L1 name", "L2 name"],
        "sources": {"ap": {
            "file": "lines.csv", "sep": ";", "decimal": ",", "date_formats": ["%d.%m.%Y"],
            "amount_is": "gross", "rate_direction": "reporting_per_unit",
            "credit": {"column": "dc", "values": ["C"]},
            "columns": {"record_id": ["doc", "pos"], "supplier_name": "vendor", "description": "text",
                        "amount": "gross", "tax_amount": "vat", "currency": "ccy", "fx_rate": "usd_per_ccy",
                        "date": "date"}}},
    }
    (tmp_path / "Y.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("SPENDCUBE_LLM", "none")
    monkeypatch.setenv("SPENDCUBE_WANDB", "0")
    return tmp_path


def test_layout_options(company):
    metrics = run(str(company / "Y.yaml"), str(company), str(company / "out.csv"))
    out = pd.read_csv(company / "out.csv")
    assert out["record_id"].tolist() == ["D1-1", "D1-2", "D2-1", "D2-1"]  # repeated ID kept, not fatal
    spend = out["spend"].tolist()
    assert spend[0] == pytest.approx(1000 * 1.10)  # net of VAT amount, USD per EUR inverted
    assert spend[1] == pytest.approx(-100 * 1.10)  # credit line is negative
    assert spend[2:] == [50.0, 60.0]
    assert set(out["l1"]) <= {"Facilities", "MRO"}  # names, not code columns
    assert metrics["output_problems"] == 1  # the repeated record ID is reported


def test_taxonomy_single_path_column(tmp_path):
    path = tmp_path / "tree.csv"
    path.write_text("path\nIT > Hardware > Laptops\nIT > Software\n")
    tax = Taxonomy.from_csv(path, separator=" > ")
    assert [leaf.path for leaf in tax.leaves] == [("IT", "Hardware", "Laptops"), ("IT", "Software")]


def test_duplicate_columns(tmp_path, monkeypatch):
    (tmp_path / "lines.csv").write_text("id,inv,line,vendor,text,amt\n1,I1,1,A,bolt,5\n2,I1,2,A,bolt,5\n3,I1,2,A,bolt,5\n")
    (tmp_path / "tree.csv").write_text("l1\nMRO\n")
    base = {"company": "D", "reporting_currency": "USD", "taxonomy": "tree.csv",
            "sources": {"s": {"file": "lines.csv", "columns": {"record_id": "id", "supplier_name": "vendor",
                                                             "description": "text", "amount": "amt"}}}}
    monkeypatch.setenv("SPENDCUBE_LLM", "none")
    monkeypatch.setenv("SPENDCUBE_WANDB", "0")
    results = {}
    for label, extra in (("default", {}), ("narrow", {"duplicate_columns": ["inv", "vendor", "text", "amt"]}),
                         ("off", {"exact_duplicates": False})):
        cfg = {**base, "sources": {"s": {**base["sources"]["s"], **extra}}}
        (tmp_path / "D.yaml").write_text(yaml.safe_dump(cfg))
        run(str(tmp_path / "D.yaml"), str(tmp_path), str(tmp_path / "out.csv"))
        results[label] = pd.read_csv(tmp_path / "out.csv")["include"].tolist()
    assert results == {"default": [1, 1, 0], "narrow": [1, 0, 0], "off": [1, 1, 1]}
