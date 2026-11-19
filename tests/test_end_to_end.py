"""A made-up company from a new industry: new config and tree, no code changes, no labels."""

import json

import pandas as pd
import yaml

from spendcube.classify import build_signatures, llm_labels, record_features
from spendcube.config import load_config, load_yaml
from spendcube.output import COLUMNS
from spendcube.pipeline import run
from spendcube.taxonomy import Taxonomy

LINES = """inv_line,vendor,item_text,amt,ccy,rate,inv_date,state,acct
H1,MedSupply Inc,Surgical gloves nitrile M,1200.00,USD,,2024-02-01,PAID,Supplies
H2,MEDSUPPLY INC.,Surgical gloves nitrile L,800.00,USD,,2024-02-03,PAID,Supplies
H3,CleanCo,Floor cleaning service Feb,500.00,EUR,0.92,2024-02-28,PAID,Services
H4,CleanCo,Floor cleaning service Mar,500.00,EUR,,2024-03-30,PAID,Services
H5,State Revenue Service,Sales Tax Expense -,90.00,USD,,2024-03-01,PAID,Tax
H6,MedSupply Inc,Surgical gloves nitrile M,1200.00,USD,,2024-02-01,VOID,Supplies
H7,ITPartner,Laptop 14in,1500.00,USD,,03/04/2024,PAID,IT
H8,ITPartner,Laptop 14in,1500.00,USD,,03/04/2024,PAID,IT
"""
TREE = """level_1,level_2,level_3,level_4
Clinical,Consumables,Gloves,Nitrile Gloves
Facilities,Services,Cleaning,
IT,Hardware,,
"""


def make_company(tmp_path):
    (tmp_path / "lines.csv").write_text(LINES)
    (tmp_path / "tree.csv").write_text(TREE)
    (tmp_path / "fx.csv").write_text("currency,month,units_per_usd\nEUR,2024-02,0.92\nEUR,2024-03,0.90\n")
    cfg = {
        "company": "Z", "reporting_currency": "USD", "taxonomy": "tree.csv",
        "sources": {"ap": {
            "file": "lines.csv", "date_formats": ["%Y-%m-%d", "%d/%m/%Y"],
            "columns": {"record_id": "inv_line", "supplier_name": "vendor", "description": "item_text",
                        "amount": "amt", "currency": "ccy", "fx_rate": "rate", "date": "inv_date",
                        "status": "state"},
            "context": ["acct"], "status": {"approved": ["PAID"], "cancelled": ["VOID"]}}},
        "fx": {"file": "fx.csv", "rate_column": "units_per_usd"},
        "classification_context": ["acct"],
    }
    path = tmp_path / "Z.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_new_company_runs_from_config_only(tmp_path, monkeypatch):
    config = make_company(tmp_path)  # H8 repeats H7 under its own ID: an exact duplicate
    monkeypatch.setenv("SPENDCUBE_LLM", "none")
    monkeypatch.setenv("SPENDCUBE_WANDB", "0")
    out_path = tmp_path / "out.csv"

    metrics = run(str(config), str(tmp_path), str(out_path))
    out = pd.read_csv(out_path, dtype={"exclusion_reason": str}).fillna("")

    assert list(out.columns) == COLUMNS
    assert len(out) == 8 and metrics["output_problems"] == 0
    reason = dict(zip(out["record_id"], out["exclusion_reason"]))
    assert reason["H6"] == "CANCELLED"
    assert reason["H5"] == "NON_PROCUREMENT"
    assert reason["H8"] == "DUPLICATE"
    spend = dict(zip(out["record_id"], out["spend"]))
    assert spend["H3"] == round(500 / 0.92, 2)  # line rate
    assert spend["H4"] == round(500 / 0.90, 2)  # FX table, March
    sup = dict(zip(out["record_id"], out["supplier_key"]))
    assert sup["H1"] == sup["H2"]
    inc = out[out["include"] == 1]
    assert (inc["l1"] != "").all()
    assert (out.loc[out["include"] == 0, "spend"] == 0).all()


class FakeLLM:
    """Answers like the real backends: a JSON object with one result per item."""

    def __init__(self, leaf_for):
        self.leaf_for = leaf_for
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        items = [line for line in prompt.splitlines() if line.startswith('{"id"')]
        results = []
        for line in items:
            card = json.loads(line)
            results.append({"id": card["id"], "leaf": self.leaf_for(card["description"]),
                            "confidence": 0.9, "non_procurement": False})
        results.append({"id": "s999", "leaf": "Hardware", "confidence": 1.0, "non_procurement": False})  # bogus id
        return {"results": results}


def test_llm_route_parses_and_validates(tmp_path):
    config = make_company(tmp_path)
    cfg = load_config(config)
    settings = load_yaml("defaults.yaml")
    tax = Taxonomy.from_csv(tmp_path / "tree.csv")
    df = pd.DataFrame({
        "record_id": ["a", "b"], "supplier_name": ["MedSupply", "ITPartner"], "supplier_id": None,
        "description": ["gloves", "laptop"], "acct": ["Supplies", "IT"], "spend_value": [10.0, 20.0],
        "supplier_key": ["MEDSUPPLY", "ITPARTNER"], "item_key": ["GLOVES", "LAPTOP"],
    })
    feats = record_features(df, cfg, settings)
    sigs, _ = build_signatures(df, feats)
    ids = {leaf.path[0]: leaf.id for leaf in tax.leaves}
    names = tax.answer_labels()
    fake = FakeLLM(lambda d: names[ids["Clinical"]] if "glove" in d else names[ids["IT"]])
    out = llm_labels(sigs, tax, feats.iloc[:0].assign(leaf=[]), fake, settings)
    assert len(out) == 2  # the bogus id is dropped
    assert dict(zip(out["sig"], out["leaf"])) == {0: ids["Clinical"], 1: ids["IT"]}
    assert "Category tree" in fake.prompts[0]


def test_llm_batches_go_in_spend_order(tmp_path):
    cfg = load_config(make_company(tmp_path))
    settings = load_yaml("defaults.yaml")
    settings["llm"]["batch_size"], settings["llm"]["concurrency"] = 2, 1
    tax = Taxonomy.from_csv(tmp_path / "tree.csv")
    df = pd.DataFrame({
        "record_id": ["a", "b", "c"], "supplier_name": ["A", "B", "C"], "supplier_id": None,
        "description": ["cheap item", "big item", "mid item"], "acct": "x", "spend_value": [1.0, 100.0, 50.0],
        "supplier_key": ["A", "B", "C"], "item_key": ["CHEAP", "BIG", "MID"],
    })
    feats = record_features(df, cfg, settings)
    sigs, _ = build_signatures(df, feats)
    fake = FakeLLM(lambda d: next(iter(tax.answer_labels().values())))
    llm_labels(sigs, tax, feats.iloc[:0].assign(leaf=[]), fake, settings)
    assert "big item" in fake.prompts[0] and "mid item" in fake.prompts[0]
    assert "cheap item" in fake.prompts[1]
