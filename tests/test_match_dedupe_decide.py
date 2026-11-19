import pandas as pd

from spendcube.config import CompanyConfig
from spendcube.decide import decide, term_hits
from spendcube.dedupe import item_keys, supplier_keys
from spendcube.match import collapse_reexports, match_systems
from spendcube.taxonomy import Taxonomy

SETTINGS = {
    "supplier_fuzzy_threshold": 94, "supplier_fuzzy_min_len": 8, "item_fuzzy_threshold": 90,
    "item_fuzzy_min_len": 8, "generic_code_min_descs": 3, "generic_code_max_top_share": 0.5,
    "generic_code_min_suppliers": 20, "uninformative_descriptions": ["product", "no object"],
}


def records(rows):
    cols = ["record_id", "supplier_name", "supplier_id", "description", "item_code", "po_ref", "po_line",
            "invoice_ref", "net_amount", "quantity_num"]
    df = pd.DataFrame(rows, columns=cols)
    df["supplier_key"] = supplier_keys(df, SETTINGS)
    return df


def test_matching_tiers_and_one_to_one():
    erp = records([
        ["E1", "ACME (PTY) LTD", None, "VALVE;2IN", "M1", "4510000001", "10", None, 100.00, 1],
        ["E2", "ACME (PTY) LTD", None, "VALVE;2IN", "M1", "4510000001", "10", None, 100.00, 1],
        ["E3", "BOLTMAKERS", None, "ROD ANCHR;1800MM", "M2", "4510000002", "10", None, 250.50, 5],
    ])
    tool = records([
        ["T1", "Acme Pty Ltd", None, "VALVE;2IN", None, "PO#4510000001", None, "INV1", 100.00, 1],
        ["T2", "ACME", None, "VA1VE;2IN", None, "4510000001/10.0", None, "INV2", 100.00, 1],
        ["T3", "BOLT MAKERS", None, "ROD ANCHR;1800MM", None, None, None, "INV3", 250.51, 5],  # no PO, 1 cent off
        ["T4", "ACME", None, "VALVE;2IN", None, "4510000001", None, "INV4", 100.00, 1],  # nothing left to match
    ])
    tool.index = tool.index + 10
    both = pd.concat([erp, tool])
    both["supplier_key"] = supplier_keys(both, SETTINGS)
    m = match_systems(both.loc[erp.index], both.loc[tool.index], 50)
    pairs = dict(zip(m["secondary_index"], m["primary_index"]))
    assert set(pairs) == {10, 11, 12}
    assert len(set(pairs.values())) == 3  # one-to-one
    assert pairs[12] == 2 and m.set_index("secondary_index").loc[12, "tier"] == 3


def test_reexport_needs_same_invoice_number():
    tool = records([
        ["T1", "DUNAX", None, "38MM TWO PRONG", None, None, None, "100059", 50.0, 1],
        ["T2", "DUNAX (PTY) LTD", None, "38MM TW0 PRONG", None, None, None, "100059", 50.0, 1],
        ["T3", "DUNAX", None, "38MM TWO PRONG", None, None, None, "777777", 50.0, 1],  # repeat purchase
    ])
    dup = collapse_reexports(tool, pd.Series(True, index=tool.index), 60)
    assert dup.tolist()[1] == 0  # position of the surviving export
    assert pd.isna(dup.iloc[0]) and pd.isna(dup.iloc[2])


def test_item_keys_codes_descriptions_and_one_way_links():
    df = records([
        ["1", "A", None, "MESH;WLD,3M,WD:2,2M", "M1", None, None, None, 1, 1],
        ["2", "A", None, "MESH;WLD,3M,WD:2.2M", "M9", None, None, None, 1, 1],  # same item, other code
        ["3", "A", None, "ROD;1500MM", "M2", None, None, None, 1, 1],
        ["4", "A", None, "ROD;1800MM", "M3", None, None, None, 1, 1],
        ["5", "A", None, "R0D;1800MM", None, None, None, None, 1, 1],  # noisy invoice text
        ["6", "A", None, "Product", None, None, None, None, 1, 1],
    ])
    keys = item_keys(df, SETTINGS, [], links=[(4, 2)])  # record 5 was matched to record 3's receipt
    assert keys[0] == keys[1]
    assert keys[2] != keys[3]
    assert keys[4] == keys[2]  # takes the receipt's item...
    assert keys[3] != keys[2]  # ...without merging the 1800MM item into it
    assert keys[5].startswith("UNSPECIFIED")


def test_supplier_ids_link_names():
    df = records([
        ["1", "ASHAX (PTY) LTD", "V1", "x", None, None, None, None, 1, 1],
        ["2", "ASHAX (PTY) LTD", "V2", "x", None, None, None, None, 1, 1],
        ["3", None, "V2", "x", None, None, None, None, 1, 1],
        ["4", "OTHER CO", "V3", "x", None, None, None, None, 1, 1],
    ])
    assert df["supplier_key"].nunique() == 2


def test_decide_order_and_terms():
    cfg = CompanyConfig.model_validate({
        "company": "T", "reporting_currency": "USD", "taxonomy": "t.csv",
        "sources": {"s": {"file": "f.csv", "columns": {"record_id": "id", "amount": "a", "status": "st",
                                                        "description": "d"},
                          "context": ["gl"], "status": {"approved": ["Paid"], "cancelled": ["Cancelled"]}}},
        "exclusion_rules": [{"column": "gl", "pattern": "^IC\\d+$"}],
        "classification_context": ["gl"],
    })
    df = pd.DataFrame({
        "source": "s", "status": ["Paid", "Cancelled", "On Hold", "Paid", "Paid", "Paid", "Paid"],
        "description": ["bolts", "bolts", "bolts", "bolts", "Sales Tax Expense -", "Sales tax advisory services", "x"],
        "gl": ["GL1", "GL1", "GL1", "IC06", "GL1", "GL1", "GL1"], "supplier_name": None,
    })
    dup = pd.Series([pd.NA] * 6 + ["R1"], dtype="string")
    out = decide(df, cfg, {"groups": [{"name": "tax", "any": [" sales tax"], "unless": [" advis"]}]}, dup,
                 pd.Series([0.9] * 7))
    assert out["exclusion_reason"].fillna("").tolist() == [
        "", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "NON_PROCUREMENT", "", "DUPLICATE"]


def test_term_unless():
    df = pd.DataFrame({"description": ["State & Local Taxes -", "Tax Services -"], "supplier_name": None})
    terms = {"groups": [{"name": "tax", "any": [" local taxes"], "unless": [" service"]}]}
    assert term_hits(df, terms, ["description"]).tolist() == [True, False]


def test_taxonomy_leaves_and_uneven_depth(tmp_path):
    path = tmp_path / "tree.csv"
    path.write_text("l1,l2,l3,l4\nA,,,\nA,A1,,\nA,A1,A1a,\nB,B1,,\nC,,,\n")
    tax = Taxonomy.from_csv(path)
    assert [leaf.path for leaf in tax.leaves] == [("A", "A1", "A1a"), ("B", "B1"), ("C",)]
    cols = tax.level_columns(pd.Series([tax.leaves[1].id, None]))
    assert cols.iloc[0].tolist() == ["B", "B1", "", ""]
    assert cols.iloc[1].tolist() == ["", "", "", ""]


def test_answer_labels_are_unique_names(tmp_path):
    path = tmp_path / "tree.csv"
    path.write_text("l1,l2,l3\nA,Spares,Filters\nB,Spares,Filters\nC,Services,Cleaning\n")
    labels = Taxonomy.from_csv(path).answer_labels()
    assert sorted(labels.values()) == ["A > Spares > Filters", "B > Spares > Filters", "Cleaning"]


def test_llm_selection_ignores_vague_spend():
    from spendcube.classify import select_for_llm
    sigs = pd.DataFrame({"sig": range(5), "spend": [1000, 50, 30, 15, 5],
                         "informative": [False, True, True, True, True]})
    chosen = select_for_llm(sigs, set(), {"llm": {"spend_coverage": 0.9, "max_items": 10}})
    # the blank 1000 is not reachable; 90% of the informative 100 needs the top three
    assert chosen["sig"].tolist() == [1, 2, 3]


def test_output_key_conventions():
    from spendcube.config import KeyConventions
    from spendcube.dedupe import SHARED_ITEM, SHARED_SUPPLIER, output_keys
    df = pd.DataFrame({
        "supplier_key": ["S1", "S1", "S2", "S3"], "supplier_id": ["V1", None, None, None],
        "supplier_name": ["ONE", "ONE", "TWO", "THREE"],
        "item_key": ["I1", "I2", "I2", "I3"], "spend_type": ["Material", "Free Text", None, "Catalogue"],
    })
    conv = KeyConventions.model_validate({"shared_supplier_without_id": True,
                                          "shared_item": {"column": "spend_type", "pattern": "^(?:Free Text|Catalogue)$"}})
    sup, item = output_keys(df, conv)
    assert sup.tolist() == ["S1", "S1", SHARED_SUPPLIER, SHARED_SUPPLIER]  # S1 has an ID on one record
    assert item.tolist() == ["I1", SHARED_ITEM, SHARED_ITEM, SHARED_ITEM]  # the record without a value doesn't vote
    assert output_keys(df, None)[0].tolist() == df["supplier_key"].tolist()
    # a supplier seen with an ID in an earlier delivery keeps its own key
    from spendcube.dedupe import output_keys as keys
    assert keys(df, conv, frozenset({"TWO"}))[0].tolist() == ["S1", "S1", "S2", SHARED_SUPPLIER]
