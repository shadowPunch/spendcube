import pandas as pd
import pytest

from spendcube.normalise import parse_amount, parse_dates, parse_rate
from spendcube.text import desc_key, invoice_core, numbers, supplier_norm


@pytest.mark.parametrize("variant", [
    "PRAHAVEN (PTY) LTD", "PRAH AVEN (PTY) LTD", "Prahaven (Pty) Ltd", "PRAHAVEN  PTY LTD", "PRAHAVEN",
])
def test_supplier_variants_share_a_key(variant):
    assert supplier_norm(variant) == "PRAHAVEN"


def test_supplier_keeps_distinct_names_apart():
    assert supplier_norm("KELMONT (PTY) LTD") != supplier_norm("KINMONT (PTY) LTD")


def test_desc_key_folds_ocr_and_decimal_comma():
    assert desc_key("RE5IN, ROOF BOLT;1,5-1,8M") == desc_key("RESIN, ROOF BOLT;1.5-1.8M")
    assert desc_key("D0OR;VNT 2") == desc_key("DOOR;VNT 2")
    # part numbers and measurements are not folded
    assert desc_key("ROD ANCHR;1500MM") != desc_key("ROD ANCHR;1800MM")


def test_numbers_guard():
    assert numbers("ROD;DIA:20MM,1800MM") == ("20", "1800")
    assert numbers("MESH;3M,WD:2,2M") == numbers("MESH;3M,WD:2.2M")


@pytest.mark.parametrize("ref,core", [
    ("INV/2023/759857", "759857"), ("TAX INV 759857", "759857"), ("IN0759857", "759857"),
    ("SI-2023-759857", "759857"), ("759857", "759857"), (None, ""),
])
def test_invoice_core(ref, core):
    assert invoice_core(ref) == core


def test_parse_amount_formats():
    values = pd.Series(["5,266.63", "-1,234.5", "(12.00)", "7.5-", "abc", None])
    out = parse_amount(values)
    assert out.iloc[:4].tolist() == [5266.63, -1234.5, -12.0, -7.5]
    assert out.iloc[4:].isna().all()
    assert parse_amount(pd.Series(["1.234,56"]), decimal=",").iloc[0] == pytest.approx(1234.56)


def test_parse_rate_notations():
    out = parse_rate(pd.Series(["15", "0.15", "15%", "0", None]))
    assert out.iloc[:4].tolist() == [0.15, 0.15, 0.15, 0.0]
    assert pd.isna(out.iloc[4])


def test_parse_dates_tries_formats_in_order():
    out = parse_dates(pd.Series(["2024-03-27", "27/03/2024", "27-Mar-24", "31/02/2024"]),
                      ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y"])
    assert out.iloc[:3].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-27"] * 3
    assert pd.isna(out.iloc[3])
