from lazyprices import parse

RISK = " ".join(f"Risk sentence number {i} about competition and regulation." for i in range(120))
MDA = " ".join(f"Revenue discussion sentence {i} about results of operations." for i in range(150))

HTML = f"""<?xml version='1.0' encoding='ASCII'?>
<html><head><title>10-K</title><style>p {{ color: red }}</style></head><body>
<div style="display: none"><ix:header><ix:hidden>HIDDENFACT 0001234567 2024 FY</ix:hidden></ix:header></div>
<p>ANNUAL REPORT</p>
<table>
 <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
 <tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
 <tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>25</td></tr>
 <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>30</td></tr>
 <tr><td>Item 7A.</td><td>Quantitative and Qualitative Disclosures</td><td>45</td></tr>
 <tr><td>Item 8.</td><td>Financial Statements</td><td>47</td></tr>
</table>
<p><b>Item 1.</b> <b>Business</b></p>
<p>We make widgets. This paragraph is wrapped
in the HTML source but is one paragraph. See Part I, Item 1A of this report.</p>
<table><tr><td><b>Item 1A.</b></td><td><b>Risk Factors</b></td></tr></table>
<p>{RISK}</p>
<table><tr><td>2024</td><td>1,234</td><td>5,678</td></tr><tr><td>2023</td><td>1,111</td><td>4,321</td></tr></table>
<p>12</p>
<p>Table of Contents</p>
<p>FINALRISKMARKER closing risk paragraph.</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
<p>Item 2. Properties</p>
<p>We own a factory.</p>
<div><span>ITEM</span><span>&nbsp;7.</span> <span>MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</span></div>
<p>{MDA}</p>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>Interest rate risk is modest.</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<table><tr><td>Revenue</td><td>10,000</td><td>9,000</td></tr><tr><td>Net income</td><td>1,000</td><td>900</td></tr></table>
</body></html>"""


def test_html_to_text_cleans_hidden_data_tables_and_page_furniture():
    text = parse.to_text(HTML.encode())
    assert "HIDDENFACT" not in text  # inline-XBRL hidden block removed
    assert "color: red" not in text  # style removed
    assert "5,678" not in text and "10,000" not in text  # numeric tables removed
    lines = text.split("\n")
    assert "12" not in lines and "Table of Contents" not in lines  # page number / header lines
    # A paragraph wrapped in the HTML source stays on one line.
    assert any("wrapped in the HTML source but is one paragraph" in line for line in lines)


def test_sections_skip_the_table_of_contents_and_stop_at_the_next_item():
    parsed = parse.parse_filing(HTML.encode())
    assert parsed.item1a.startswith("Risk sentence number 0")
    assert "FINALRISKMARKER" in parsed.item1a
    assert "Unresolved Staff Comments" not in parsed.item1a and "factory" not in parsed.item1a
    assert parsed.item7.startswith("Revenue discussion sentence 0")
    assert "Interest rate risk" not in parsed.item7
    assert parsed.qa["item1a_ok"] and parsed.qa["item7_ok"]


def test_cross_reference_lines_are_not_headings():
    lines = [
        "Item 7 of this report describes our liquidity.",  # mentions the item, wrong title
        "Item 7. Management's Discussion and Analysis",
        "real mda text " * 50,
        "Item 8. Financial Statements",
    ]
    section = parse.extract_section(lines, parse.find_item_headings(lines), "item7")
    assert section.startswith("real mda text") and "liquidity" not in section


def test_item_code_is_not_confused_by_the_title():
    # "Item 1 Business" must be item 1, not item "1b".
    assert parse.find_item_headings(["Item 1 Business", "Items 7 and 7A. Management's Discussion"]) == [
        (0, "1", "Business"),
        (1, "7", "Management's Discussion"),
    ]


def test_missing_section_is_reported_not_guessed():
    parsed = parse.parse_filing(b"<html><body><p>Short wrapper. See Exhibit 13.</p></body></html>")
    assert parsed.item1a == "" and parsed.item7 == ""
    assert not parsed.qa["full_ok"] and not parsed.qa["item1a_ok"]


def test_sgml_wrapper_selects_the_10k_document():
    raw = b"<SEC-DOCUMENT>\n<DOCUMENT>\n<TYPE>10-K\n<TEXT>\nPlain text annual report body.\n</TEXT>\n</DOCUMENT>\n<DOCUMENT>\n<TYPE>EX-21\n<TEXT>\nSubsidiaries list.\n</TEXT>\n</DOCUMENT>"
    text = parse.to_text(raw)
    assert "annual report body" in text and "Subsidiaries" not in text


def test_incomplete_table_of_contents_does_not_swallow_the_document():
    lines = [
        "Item 7. Management's Discussion 30",  # table of contents entry, no 7A/8 entries after it
        "Item 1. Business",
        "business text " * 100,
        "Item 7. Management's Discussion and Analysis",
        "real mda text " * 50,
        "Item 8. Financial Statements",
    ]
    section = parse.extract_section(lines, parse.find_item_headings(lines), "item7")
    assert section.startswith("real mda text") and "business text" not in section


def test_running_page_headers_do_not_break_or_pollute_a_section():
    lines = [
        "ITEM 1A",  # table of contents
        "ITEM 1B",
        "ITEM 1A | Risk Factors",
        "first page of risks " * 40,
        "ITEM 1A | Risk Factors",  # repeated at the top of the next page
        "second page of risks " * 40,
        "ITEM 1B | Unresolved Staff Comments",
    ]
    section = parse.extract_section(lines, parse.find_item_headings(lines), "item1a")
    assert "first page" in section and "second page" in section and "ITEM 1A" not in section
