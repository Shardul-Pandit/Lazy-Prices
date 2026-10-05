"""Turn a raw 10-K into clean text and pull out Item 1A (Risk Factors) and Item 7 (MD&A).

10-K HTML is messy and changes across years and filers, so this module is defensive:
- Hidden inline-XBRL metadata is dropped (it is machine data, not prose).
- Tables that are mostly numbers are dropped, following the Loughran-McDonald
  convention, so the signal measures language rather than financial statements.
- Section boundaries come from "Item N" heading lines. A heading usually appears at
  least twice (table of contents, then the real section), so the extractor takes the
  occurrence that yields the longest section instead of the first one.
- Nothing is trusted silently: every filing gets QA numbers, and sections that look
  wrong (too short, or swallowing most of the document) are flagged and excluded.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import lxml.html
import pandas as pd
from tqdm import tqdm

from . import config
from .edgar import raw_path

PARSER_VERSION = 1

BLOCK_TAGS = frozenset(
    "p div br tr li h1 h2 h3 h4 h5 h6 table ul ol blockquote pre hr section article "
    "title center dd dt caption".split()
)
CELL_TAGS = frozenset({"td", "th"})

# A table is treated as numeric (and removed) when digits make up more than this share
# of its alphanumeric characters. Layout tables that only hold prose stay.
NUMERIC_TABLE_THRESHOLD = 0.15

# QA thresholds
MIN_FULL_WORDS = 5_000  # shorter: the 10-K likely incorporates its content by reference
MIN_SECTION_WORDS = 500
MAX_SECTION_SHARE = 0.75  # a "section" this large is a boundary error, not a section

_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
_SGML_DOC = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_SGML_TYPE = re.compile(r"<TYPE>\s*([^\s<]+)", re.I)
_SGML_TEXT = re.compile(r"<TEXT>(.*)", re.S | re.I)
_HTML_HINT = re.compile(r"<(html|body|div|p|table|font|span)\b", re.I)
_WS = re.compile(r"[ \t\f\v]+")
_NOISE_LINE = re.compile(r"^(?:-?\s*\d{1,3}\s*-?|page\s+\d{1,3}(?:\s+of\s+\d+)?|table\s+of\s+contents)$", re.I)
_WORD = re.compile(r"[A-Za-z]{2,}")
_LEADING_PUNCT = re.compile(r"^[^a-z0-9]+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACE_CHARS = dict.fromkeys(map(ord, "\xa0          　"), " ")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)

# "Item 1A. Risk Factors", "ITEM 7 - MANAGEMENT'S...", "Part II, Item 7.", "Items 7 and 7A."
_ITEM_LINE = re.compile(
    r"^(?:part\s+[iv]{1,3}\s*[\.,:;\-–—]*\s*)?"
    r"items?\s*(\d{1,2})\s*\(?([abc])?\)?(?![a-z0-9])"
    r"(?:\s*(?:and|&|,)\s*\d{1,2}[abc]?(?![a-z0-9]))*"
    r"\s*[\.\:\-–—\)]*\s*(.*)$",
    re.I,
)
MAX_HEADING_CHARS = 200

SECTIONS = {
    # name: (start code, end codes, words the heading title may begin with)
    "item1a": ("1a", ("1b", "1c", "2", "3"), ("risk",)),
    "item7": ("7", ("7a", "8"), ("management", "md&a")),
}


@dataclass
class ParsedFiling:
    full: str
    item1a: str
    item7: str
    qa: dict


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _main_document(text: str) -> str:
    """A full-submission .txt file wraps several documents; keep the 10-K itself."""
    if "<DOCUMENT>" not in text[:20_000].upper():
        return text
    for match in _SGML_DOC.finditer(text):
        body = match.group(1)
        doc_type = _SGML_TYPE.search(body)
        if doc_type and doc_type.group(1).upper().startswith("10-K"):
            inner = _SGML_TEXT.search(body)
            return re.sub(r"</TEXT>\s*$", "", inner.group(1) if inner else body, flags=re.I)
    return text


def normalize_text(text: str) -> str:
    """One paragraph per line, single spaces, page furniture removed."""
    text = text.translate(_ZERO_WIDTH).translate(_SPACE_CHARS).replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line).strip()
        if line and not _NOISE_LINE.match(line):
            lines.append(line)
    return "\n".join(lines)


def _is_numeric_table(table) -> bool:
    text = table.text_content()
    digits = sum(c.isdigit() for c in text)
    alpha = sum(c.isalpha() for c in text)
    return digits + alpha > 0 and digits / (digits + alpha) > NUMERIC_TABLE_THRESHOLD


def html_to_text(html: str) -> str:
    """Convert filing HTML to text with one line per block-level element."""
    html = _XML_DECL.sub("", html, count=1)  # lxml rejects str input that declares an encoding
    html = _CONTROL_CHARS.sub("", html)  # a few filings contain stray NULL bytes
    try:
        root = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return ""

    doomed = []
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue  # comments and processing instructions
        tag = tag.lower()
        if tag in ("script", "style", "head", "noscript") or tag.endswith(":header"):
            doomed.append(el)
            continue
        style = el.get("style")
        if style and "display:none" in style.replace(" ", "").lower():
            doomed.append(el)  # inline-XBRL hidden facts live here
    for el in doomed:
        if el.getparent() is not None:
            el.drop_tree()

    # Innermost tables first: an outer layout table is judged after its numeric
    # children are gone, so a page-wide wrapper is not thrown away with them.
    for table in reversed(root.xpath("//table")):
        if table.getparent() is not None and _is_numeric_table(table):
            table.drop_tree()

    # Line breaks in HTML source are just whitespace, except inside <pre>.
    preformatted = {id(d) for pre in root.iter("pre") for d in pre.iter()}
    pre_roots = {id(pre) for pre in root.iter("pre")}
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        tag = tag.lower()
        inside_pre = id(el) in preformatted
        if el.text and not inside_pre:
            el.text = el.text.replace("\n", " ").replace("\r", " ")
        if el.tail and (not inside_pre or id(el) in pre_roots):
            el.tail = el.tail.replace("\n", " ").replace("\r", " ")
        if tag in BLOCK_TAGS:
            el.text = "\n" + (el.text or "")
            el.tail = "\n" + (el.tail or "")
        elif tag in CELL_TAGS:
            el.tail = " " + (el.tail or "")
    return normalize_text(root.text_content())


def to_text(raw: bytes) -> str:
    text = _main_document(decode(raw))
    if _HTML_HINT.search(text[:200_000]):
        return html_to_text(text)
    return normalize_text(text)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def find_item_headings(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (line number, item code, rest of line) for every line that opens with 'Item N'."""
    found = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS:
            continue
        m = _ITEM_LINE.match(line)
        if m:
            found.append((i, (m.group(1) + (m.group(2) or "")).lower(), m.group(3).strip()))
    return found


def extract_section(lines: list[str], headings: list[tuple[int, str, str]], name: str) -> str:
    """Longest span from a start heading to the next end heading; '' if none is found."""
    start_code, end_codes, title_words = SECTIONS[name]
    word_counts = [count_words(line) for line in lines]
    cumulative = [0]
    for n in word_counts:
        cumulative.append(cumulative[-1] + n)

    # Rank candidates by (clean, words). A span is "clean" when no other item heading
    # sits inside it. A span that starts in the table of contents and runs into the
    # body contains other items' headings, so a clean span always wins over it.
    best: tuple[bool, int, int, int] | None = None
    for line_no, code, rest in headings:
        if code != start_code:
            continue
        # If the title is on the heading line it must be the right title. This rejects
        # lines such as "Item 7 of this report describes..." that only mention the item.
        title = _LEADING_PUNCT.sub("", rest.lower())
        if title and not title.startswith(title_words):
            continue
        end = next((ln for ln, c, _ in headings if ln > line_no and c in end_codes), None)
        if end is None:
            continue
        clean = not any(line_no < ln < end and c != start_code for ln, c, _ in headings)
        words = cumulative[end] - cumulative[line_no + 1]
        if best is None or (clean, words) > best[:2]:
            best = (clean, words, line_no, end)
    if best is None:
        return ""
    _, _, start, end = best
    # Some filers repeat "ITEM 1A | Risk Factors" as a running header on every page.
    # Those lines are page furniture, not prose, so they are left out of the section.
    heading_lines = {ln for ln, _, _ in headings}
    return "\n".join(lines[i] for i in range(start + 1, end) if i not in heading_lines)


def parse_filing(raw: bytes) -> ParsedFiling:
    full = to_text(raw)
    lines = full.split("\n") if full else []
    headings = find_item_headings(lines)
    item1a = extract_section(lines, headings, "item1a")
    item7 = extract_section(lines, headings, "item7")

    n_full, n_1a, n_7 = count_words(full), count_words(item1a), count_words(item7)

    def section_ok(n: int) -> bool:
        return n >= MIN_SECTION_WORDS and n_full > 0 and n / n_full <= MAX_SECTION_SHARE

    qa = {
        "n_words_full": n_full,
        "n_words_item1a": n_1a,
        "n_words_item7": n_7,
        "full_ok": n_full >= MIN_FULL_WORDS,
        "item1a_ok": section_ok(n_1a),
        "item7_ok": section_ok(n_7),
        "parser_version": PARSER_VERSION,
    }
    return ParsedFiling(full=full, item1a=item1a, item7=item7, qa=qa)


def text_path(cik: int, accession: str):
    return config.TEXT_DIR / str(int(cik)) / f"{accession}.json.gz"


def load_text(cik: int, accession: str) -> dict:
    with gzip.open(text_path(cik, accession), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_one(job: tuple[int, str]) -> dict:
    cik, accession = job
    row = {"cik": cik, "accession": accession, "error": ""}
    try:
        with gzip.open(raw_path(cik, accession), "rb") as fh:
            parsed = parse_filing(fh.read())
        out = text_path(cik, accession)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as fh:
            json.dump({"full": parsed.full, "item1a": parsed.item1a, "item7": parsed.item7}, fh)
        os.replace(tmp, out)
        row.update(parsed.qa)
    except Exception as exc:  # one bad filing must not stop the batch; it is recorded
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["parser_version"] = PARSER_VERSION
    return row


def parse_all(index: pd.DataFrame, workers: int | None = None, force: bool = False) -> pd.DataFrame:
    """Parse every downloaded filing that has not been parsed by this parser version."""
    have_raw = [raw_path(c, a).exists() for c, a in zip(index["cik"], index["accession"], strict=True)]
    index = index[have_raw]
    done = pd.DataFrame()
    if config.PARSE_QA.exists() and not force:
        done = pd.read_parquet(config.PARSE_QA)
        done = done[(done["parser_version"] == PARSER_VERSION) & done["accession"].isin(index["accession"])]
        done = done[[text_path(c, a).exists() for c, a in zip(done["cik"], done["accession"], strict=True)]]
    todo = index[~index["accession"].isin(done["accession"] if len(done) else [])]
    print(f"{len(done)} filings already parsed, {len(todo)} to parse")
    rows = []
    if len(todo):
        jobs = list(zip(todo["cik"].astype(int), todo["accession"], strict=True))
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(tqdm(pool.map(_parse_one, jobs, chunksize=8), total=len(jobs), desc="Parsing 10-Ks"))
    qa = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    for col in ("full_ok", "item1a_ok", "item7_ok"):
        qa[col] = qa[col].fillna(False).astype(bool)
    config.PARSE_QA.parent.mkdir(parents=True, exist_ok=True)
    qa.to_parquet(config.PARSE_QA, index=False)
    return qa


def qa_summary(qa: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Extraction success rates by filing year: the honest scorecard for the parser."""
    df = qa.merge(index[["accession", "filing_date"]], on="accession", how="left")
    df["year"] = pd.to_datetime(df["filing_date"]).dt.year
    out = df.groupby("year").agg(
        filings=("accession", "size"),
        parse_errors=("error", lambda s: int((s != "").sum())),
        full_ok=("full_ok", "mean"),
        item1a_ok=("item1a_ok", "mean"),
        item7_ok=("item7_ok", "mean"),
        median_words=("n_words_full", "median"),
    )
    return out
