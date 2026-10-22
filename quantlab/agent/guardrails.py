"""Checks that hold the language model to the research rules, in code.

The model writes the narrative, but it is not trusted to police itself. Before
a report is shown, these checks compare it with what the tools actually
returned:

* **Grounding** — every number in the report must appear in a tool result or
  in the user's own message (at the precision the report states it). A number
  the model computed in its head or invented is flagged.
* **Citations** — every URL must have come from a tool result.
* **Structure** — a research report must contain the ten required sections.
* **Language** — no guarantees ("risk-free", "guaranteed returns") and no
  claims that a real order was placed: this application cannot trade.
* **Untrusted input** — text from news, filings or uploads is wrapped so the
  model sees it as data, and it cannot close the wrapper from inside.
* **Secrets** — API keys and tokens are redacted from anything logged or shown.

The checks are deliberately conservative: a false alarm costs a reader a
second look; a fabricated number that slips through costs money.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Untrusted data and secrets
# --------------------------------------------------------------------------- #

_TAG = "untrusted_data"


def wrap_untrusted(text: str | None, source: str, max_chars: int = 2_000) -> str:
    """Fence external text so the model treats it as data, never as instructions.

    Any attempt inside the text to close (or re-open) the fence is neutralised,
    control characters are removed and the length is capped.
    """
    body = "" if text is None else str(text)
    body = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", body)
    body = re.sub(rf"</?\s*{_TAG}[^>]*>", "[removed tag]", body, flags=re.IGNORECASE)
    if len(body) > max_chars:
        body = body[:max_chars] + " …[truncated]"
    safe_source = re.sub(r"[^A-Za-z0-9 ._:/\-]", "", source)[:80]
    return f'<{_TAG} source="{safe_source}">{body}</{_TAG}>'


_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),                     # Anthropic keys
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                        # other provider-style keys
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b(\s*[:=]\s*)['\"]?[^\s'\"]{6,}"),
)


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a credential with ``[REDACTED]``."""
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
        elif pattern.groups == 1:
            out = pattern.sub(lambda m: f"{m.group(1)} [REDACTED]", out)
        else:
            out = pattern.sub("[REDACTED]", out)
    return out


# --------------------------------------------------------------------------- #
# Execution boundary
# --------------------------------------------------------------------------- #

# An allowlist, not a blocklist: a tool that is not a simulated paper_* tool must be named like a read or
# compute operation, and neither its name nor its description may describe acting on an account.
SAFE_TOOL_PREFIXES = ("get_", "list_", "run_", "calculate_", "evaluate_", "review_")
_DENY_NAME_TOKENS = frozenset({"order", "orders", "buy", "sell", "purchase", "liquidate", "rebalance", "broker",
                               "brokerage", "wire", "withdraw", "transfer", "deposit", "margin", "leverage", "execute",
                               "place", "submit", "route", "cancel", "fund", "funds", "close"})
_EXECUTION_WORDS = re.compile(
    r"\b(place|submit|send|route|execute|cancel|modify|create)s?\b[\w\s-]{0,30}?\b(order|trade)s?\b"
    r"|\b(buy|sell|purchase)s?\b[\w\s-]{0,20}?\b(shares?|stocks?|positions?|securities)\b"
    r"|\b(withdraw|transfer|deposit|wire|liquidate)\b|\bbroker(age)?\b|\bmargin\b|\bleverage\b"
    r"|\blive\s+(account|order|trading)\b|\breal\s+(account|money)\b",
    re.IGNORECASE,
)
_PAPER_DENY = re.compile(r"\bnot\s+simulated\b|\bbroker(age)?\b|\blive\s+(account|order|trading)\b"
                         r"|\breal\s+(account|money)\b|\b(withdraw|transfer|deposit|wire)\b", re.IGNORECASE)


def is_execution_tool(name: str, description: str, simulated: bool) -> bool:
    """True when a tool could act on a real account (and so may not be registered).

    A simulated tool must be named ``paper_*``, say "simulated" and never mention a broker, a live or
    real account, or moving money. Any other tool must have a read/compute prefix
    (:data:`SAFE_TOOL_PREFIXES`), no account-action word in its name, and no description of placing
    orders, trading shares or moving money.
    """
    if simulated:
        return not (name.startswith("paper_") and "simulated" in description.lower()
                    and not _PAPER_DENY.search(description))
    tokens = set(name.lower().split("_"))
    return (not name.startswith(SAFE_TOOL_PREFIXES) or bool(tokens & _DENY_NAME_TOKENS)
            or bool(_EXECUTION_WORDS.search(description)))


_LIVE_CLAIMS = re.compile(
    r"\b(?:I|we)(?:'ve|\s+have)?\s+(?:\w+\s+)?(?:placed|submitted|executed|sent|bought|sold)\b"
    r"|\b(?:orders?|trades?|purchases?)\s+(?:was|were|has\s+been|have\s+been|went|is|are)\s+"
    r"(?:placed|executed|filled|submitted|sent|through)\b"
    r"|(?<![\w-])(?:bought|sold)\s+[\d,.]+\s+(?:shares|units|contracts)\b",
    re.IGNORECASE,
)
_SIMULATED_NEARBY = re.compile(r"\b(paper|simulat\w*|backtest\w*|hypothetical)\b", re.IGNORECASE)
_REAL_ACCOUNT = re.compile(r"\b(broker(age)?|live\s+account|real\s+account|real\s+money|live\s+order)\b", re.IGNORECASE)
_GUARANTEES = re.compile(
    r"\bguarantee(d|s)?\b|\brisk[- ]free\b(?!\s+(rate|interest|asset|return))|\bcan(?:'|no)t\s+lose\b"
    r"|\bcannot\s+lose\b|\bsure\s+thing\b|\bcertain\s+(profit|gain|win)s?\b|\b100\s?%\s+(sure|certain)\b"
    r"|\bwill\s+definitely\b|\bno[- ]brainer\b",
    re.IGNORECASE,
)

# "no guarantee", "is not guaranteed", "without any guarantee", "cannot be guaranteed": the negation must govern
# the guarantee word directly; "no downside and guaranteed" is still a guarantee.
_NEGATION = re.compile(r"\b(?:no|not|never|without|nor|isn't|aren't|cannot|can't|won't|wouldn't|isnt|arent)"
                       r"(?:\s+(?:be|been|any|a|an|ever|always))*\s*$", re.IGNORECASE)

REPORT_SECTIONS = (
    "Executive summary",
    "Data freshness and source quality",
    "Market and sector context",
    "Key quantitative findings",
    "Fundamental and catalyst analysis",
    "Trading thesis and alternative scenarios",
    "Entry, exit and position-sizing framework",
    "Risk/reward and portfolio implications",
    "Evidence that would invalidate the thesis",
    "Sources, assumptions and outstanding uncertainties",
)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
          "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12}
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−+])?\s?(?P<cur>[$€£¥])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b|pp\b|bps\b)|\s?(?P<scale>k|mm|mn|m|bn|b|tn|t|thousand|million|billion|trillion)\b"
    r"|(?P<mult>x)\b)?",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b(?P<year>\d{4})-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b")
_HASH = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{8,}\b", re.IGNORECASE)   # fingerprints, ids
_URL = re.compile(r"https?://[^\s)\]>\"'`]+")
_BARE_LINK = re.compile(r"(?<![\w/@.:-])(?:www\.)?(?:[a-z0-9-]+\.)+(?:com|org|net|gov|edu|io|co|uk|de|fr|jp|info|biz|us)"
                        r"(?:/[^\s)\]>\"'`]*)?(?![\w.-]*\w)", re.IGNORECASE)
_YEAR_CONTEXT = re.compile(r"\b(?:in|since|by|from|until|through|during|fiscal|year|FY|of|before|after|early|late|mid|"
                           r"between|and|to)\s*$", re.IGNORECASE)
_LEVEL_CONTEXT = re.compile(r"\b(?:at|to|target|stop|price|level|entry|exit|of|from|near|above|below)\s*$",
                            re.IGNORECASE)
_IDENTIFIER = re.compile(r"\b[A-Za-z]+[-_/]?\d+[A-Za-z]*\b")      # 10-K, Q3, SMA50, RSI14, 8-K, S&P500 …
_HEADING_NUMBER = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s*)?\d{1,2}[.)]\s")


@dataclass(frozen=True)
class ReportedNumber:
    text: str
    value: float
    decimals: int
    percent: bool
    context: str
    scale: float = 1.0          # 1e9 for "1.2bn": precision is half a unit of the stated scale
    signed: bool = False        # an explicit + or - was written: the direction is part of the claim
    bps: bool = False           # written in basis points (``value`` is in percent): may match a raw bps figure

    @property
    def tolerance(self) -> float:
        return 0.5 * 10 ** (-self.decimals) * self.scale


def _decimals(num: str) -> int:
    return len(num.split(".")[1]) if "." in num else 0


def _mask(text: str) -> str:
    """Remove what is not a stated quantity: links, dates and times, hashes, identifiers, list numbering."""
    masked = _URL.sub(" ", text)
    masked = _DATE.sub(" ", masked)
    masked = _TIME.sub(" ", masked)
    masked = _HASH.sub(" ", masked)
    masked = _IDENTIFIER.sub(" ", masked)
    return _HEADING_NUMBER.sub(" ", masked)


def extract_numbers(text: str, exempt: bool = True) -> list[ReportedNumber]:
    """Numbers stated in prose, excluding dates, times, links, hashes, identifiers such as 10-K or
    SMA50 and list numbering. With ``exempt`` (for reports), small bare counts (0-10 without a unit)
    and years are not checked — unless they read as a price level ("stop at 9", "target 2050")."""
    masked = _mask(text)
    found: list[ReportedNumber] = []
    for m in _NUMBER.finditer(masked):
        num = m.group("num")
        value = float(num.replace(",", ""))
        if m.group("sign") in ("-", "−"):
            value = -value
        pct = bool(m.group("pct"))
        unit = m.group("cur") or m.group("pct") or m.group("scale") or m.group("mult")
        scale = _SCALE[m.group("scale").lower()] if m.group("scale") else 1.0
        value *= scale
        if pct and m.group("pct").lower() == "bps":
            value /= 100.0  # basis points -> percent
            scale = 0.01    # and the precision with it: "12 bps" is 0.12% +/- 0.005%
        before = masked[max(0, m.start() - 15):m.start()]
        plain = not unit and "." not in num and "," not in num and not m.group("sign")
        bare_count = plain and 0 <= abs(value) <= 10 and not _LEVEL_CONTEXT.search(before)
        bare_year = plain and 1900 <= value <= 2100 and bool(_YEAR_CONTEXT.search(before))
        if exempt and (bare_count or bare_year):
            continue
        start = max(0, m.start() - 40)
        found.append(ReportedNumber(m.group(0).strip(), value, _decimals(num), pct,
                                    masked[start:m.end() + 40].replace("\n", " ").strip(), scale,
                                    m.group("sign") is not None, bool(pct and m.group("pct").lower() == "bps")))
    return found


@dataclass
class Grounding:
    """The numbers a report may state: ``values`` (numbers and fractions, with their sign) and ``percents``
    (the x of an "x%" written in a tool result or the user's input)."""

    values: set[float] = field(default_factory=set)
    percents: set[float] = field(default_factory=set)


def collect_numbers(obj: Any, out: Grounding | None = None) -> Grounding:
    """Every number in a tool result: values, quantities written inside strings (not the digits of dates,
    times, links or hashes), the year of each date, and digits in identifiers and keys ("sma_200")."""
    out = Grounding() if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, int | float):
        if math.isfinite(float(obj)):
            out.values.add(float(obj))
        return out
    if isinstance(obj, str):
        for n in extract_numbers(obj, exempt=False):
            if n.percent:
                out.percents.add(n.value)
                out.values.add(n.value / 100.0)
            else:
                out.values.add(n.value)
        out.values.update(float(m.group("year")) for m in _DATE.finditer(_URL.sub(" ", obj)))
        for ident in _IDENTIFIER.findall(_URL.sub(" ", obj)):
            out.values.update(float(d) for d in re.findall(r"\d+", ident))
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            collect_numbers(str(k), out)
            collect_numbers(v, out)
        return out
    if isinstance(obj, list | tuple | set):
        for v in obj:
            collect_numbers(v, out)
    return out


def _matches(reported: ReportedNumber, grounded: Grounding) -> bool:
    """Equal to a grounded value at the precision the report states (half a unit in the last place shown,
    of the stated scale). A percentage matches a fraction (12.3% vs 0.123) or a percentage written as such,
    never a raw number of the same digits (0.12% is not 0.12). The sign is compared only when the report
    writes one ("+35%" is not -0.35; "a 12% drawdown" may be -0.12)."""
    tol = reported.tolerance + 1e-12

    def same(g: float, target: float, tolerance: float) -> bool:
        if reported.signed:
            return abs(g - target) <= tolerance
        return abs(abs(g) - abs(target)) <= tolerance

    if reported.percent:
        return (any(same(g, reported.value / 100.0, tol / 100.0) for g in grounded.values)
                or any(same(g, reported.value, tol) for g in grounded.percents)
                or (reported.bps and any(same(g, reported.value * 100.0, tol * 100.0) for g in grounded.values)))
    return any(same(g, reported.value, tol) for g in grounded.values)


def _link_key(link: str) -> tuple[str, str]:
    bare = re.sub(r"^https?://", "", link.strip().rstrip(".,;)"), flags=re.IGNORECASE)
    host, _, path = bare.partition("/")
    return host.lower().removeprefix("www."), path.rstrip("/")


# --------------------------------------------------------------------------- #
# The report check
# --------------------------------------------------------------------------- #


@dataclass
class GuardrailReport:
    numbers_checked: int = 0
    ungrounded_numbers: list[ReportedNumber] = field(default_factory=list)
    unsourced_urls: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    guarantee_language: list[str] = field(default_factory=list)
    execution_claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.ungrounded_numbers or self.unsourced_urls or self.missing_sections
                    or self.guarantee_language or self.execution_claims)

    def issues(self) -> list[str]:
        out = []
        if self.ungrounded_numbers:
            shown = ", ".join(f"'{n.text}'" for n in self.ungrounded_numbers[:8])
            more = len(self.ungrounded_numbers) - 8
            out.append(f"{len(self.ungrounded_numbers)} number(s) not found in any tool result or your input: {shown}"
                       + (f" and {more} more" if more > 0 else ""))
        if self.unsourced_urls:
            out.append("Link(s) not returned by any tool: " + ", ".join(self.unsourced_urls[:5]))
        if self.missing_sections:
            out.append("Missing report section(s): " + "; ".join(self.missing_sections))
        if self.guarantee_language:
            out.append("Prohibited certainty language: " + "; ".join(self.guarantee_language))
        if self.execution_claims:
            out.append("Claims of live execution (this application cannot trade): " + "; ".join(self.execution_claims))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "numbers_checked": self.numbers_checked, "issues": self.issues(),
                "ungrounded_numbers": [{"text": n.text, "context": n.context} for n in self.ungrounded_numbers],
                "unsourced_urls": self.unsourced_urls, "missing_sections": self.missing_sections,
                "guarantee_language": self.guarantee_language, "execution_claims": self.execution_claims,
                "notes": self.notes}


def _normalise_heading(text: str) -> str:
    return re.sub(r"[^a-z/]+", " ", text.lower()).strip()


def check_report(text: str, tool_results: Iterable[Any], user_inputs: Iterable[str] = (),
                 require_sections: bool = True, simulated_context: bool = False) -> GuardrailReport:
    """Check a model-written report against the tool results it was given.

    ``simulated_context`` (PAPER TRADE, POST-TRADE REVIEW, BACKTEST) allows narrating simulated trades
    ("we sold too early"); claims that involve a broker, a live or real account or real money are
    still flagged.
    """
    report = GuardrailReport()
    results = list(tool_results)
    grounded = collect_numbers(results)
    for user_text in user_inputs:
        collect_numbers(user_text, grounded)

    numbers = extract_numbers(text)
    report.numbers_checked = len(numbers)
    report.ungrounded_numbers = [n for n in numbers if not _matches(n, grounded)]

    blob = json.dumps(results, default=str)
    known = {_link_key(u) for u in _URL.findall(blob)}
    known_hosts = {host for host, _ in known}
    cited = [u.rstrip(".,;") for u in _URL.findall(text)]
    cited += [u.rstrip(".,;") for u in _BARE_LINK.findall(_URL.sub(" ", text))]
    report.unsourced_urls = sorted({u for u in cited if _link_key(u) not in known
                                    and not (not _link_key(u)[1] and _link_key(u)[0] in known_hosts)})

    if require_sections:
        headings = {_normalise_heading(h) for h in re.findall(r"(?m)^\s{0,3}#{1,6}\s*(?:\d{1,2}[.)]\s*)?(.+?)\s*$", text)}
        headings |= {_normalise_heading(h) for h in re.findall(r"(?m)^\s{0,3}\d{1,2}[.)]\s*\**(.+?)\**\s*$", text)}
        headings |= {_normalise_heading(h) for h in
                     re.findall(r"(?m)^\s{0,3}\*\*\s*(?:\d{1,2}[.)]\s*)?(.+?)\s*\*\*:?\s*$", text)}
        report.missing_sections = [s for s in REPORT_SECTIONS
                                   if not any(_normalise_heading(s) in h for h in headings)]

    report.guarantee_language = sorted({m.group(0) for m in _GUARANTEES.finditer(text)
                                        if not _NEGATION.search(text[max(0, m.start() - 30):m.start()])})
    claims = set()
    for m in _LIVE_CLAIMS.finditer(text):
        sentence_start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        sentence_end = min((i for i in (text.find(".", m.end()), text.find("\n", m.end())) if i >= 0), default=len(text))
        sentence = text[sentence_start:sentence_end]
        if _REAL_ACCOUNT.search(sentence):
            claims.add(m.group(0))
        elif not simulated_context and not _SIMULATED_NEARBY.search(sentence):
            claims.add(m.group(0))
    report.execution_claims = sorted(claims)
    return report
