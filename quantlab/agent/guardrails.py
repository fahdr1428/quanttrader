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

# Words that describe acting on a real account. A tool whose name or description
# uses them is an execution tool and may not be registered (paper tools are
# allowed, and must say they are simulated).
_EXECUTION_WORDS = re.compile(
    r"\b(place|submit|send|route|execute|cancel|modify)[_ ]?(live|real)?[_ ]?(order|trade)s?\b"
    r"|\b(withdraw|transfer|deposit)\b|\bbroker\b|\bmargin\b|\bleverage\b",
    re.IGNORECASE,
)


def is_execution_tool(name: str, description: str, simulated: bool) -> bool:
    """True when a tool could act on a real account. Paper tools must be marked simulated,
    be named ``paper_*`` and say "simulated" in their description."""
    if simulated:
        return not (name.startswith("paper_") and "simulated" in description.lower())
    return bool(_EXECUTION_WORDS.search(name.replace("_", " ")) or _EXECUTION_WORDS.search(description))


_LIVE_CLAIMS = re.compile(
    r"\b(I|we)\s+(have\s+)?(placed|submitted|executed|sent|bought|sold)\b(?![^.]{0,80}\b(paper|simulat))"
    r"|\border\s+(was|has been)\s+(placed|executed|filled)\b(?![^.]{0,80}\b(paper|simulat))",
    re.IGNORECASE,
)
_GUARANTEES = re.compile(
    r"\bguarantee(d|s)?\b|\brisk[- ]free\b(?!\s+(rate|interest|asset|return))|\bcan(?:'|no)t\s+lose\b"
    r"|\bcannot\s+lose\b|\bsure\s+thing\b|\bcertain\s+(profit|gain|win)s?\b|\b100\s?%\s+(sure|certain)\b"
    r"|\bwill\s+definitely\b|\bno[- ]brainer\b",
    re.IGNORECASE,
)

# "no guarantee", "is not guaranteed", "without any guarantee": honest disclaimers, not claims.
_NEGATION = re.compile(r"\b(no|not|never|without|nor|cannot|can't|isn't|aren't|nothing)\b[^.!?]{0,20}$", re.IGNORECASE)

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
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_URL = re.compile(r"https?://[^\s)\]>\"'`]+")
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

    @property
    def tolerance(self) -> float:
        return 0.5 * 10 ** (-self.decimals) * self.scale


def _decimals(num: str) -> int:
    return len(num.split(".")[1]) if "." in num else 0


def extract_numbers(text: str) -> list[ReportedNumber]:
    """Numbers stated in prose, excluding dates, URLs, identifiers such as 10-K or SMA50,
    list numbering and small bare counts (0-10 without a unit)."""
    masked = _URL.sub(" ", text)
    masked = _DATE.sub(" ", masked)
    masked = _IDENTIFIER.sub(" ", masked)
    masked = _HEADING_NUMBER.sub(" ", masked)
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
        bare_count = not unit and "." not in num and "," not in num and 0 <= abs(value) <= 10
        bare_year = not unit and "." not in num and "," not in num and 1900 <= value <= 2100
        if bare_count or bare_year:
            continue
        start = max(0, m.start() - 40)
        found.append(ReportedNumber(m.group(0).strip(), value, _decimals(num), pct,
                                    masked[start:m.end() + 40].replace("\n", " ").strip(), scale))
    return found


def collect_numbers(obj: Any, out: set[float] | None = None) -> set[float]:
    """Every number in a tool result (values, numbers inside strings, digits in keys)."""
    out = set() if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, int | float):
        if math.isfinite(float(obj)):
            out.add(float(obj))
        return out
    if isinstance(obj, str):
        for n in extract_numbers(obj):
            out.add(n.value)
            if n.percent:
                out.add(n.value / 100.0)
        # Digits inside identifiers ("sma_200", "RSI(14)") count; digits inside URLs do not.
        for raw in re.findall(r"-?\d+(?:\.\d+)?", _URL.sub(" ", obj).replace(",", "")):
            out.add(float(raw))
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


def _matches(reported: ReportedNumber, grounded: Iterable[float]) -> bool:
    """Equal to a grounded value at the precision the report states (half a unit in the last
    place shown, of the stated scale). A percentage also matches its fraction (12.3% vs 0.123),
    and the sign is not compared ("a 12% drawdown" for -0.12)."""
    candidates = [(abs(reported.value), reported.tolerance)]
    if reported.percent:
        candidates.append((abs(reported.value) / 100.0, reported.tolerance / 100.0))
    return any(abs(abs(g) - c) <= tol + 1e-12 for g in grounded for c, tol in candidates)


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
                 require_sections: bool = True) -> GuardrailReport:
    """Check a model-written report against the tool results it was given."""
    report = GuardrailReport()
    results = list(tool_results)
    grounded = collect_numbers(results)
    for user_text in user_inputs:
        collect_numbers(user_text, grounded)

    numbers = extract_numbers(text)
    report.numbers_checked = len(numbers)
    report.ungrounded_numbers = [n for n in numbers if not _matches(n, grounded)]

    known_urls = set()
    for blob in results:
        known_urls.update(u.rstrip(".,;") for u in _URL.findall(json.dumps(blob, default=str)))
    report.unsourced_urls = sorted({u.rstrip(".,;") for u in _URL.findall(text)} - known_urls)

    if require_sections:
        headings = {_normalise_heading(h) for h in re.findall(r"(?m)^\s{0,3}#{1,6}\s*(?:\d{1,2}[.)]\s*)?(.+?)\s*$", text)}
        headings |= {_normalise_heading(h) for h in re.findall(r"(?m)^\s{0,3}\d{1,2}[.)]\s*\**(.+?)\**\s*$", text)}
        report.missing_sections = [s for s in REPORT_SECTIONS
                                   if not any(_normalise_heading(s) in h for h in headings)]

    report.guarantee_language = sorted({m.group(0) for m in _GUARANTEES.finditer(text)
                                        if not _NEGATION.search(text[max(0, m.start() - 25):m.start()])})
    report.execution_claims = sorted({m.group(0) for m in _LIVE_CLAIMS.finditer(text)})
    return report
