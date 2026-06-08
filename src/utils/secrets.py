"""
Secret detection and redaction for message ingestion.

Two responsibilities:
  1. Detect API keys / credentials in free text (provider-specific patterns
     plus a high-entropy fallback for unknown token shapes).
  2. Redact them in place, preserving the surrounding text so an observation
     like "my key is sk-... use it" still derives "user shared an API key"
     without storing the live secret.

Used on the ingest path (crud.create_messages) so secrets never reach the
messages table, the embeddings, or the deriver. The deriver also calls
scrub_text as defence in depth.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Per-provider patterns. Order matters only for nicer hit-type labels; the
# scrub is a single pass over a combined alternation.
#
# Each entry: (label, compiled regex). Patterns are deliberately anchored on
# distinctive prefixes to keep false positives low.
_PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic: sk-ant-api..., sk-ant-oat... (OAuth tokens). MUST precede the
    # OpenAI pattern below — otherwise the broad sk- prefix mislabels these.
    ("anthropic", re.compile(r"\bsk-ant-(?:api|oat|sid)[0-9]{0,2}-[A-Za-z0-9_-]{20,}")),
    # OpenAI: sk-..., sk-proj-..., sk-svcacct-..., sk-admin-... Negative
    # lookahead on "ant-" so it never swallows an Anthropic token.
    ("openai", re.compile(r"\bsk-(?!ant-)(?:proj|svcacct|admin)?-?[A-Za-z0-9_-]{20,}")),
    # GitHub: classic ghp_, fine-grained github_pat_, plus gho_/ghu_/ghs_/ghr_
    ("github", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}")),
    ("github", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}")),
    # Google API keys: AIza...
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    # Google OAuth client secret: GOCSPX-...
    ("google_oauth", re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{20,}")),
    # AWS access key id + secret
    ("aws", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}")),
    # Slack tokens
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    # Stripe live/test keys
    ("stripe", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}")),
    # Generic bearer/JWT-ish: header.payload.signature (three b64url segments)
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # PEM private keys (multi-line) — grab the whole block.
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
]

# Tokens that *look* like high-entropy secrets but ride after a telltale
# assignment keyword. Catches `api_key="...."`, `password: ....`, `token=....`
# for shapes the provider patterns miss.
_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    \b(?:api[_-]?key|secret|token|password|passwd|pwd|bearer|authorization|access[_-]?token|client[_-]?secret)
    \b\s*[:=]\s*
    ['"]?(?P<val>[A-Za-z0-9_\-\.+/=]{16,})['"]?
    """
)

_REDACTION = "[REDACTED:{label}]"
# Minimum Shannon entropy (bits/char) for the assignment fallback to fire.
_ENTROPY_THRESHOLD = 3.2
_MIN_ENTROPY_LEN = 16


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class ScrubResult:
    text: str
    hit_types: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.hit_types)


def scrub_text(text: str | None) -> ScrubResult:
    """
    Redact secrets in `text`, returning the cleaned text and the distinct
    hit types found. Idempotent: running it on already-scrubbed text is a
    no-op (the [REDACTED:...] markers contain no secret material).
    """
    if not text:
        return ScrubResult(text=text or "", hit_types=[])

    hits: list[str] = []
    out = text

    # Pass 1: provider-specific patterns.
    for label, pattern in _PROVIDER_PATTERNS:
        def _sub(_m: re.Match[str], _label: str = label) -> str:
            hits.append(_label)
            return _REDACTION.format(label=_label)

        out = pattern.sub(_sub, out)

    # Pass 2: assignment fallback with entropy gate (avoid nuking ordinary
    # words that follow "password:").
    def _assign_sub(m: re.Match[str]) -> str:
        val = m.group("val")
        if len(val) >= _MIN_ENTROPY_LEN and _shannon_entropy(val) >= _ENTROPY_THRESHOLD:
            hits.append("generic")
            return m.group(0).replace(val, _REDACTION.format(label="generic"))
        return m.group(0)

    out = _ASSIGNMENT_PATTERN.sub(_assign_sub, out)

    # Distinct, stable order.
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h, None)
    return ScrubResult(text=out, hit_types=list(seen.keys()))


def contains_secret(text: str | None) -> bool:
    """Cheap boolean check — does this text carry a detectable secret?"""
    return scrub_text(text).found
