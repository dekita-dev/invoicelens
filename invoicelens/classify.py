"""Categorization and deductibility, driven by a user-editable YAML file.

Tax rules differ by jurisdiction and by year, so they must not be hardcoded.
This module only does keyword matching; what the keywords mean is the user's
call, expressed in the rules file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rules", "rules.example.yaml",
)


@dataclass
class Category:
    name: str
    label: str
    keywords: list = field(default_factory=list)
    deductible: object = None          # True / False / None (unknown)
    note: str = ""


@dataclass
class RuleSet:
    categories: list = field(default_factory=list)
    default_deductible: object = None
    default_note: str = "No rule matched. Review manually."
    source: str = ""


def load_rules(path=None):
    import yaml

    path = path or DEFAULT_RULES_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError("rules file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rules = RuleSet(source=path)
    default = data.get("default") or {}
    rules.default_deductible = default.get("deductible")
    rules.default_note = default.get("note", rules.default_note)

    for entry in data.get("categories") or []:
        rules.categories.append(Category(
            name=entry.get("name", ""),
            label=entry.get("label") or entry.get("name", ""),
            keywords=[str(k).lower() for k in entry.get("keywords") or []],
            deductible=entry.get("deductible"),
            note=entry.get("note", ""),
        ))
    if not rules.categories:
        raise ValueError("rules file has no categories: %s" % path)
    return rules


def _haystack(fields):
    parts = [str(fields.get("vendor") or "")]
    for item in fields.get("line_items") or []:
        parts.append(str(item.get("description") or ""))
    parts.append(str(fields.get("notes") or ""))
    return " ".join(parts).lower()


def classify(fields, rules):
    """Return (category, deductible, note) for extracted *fields*."""
    hay = _haystack(fields)
    best = None
    best_score = 0
    for cat in rules.categories:
        score = sum(1 for kw in cat.keywords if kw and kw in hay)
        if score > best_score:
            best, best_score = cat, score
    if best is None:
        return ("uncategorized", rules.default_deductible, rules.default_note)
    return (best.name, best.deductible, best.note)
