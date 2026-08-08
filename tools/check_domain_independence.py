"""Enforce Constitution Article 5: the core engine is domain independent.

The engine may not import from a plugin, branch on a plugin's identity, or
contain the name of a scientific field anywhere in its source. That rule is
the project's central architectural claim, so it is checked mechanically
rather than left to reviewer attention.

Two classes of violation are detected:

1. Forbidden imports -- the engine importing any package that sits on the
   plugin side of the contract.
2. Domain vocabulary -- a scientific field, unit, or physical constant
   appearing in an identifier, string, docstring, or comment.

Escape hatch: append ``# domain-independence: allow -- <reason>`` (or the
short form ``# di: allow -- <reason>``) to a line to suppress it. A reason is
mandatory. Suppressions are reported in the summary so they stay visible and
auditable rather than accumulating silently.

Usage::

    python tools/check_domain_independence.py            # checks engine/
    python tools/check_domain_independence.py --path pkg # checks another tree

Exits 0 when clean, 1 when any violation is found.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Packages that sit on the plugin side of the contract. The engine defines the
# interfaces these implement; importing them inverts the dependency direction
# and breaks Constitution Article 9.
FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "plugins",
        "algorithms",
        "constraints",
        "validators",
        "reports",
        "visualization",
        "physics_discovery",  # legacy tree, pending migration
    }
)

# Scientific vocabulary that must not appear in the core. Matched
# case-insensitively on word boundaries. Prefixes are used where a term has
# many inflections (``thermodynamic`` covers ``thermodynamics``).
DOMAIN_LEXICON: tuple[str, ...] = (
    # fields
    "physics",
    "physical",
    "chemistry",
    "chemical",
    "biology",
    "biological",
    "biochem",
    "climate",
    "climatic",
    "medicine",
    "medical",
    "astronomy",
    "astrophysic",
    "geology",
    "epidemiolog",
    "pharmac",
    "genomic",
    "thermodynamic",
    "quantum",
    "relativistic",
    "ecology",
    # named benchmarks and domains
    "feynman",
    "ocean",
    "atmospheric",
    "carbon",
    "molecule",
    "molecular",
    "protein",
    "particle",
    # units and constants
    "joule",
    "kelvin",
    "pascal",
    "avogadro",
    "boltzmann",
    "planck",
    "coulomb",
    "angstrom",
    "electronvolt",
)

# The leading boundary is written as a lookbehind rather than ``\b`` because
# ``_`` is a word character: ``\bchemistry`` does not match inside
# ``load_chemistry_data``, which is exactly where domain vocabulary hides in
# Python. There is deliberately no trailing boundary, so a prefix in the
# lexicon covers its inflections (``thermodynamic`` -> ``thermodynamics``).
_LEXICON_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(term) for term in DOMAIN_LEXICON) + r")",
    re.IGNORECASE,
)

_PRAGMA_RE = re.compile(
    r"#\s*(?:domain-independence|di)\s*:\s*allow\s*(?:--|—)\s*(?P<reason>\S.*)$"
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    kind: str
    detail: str

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: [{self.kind}] {self.detail}"


@dataclass(frozen=True)
class Suppression:
    path: Path
    line: int
    reason: str


def _suppressed_lines(source: str) -> dict[int, str]:
    """Map 1-indexed line numbers carrying an allow pragma to their reason."""
    found: dict[int, str] = {}
    for index, text in enumerate(source.splitlines(), start=1):
        match = _PRAGMA_RE.search(text)
        if match:
            found[index] = match.group("reason").strip()
    return found


def _import_roots(node: ast.AST) -> list[tuple[str, int]]:
    """Extract (root package, line) for every import statement in a node."""
    roots: list[tuple[str, int]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            roots.append((alias.name.split(".")[0], node.lineno))
    # A relative import (level > 0) stays inside the current package and cannot
    # reach a sibling top-level package, so it is never a violation.
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        roots.append((node.module.split(".")[0], node.lineno))
    return roots


def check_file(path: Path) -> tuple[list[Violation], list[Suppression]]:
    source = path.read_text(encoding="utf-8")
    suppressed = _suppressed_lines(source)

    violations: list[Violation] = []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        violations.append(Violation(path, exc.lineno or 0, "syntax", f"could not parse: {exc.msg}"))
        return violations, []

    for node in ast.walk(tree):
        for root, lineno in _import_roots(node):
            if root in FORBIDDEN_IMPORT_ROOTS and lineno not in suppressed:
                violations.append(
                    Violation(
                        path,
                        lineno,
                        "forbidden-import",
                        f"engine must not import {root!r}; dependencies flow "
                        f"through the plugin contract, not directly",
                    )
                )

    # Domain vocabulary is checked against raw text rather than the AST so that
    # comments and docstrings are covered too -- a domain name in a comment is
    # still the engine knowing about a domain.
    for lineno, text in enumerate(source.splitlines(), start=1):
        if lineno in suppressed:
            continue
        for match in _LEXICON_RE.finditer(text):
            violations.append(
                Violation(
                    path,
                    lineno,
                    "domain-vocabulary",
                    f"scientific term {match.group(1)!r} in the domain-independent "
                    f"core; move it to a plugin or suppress with a stated reason",
                )
            )

    records = [Suppression(path, line, reason) for line, reason in suppressed.items()]
    return violations, records


def check_tree(root: Path) -> tuple[list[Violation], list[Suppression]]:
    violations: list[Violation] = []
    suppressions: list[Suppression] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        file_violations, file_suppressions = check_file(path)
        violations.extend(file_violations)
        suppressions.extend(file_suppressions)
    return violations, suppressions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that a package tree contains no scientific domain knowledge."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("engine"),
        help="package tree to check (default: engine)",
    )
    args = parser.parse_args(argv)

    target: Path = args.path
    if not target.is_dir():
        print(f"error: {target} is not a directory", file=sys.stderr)
        return 2

    repo_root = Path.cwd()
    violations, suppressions = check_tree(target)

    if suppressions:
        print(f"{len(suppressions)} suppression(s) in {target}:")
        for record in suppressions:
            try:
                shown = record.path.relative_to(repo_root)
            except ValueError:
                shown = record.path
            print(f"  {shown}:{record.line}: allowed -- {record.reason}")
        print()

    if violations:
        print(
            f"Constitution Article 5 violated: {len(violations)} finding(s) in {target}",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation.render(repo_root)}", file=sys.stderr)
        print(
            "\nThe core engine must contain no scientific content. If a finding is a "
            "false positive, suppress the line with:\n"
            "    # domain-independence: allow -- <reason>",
            file=sys.stderr,
        )
        return 1

    print(f"{target}: domain independent (Constitution Article 5 upheld)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
