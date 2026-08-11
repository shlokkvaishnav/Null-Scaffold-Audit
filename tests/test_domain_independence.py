"""Tests for the Article 5 enforcement check.

The checker is the mechanism that turns Constitution Article 5 from a
convention into a guarantee, so its own failure modes matter more than most.
Two are tested with particular care:

* A checker that misses violations gives false assurance -- worse than no
  checker, because it is trusted.
* A checker that fires on legitimate code gets suppressed wholesale, which
  ends in the same place by a different route.

The module is loaded by path rather than imported, because ``tools/`` is
repository tooling and is deliberately not an installed package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "tools" / "check_domain_independence.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_domain_independence", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def kinds(violations: list) -> list[str]:
    return [v.kind for v in violations]


# --------------------------------------------------------------------------
# Forbidden imports
# --------------------------------------------------------------------------


# A package name can trip both rules at once -- a root that is also a domain
# forbidden root and a domain word -- so these assert on membership rather than
# on the exact finding list.
@pytest.mark.parametrize("root", sorted(checker.FORBIDDEN_IMPORT_ROOTS))
def test_plain_import_of_a_plugin_package_is_a_violation(tmp_path: Path, root: str) -> None:
    path = write(tmp_path, "mod.py", f"import {root}\n")
    violations, _ = checker.check_file(path)
    assert "forbidden-import" in kinds(violations)


@pytest.mark.parametrize("root", sorted(checker.FORBIDDEN_IMPORT_ROOTS))
def test_from_import_of_a_plugin_package_is_a_violation(tmp_path: Path, root: str) -> None:
    path = write(tmp_path, "mod.py", f"from {root}.thing import Widget\n")
    violations, _ = checker.check_file(path)
    assert "forbidden-import" in kinds(violations)


def test_import_inside_a_function_is_still_a_violation(tmp_path: Path) -> None:
    """A deferred import is still a dependency; it is only a later one."""
    source = "def build():\n    import plugins\n    return plugins\n"
    path = write(tmp_path, "mod.py", source)
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["forbidden-import"]


def test_aliased_import_is_a_violation(tmp_path: Path) -> None:
    path = write(tmp_path, "mod.py", "import algorithms.search as s\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["forbidden-import"]


def test_relative_imports_are_never_violations(tmp_path: Path) -> None:
    """A relative import cannot reach a sibling top-level package.

    ``from . import plugins`` refers to a submodule of the current package,
    not to the repository-level ``plugins/``. Flagging it would push authors
    toward blanket suppressions.
    """
    source = "from . import plugins\nfrom .plugins import thing\nfrom ..reports import x\n"
    path = write(tmp_path, "mod.py", source)
    violations, _ = checker.check_file(path)
    assert violations == []


def test_similarly_named_package_is_not_a_violation(tmp_path: Path) -> None:
    """Matching is on the root package, not on a substring of it."""
    path = write(tmp_path, "mod.py", "import plugins_registry_stub\n")
    violations, _ = checker.check_file(path)
    assert violations == []


def test_permitted_imports_are_clean(tmp_path: Path) -> None:
    source = (
        "import json\n"
        "from pathlib import Path\n"
        "import pydantic\n"
        "from engine.plugin import Contract\n"
    )
    path = write(tmp_path, "mod.py", source)
    violations, _ = checker.check_file(path)
    assert violations == []


# --------------------------------------------------------------------------
# Domain vocabulary
# --------------------------------------------------------------------------


def test_domain_term_in_an_identifier_is_a_violation(tmp_path: Path) -> None:
    path = write(tmp_path, "mod.py", "def load_chemistry_data():\n    return None\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["domain-vocabulary"]


@pytest.mark.parametrize(
    "identifier",
    ["load_chemistry_data", "OCEAN_DEFAULTS", "run_feynman_suite", "flux_carbon", "_biology"],
)
def test_domain_term_inside_a_snake_case_identifier_is_a_violation(
    tmp_path: Path, identifier: str
) -> None:
    """Regression: ``_`` is a word character, so ``\\b`` missed mid-identifier terms.

    This is where domain vocabulary actually hides in Python. A checker that
    only catches terms at the start of a name gives false assurance.
    """
    path = write(tmp_path, "mod.py", f"{identifier} = 1\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["domain-vocabulary"]


def test_domain_term_in_a_comment_is_a_violation(tmp_path: Path) -> None:
    """A domain name in a comment is still the core knowing about a domain."""
    path = write(tmp_path, "mod.py", "x = 1  # used by the ocean model\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["domain-vocabulary"]


def test_domain_term_in_a_docstring_is_a_violation(tmp_path: Path) -> None:
    path = write(tmp_path, "mod.py", '"""Runs the Feynman benchmark suite."""\n')
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["domain-vocabulary"]


def test_domain_term_matching_is_case_insensitive(tmp_path: Path) -> None:
    path = write(tmp_path, "mod.py", "# PHYSICS\n# Physics\n# physics\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["domain-vocabulary"] * 3


def test_every_occurrence_on_a_line_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, "mod.py", "# quantum chemistry\n")
    violations, _ = checker.check_file(path)
    assert len(violations) == 2


def test_domain_terms_match_on_a_word_boundary(tmp_path: Path) -> None:
    """``oceanography`` is a domain word; ``locean`` is not one at all."""
    flagged = write(tmp_path, "a.py", "# oceanography\n")
    clean = write(tmp_path, "b.py", "# the locean variable\n")
    assert kinds(checker.check_file(flagged)[0]) == ["domain-vocabulary"]
    assert checker.check_file(clean)[0] == []


def test_domain_neutral_source_is_clean(tmp_path: Path) -> None:
    source = (
        '"""Orchestrate a search over candidate hypotheses."""\n'
        "\n"
        "def rank(candidates, score):\n"
        "    return sorted(candidates, key=score)\n"
    )
    path = write(tmp_path, "mod.py", source)
    violations, suppressions = checker.check_file(path)
    assert violations == []
    assert suppressions == []


# --------------------------------------------------------------------------
# The escape hatch
# --------------------------------------------------------------------------


def test_pragma_suppresses_a_forbidden_import(tmp_path: Path) -> None:
    source = "import plugins  # domain-independence: allow -- tracked by ADR-0001\n"
    path = write(tmp_path, "mod.py", source)
    violations, suppressions = checker.check_file(path)
    assert violations == []
    assert [s.reason for s in suppressions] == ["tracked by ADR-0001"]


def test_pragma_suppresses_domain_vocabulary(tmp_path: Path) -> None:
    source = "URL = 'https://example.org/quantum'  # di: allow -- external URL, not a concept\n"
    path = write(tmp_path, "mod.py", source)
    violations, suppressions = checker.check_file(path)
    assert violations == []
    assert [s.reason for s in suppressions] == ["external URL, not a concept"]


def test_pragma_without_a_reason_does_not_suppress(tmp_path: Path) -> None:
    """A suppression with no stated reason cannot be reviewed, so it does not count."""
    path = write(tmp_path, "mod.py", "import plugins  # domain-independence: allow\n")
    violations, suppressions = checker.check_file(path)
    assert kinds(violations) == ["forbidden-import"]
    assert suppressions == []


def test_pragma_applies_only_to_its_own_line(tmp_path: Path) -> None:
    source = "import plugins  # di: allow -- reviewed\nimport reports\n"
    path = write(tmp_path, "mod.py", source)
    violations, suppressions = checker.check_file(path)
    assert kinds(violations) == ["forbidden-import"]
    assert violations[0].line == 2
    assert len(suppressions) == 1


def test_multiline_import_is_suppressed_at_the_statement_line(tmp_path: Path) -> None:
    """AST reports the statement's first line, so the pragma belongs there."""
    source = "from plugins import (  # di: allow -- reviewed\n    Widget,\n)\n"
    path = write(tmp_path, "mod.py", source)
    violations, _ = checker.check_file(path)
    assert violations == []


# --------------------------------------------------------------------------
# Tree walking and the command-line entry point
# --------------------------------------------------------------------------


def test_unparseable_source_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """A file the checker cannot read must never be silently treated as clean."""
    path = write(tmp_path, "mod.py", "def broken(:\n")
    violations, _ = checker.check_file(path)
    assert kinds(violations) == ["syntax"]


def test_check_tree_skips_bytecode_caches(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "stale.py").write_text("import plugins\n", encoding="utf-8")
    write(tmp_path, "clean.py", "x = 1\n")
    violations, _ = checker.check_tree(tmp_path)
    assert violations == []


def test_check_tree_aggregates_across_nested_packages(tmp_path: Path) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    write(tmp_path, "a.py", "import plugins\n")
    (nested / "b.py").write_text("# biology\n", encoding="utf-8")
    violations, _ = checker.check_tree(tmp_path)
    assert sorted(kinds(violations)) == ["domain-vocabulary", "forbidden-import"]


def test_main_exits_zero_on_a_clean_tree(tmp_path: Path) -> None:
    write(tmp_path, "mod.py", "def rank(items):\n    return sorted(items)\n")
    assert checker.main(["--path", str(tmp_path)]) == 0


def test_main_exits_one_on_a_violation(tmp_path: Path) -> None:
    write(tmp_path, "mod.py", "import plugins\n")
    assert checker.main(["--path", str(tmp_path)]) == 1


def test_main_exits_two_when_the_target_is_missing(tmp_path: Path) -> None:
    """A missing directory must not be mistaken for a directory with no violations."""
    assert checker.main(["--path", str(tmp_path / "absent")]) == 2


def test_main_reports_suppressions_on_an_otherwise_clean_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Suppressions stay visible, so they can be audited instead of accumulating."""
    write(tmp_path, "mod.py", "import plugins  # di: allow -- tracked by ADR-0001\n")
    assert checker.main(["--path", str(tmp_path)]) == 0
    assert "tracked by ADR-0001" in capsys.readouterr().out
