"""Architecture tests: the dependency rules are enforced automatically, not just documented."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "metricmind"
HEAVY = {"pandas", "sklearn", "scipy", "fastapi", "matplotlib", "seaborn", "joblib"}


def imports(layer: str) -> list[tuple[Path, str]]:
    found = []
    for path in (SRC / layer).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found += [(path, a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append((path, node.module))
    return found


def offenders(layer, forbidden_roots=(), forbidden_prefixes=()):
    return [
        f"{p.relative_to(SRC)} imports {m}"
        for p, m in imports(layer)
        if m.split(".")[0] in forbidden_roots or m.startswith(tuple(forbidden_prefixes))
    ]


def test_domain_has_no_framework_dependencies():
    assert not offenders("domain", HEAVY)


def test_domain_depends_on_nothing_outward():
    assert not offenders(
        "domain",
        forbidden_prefixes=(
            "metricmind.application",
            "metricmind.infrastructure",
            "metricmind.interfaces",
            "metricmind.config",
        ),
    )


def test_application_does_not_know_concrete_libraries_or_infrastructure():
    assert not offenders(
        "application",
        {"sklearn", "scipy", "fastapi", "matplotlib", "seaborn", "joblib"},
        ("metricmind.infrastructure", "metricmind.interfaces"),
    )


@pytest.mark.parametrize("layer", ["infrastructure"])
def test_infrastructure_does_not_depend_on_interfaces(layer):
    assert not offenders(layer, forbidden_prefixes=("metricmind.interfaces",))
