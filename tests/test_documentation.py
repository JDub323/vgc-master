"""Documentation coverage/alignment gates for the reviewer-facing contracts."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def production_python_files():
    """Return production ``.py`` paths, excluding tests, artifacts, and venvs."""
    files = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") or part in {"artifacts", "tests",
                                                "node_modules"}
               for part in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def declared_functions(path):
    """Yield ``(qualified_name, leaf_name, has_docstring)`` outside closures."""
    tree = ast.parse(path.read_text())

    def walk(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                yield from walk(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + node.name, node.name, ast.get_docstring(node) is not None

    yield from walk(tree.body)


def test_production_functions_are_documented():
    """Every production function must have inline docs or a catalog contract."""
    catalog = (ROOT / "DATA_CONTRACTS.md").read_text()
    missing = []
    for path in production_python_files():
        if path.name == "contracts.py":
            continue
        for qualified, leaf, inline in declared_functions(path):
            owner = qualified.rpartition(".")[0]
            documented = (
                inline or qualified in catalog or f"`{leaf}" in catalog or
                f", `{leaf}" in catalog or
                (leaf == "__init__" and owner + "(" in catalog)
            )
            if not documented:
                missing.append(f"{path.relative_to(ROOT)}:{qualified}")
    assert not missing, "missing input/output contracts:\n" + "\n".join(missing)


def test_every_production_module_is_referenced():
    """Every non-empty production module must appear in reviewer documentation."""
    docs = ((ROOT / "README.md").read_text() + "\n" +
            (ROOT / "DATA_CONTRACTS.md").read_text())
    missing = []
    for path in production_python_files():
        rel = path.relative_to(ROOT)
        if path.name == "__init__.py":
            continue
        if str(rel) not in docs and path.stem not in docs:
            missing.append(str(rel))
    assert not missing, "undocumented modules:\n" + "\n".join(missing)


def test_architecture_docs_have_no_known_stale_claims():
    """Block terminology from the pre-AgentSpec/pre-layout-3 architecture."""
    reviewed_docs = ("README.md", "DATA_CONTRACTS.md", "commands.txt", "colab.ipynb")
    text = "\n".join(path.read_text() for path in production_python_files())
    text += "\n" + "\n".join((ROOT / name).read_text() for name in reviewed_docs)
    stale = (
        "Layout 2 (current",
        "layout-2 tokenizer",
        "phase-2 tokenizer layout",
        "Phase 3 (next",
        "the Python Searcher implementation is still",
        "Still current-code behavior",
        "Passed to Searcher AS the model",
        "immutable model archives",
        "Five changes, largest first",
        "Reconstruction drops volatiles: choice lock, Protect streaks",
        "belief particles use\n  0 stat points",
    )
    found = [phrase for phrase in stale if phrase in text]
    assert not found, f"stale architecture documentation: {found}"


def test_colab_notebook_is_valid_json():
    """Keep the documented Colab workflow loadable after documentation edits."""
    with (ROOT / "colab.ipynb").open() as fh:
        notebook = json.load(fh)
    assert notebook.get("nbformat") == 4


if __name__ == "__main__":
    test_production_functions_are_documented()
    test_every_production_module_is_referenced()
    test_architecture_docs_have_no_known_stale_claims()
    test_colab_notebook_is_valid_json()
    print("all documentation contract tests passed")
