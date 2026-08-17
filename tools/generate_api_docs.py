# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Generate the dependency-free public API reference from Python source."""

from __future__ import annotations

import argparse
import ast
from datetime import date
from html import escape
from pathlib import Path
import re


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PUBLIC_ROOT / "src" / "mmWaveRadar"
DEFAULT_OUTPUT = PUBLIC_ROOT / "docs" / "api" / "index.html"


def _summary(docstring: str | None) -> str:
    if not docstring:
        return "Public API declarations for this module."
    paragraphs = re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)
    return " ".join(line.strip() for line in paragraphs[0].splitlines())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    clone_type = (
        ast.AsyncFunctionDef
        if isinstance(node, ast.AsyncFunctionDef)
        else ast.FunctionDef
    )
    clone = clone_type(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=node.returns,
        type_comment=None,
    )
    ast.fix_missing_locations(clone)
    declaration = ast.unparse(clone).splitlines()[0]
    return declaration.removesuffix(":")


def _class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    bases.extend(
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        for keyword in node.keywords
    )
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}"


def _public_entries(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield "function", node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield "class", node


def _method_table(node: ast.ClassDef) -> str:
    methods = [
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not child.name.startswith("_")
    ]
    if not methods:
        return ""
    rows = "".join(
        "<tr><td><code>"
        + escape(_function_signature(method))
        + "</code></td><td>"
        + escape(_summary(ast.get_docstring(method)))
        + "</td></tr>"
        for method in methods
    )
    return (
        "<details><summary>Public methods and properties</summary>"
        "<table><thead><tr><th>Signature</th><th>Description</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


def _module_records():
    records = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(SOURCE_ROOT)
        module = "mmWaveRadar." + ".".join(relative.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        entries = list(_public_entries(tree))
        records.append((module, relative, tree, entries))
    return records


def _render_entry(module: str, kind: str, node: ast.AST) -> str:
    name = node.name
    anchor = _slug(f"{module}-{name}")
    signature = (
        _class_signature(node)
        if isinstance(node, ast.ClassDef)
        else _function_signature(node)
    )
    methods = _method_table(node) if isinstance(node, ast.ClassDef) else ""
    return f"""
      <article class="api-card" id="{anchor}">
        <div class="api-title"><span class="kind">{kind}</span>
          <h4>{escape(name)}</h4><a href="#{anchor}">#</a></div>
        <pre><code>{escape(signature)}</code></pre>
        <p>{escape(_summary(ast.get_docstring(node)))}</p>{methods}
      </article>"""


def generate(*, updated: str) -> str:
    records = _module_records()
    callable_count = sum(len(entries) for _module, _path, _tree, entries in records)
    packages: dict[str, list[tuple]] = {}
    for record in records:
        module = record[0]
        parts = module.split(".")
        package = ".".join(parts[:2]) if len(parts) > 2 else "mmWaveRadar"
        packages.setdefault(package, []).append(record)

    navigation = []
    content = []
    for package, modules in packages.items():
        package_anchor = _slug(package)
        navigation.append(
            f'<a class="nav-package" href="#{package_anchor}">{escape(package)}</a>'
        )
        module_sections = []
        for module, relative, tree, entries in modules:
            module_anchor = _slug(module)
            navigation.append(
                f'<a class="nav-module" href="#{module_anchor}">{escape(module.split(".")[-1])}</a>'
            )
            cards = "".join(
                _render_entry(module, kind, node) for kind, node in entries
            ) or '<p class="empty">No public classes or functions.</p>'
            module_sections.append(f"""
              <section class="module-section" id="{module_anchor}">
                <div class="module-header"><div><h3>{escape(module)}</h3>
                  <p>{escape(_summary(ast.get_docstring(tree)))}</p></div>
                  <a class="source-link" href="../../src/mmWaveRadar/{relative.as_posix()}">source</a>
                </div>{cards}
              </section>""")
        content.append(f"""
          <section class="package-section" id="{package_anchor}">
            <h2>{escape(package)}</h2>{''.join(module_sections)}
          </section>""")

    return f"""<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HERMES Developer API Reference</title>
<style>
:root{{--bg:#07111f;--surface:#101d30;--surface2:#15253a;--text:#e7eef8;--muted:#9fb0c4;--accent:#38bdf8;--border:#29415d}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 Inter,system-ui,sans-serif}}
.layout{{display:grid;grid-template-columns:285px minmax(0,1fr)}}nav{{position:sticky;top:0;height:100vh;overflow:auto;padding:24px 18px;background:#0b1728;border-right:1px solid var(--border)}}
nav h2{{margin:0 0 14px}}nav a{{display:block;color:var(--muted);text-decoration:none;padding:5px 8px;border-radius:6px}}nav a:hover{{color:var(--text);background:var(--surface2)}}.nav-package{{color:var(--accent);font-weight:700;margin-top:12px}}.nav-module{{padding-left:18px}}
main{{min-width:0}}.hero{{padding:50px clamp(24px,5vw,72px) 34px;background:linear-gradient(135deg,#10243c,#07111f);border-bottom:1px solid var(--border)}}h1{{font-size:clamp(32px,5vw,52px);margin:0 0 12px}}.hero p{{max-width:850px;color:var(--muted)}}.meta{{display:flex;gap:10px;flex-wrap:wrap}}.pill,.kind{{border:1px solid var(--border);border-radius:999px;padding:4px 10px;color:var(--accent);font-size:13px}}
.content{{padding:28px clamp(24px,5vw,72px) 70px}}.package-section>h2{{margin-top:48px;border-bottom:1px solid var(--border);padding-bottom:8px}}.module-section{{scroll-margin-top:20px;margin:30px 0}}.module-header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}.module-header h3{{margin:0}}.module-header p{{color:var(--muted);max-width:900px}}.source-link{{color:var(--accent)}}
.api-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;margin:13px 0}}.api-title{{display:flex;gap:10px;align-items:center}}.api-title h4{{font-size:19px;margin:0;flex:1}}.api-title a{{color:var(--muted);text-decoration:none}}pre{{overflow:auto;background:#07101c;border-radius:7px;padding:12px}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}details{{margin-top:12px}}table{{border-collapse:collapse;width:100%;margin-top:8px}}th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid var(--border)}}th{{color:var(--muted)}}.empty{{color:var(--muted)}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}nav{{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--border)}}.nav-module{{display:none}}}}
</style></head><body><div class="layout"><nav><h2>API Reference</h2>
<a href="#overview">Overview</a>{''.join(navigation)}</nav><main>
<header class="hero" id="overview"><h1>HERMES Developer API Reference</h1>
<p>Public Python classes and functions for radar hardware, targets, simulation,
DSP, experiments, bundle measurements, diagnostics, visualization, and the local GUI.</p>
<div class="meta"><span class="pill">Updated {escape(updated)}</span>
<span class="pill">{callable_count} public classes/functions</span>
<span class="pill">{len(records)} modules</span></div></header>
<div class="content">{''.join(content)}</div></main></div></body></html>
"""


def main() -> int:
    today = date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--date",
        default=f"{today:%B} {today.day}, {today:%Y}",
        help="Displayed documentation update date.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(updated=args.date), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
