from __future__ import annotations

# AI-assisted maintenance note:
# This script generates the repository's static dependency graph artifacts used
# by the documentation under `docs/dependency_graphs/`.
#
# Purpose:
# provide one repeatable place to answer architecture questions such as:
# - which production modules import which other modules?
# - which packages depend on each other?
# - which production modules are exercised directly by the tests?
#
# Context:
# the output is intentionally static and source-based rather than runtime-based.
# That makes the graphs cheap to regenerate in CI or local documentation work
# and avoids requiring the project to import every module successfully at graph
# generation time.
#
# Important note:
# this script focuses on import relationships, not execution-time call graphs.
# It is therefore best read as an architectural map for maintenance and review,
# not as a precise trace of every runtime code path.

import argparse
import ast
import json
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "dependency_graphs"
DEFAULT_PACKAGE_ASSET = REPO_ROOT / "docs" / "assets" / "codebase_dependency_packages.svg"
INTERNAL_PREFIXES = {
    "config",
    "data",
    "environment",
    "evaluation",
    "models",
    "observability",
    "utils",
    "workflows",
    "train",
    "defaults",
    "main",
    "tests",
    "reporting",
}
ENTRYPOINT_MODULES = {"main", "defaults"}
PACKAGE_FACADES = {"config", "environment", "evaluation", "observability", "workflows"}


@dataclass(frozen=True)
class ModuleNode:
    """
    Metadata describing one Python module in the repository graph.

    Context:
    the graph writers need more than a raw module name. They also need the
    source path, the top-level package, and lightweight classification labels
    so production, test, package-facade, and entrypoint nodes can be styled and
    summarized differently.
    """

    name: str
    path: str
    package: str
    category: str
    layer: str


@dataclass(frozen=True)
class DependencyEdge:
    """
    Directed import edge between two internal modules or packages.

    Context:
    the edge type distinguishes ordinary imports from package-facade re-exports
    and from the derived test-to-production dependency view used in the docs.
    """

    source: str
    target: str
    edge_type: str
    line: int | None = None


def _normalize_module_name(path: Path) -> str:
    """
    Convert a repository file path into the canonical internal module name.

    Context:
    the docs treat `src/foo/bar.py` as the importable module `foo.bar`, while
    `__init__.py` files represent the package itself rather than a literal
    `.__init__` suffix.
    """
    relative = path.relative_to(REPO_ROOT)
    without_suffix = relative.with_suffix("")
    if without_suffix.parts[:1] == ("src",):
        without_suffix = Path(*without_suffix.parts[1:])
    if without_suffix.name == "__init__":
        return ".".join(without_suffix.parts[:-1])
    return ".".join(without_suffix.parts)


def _node_category(module_name: str, path: Path, *, is_test: bool) -> str:
    """
    Classify a module for graph styling and summary generation.

    Context:
    entrypoints, package facades, tests, and ordinary implementation modules
    are rendered differently so the generated SVGs communicate architectural
    role at a glance rather than showing every node as the same generic shape.
    """
    if module_name in ENTRYPOINT_MODULES:
        return "entrypoint"
    if path.name == "__init__.py":
        return "package"
    if is_test:
        return "test_module"
    return "module"


def _iter_python_files(root: Path) -> Iterable[Path]:
    """
    Yield the Python files that belong to the documented repository surface.

    Context:
    the explicit root entrypoints are yielded first because they sit outside
    `src/`, while the rest of the graph comes from the production and test
    trees. This keeps the graph aligned with the repo's runnable and validated
    code, not with generated assets under `docs/`.
    """
    explicit_files = [root / "main.py", root / "defaults.py"]
    for path in explicit_files:
        if path.exists():
            yield path
    for subdir in (root / "src", root / "tests"):
        if subdir.exists():
            yield from sorted(p for p in subdir.rglob("*.py") if ".git" not in p.parts)


def _import_target(module_name: str | None) -> str | None:
    """
    Keep only internal import targets that belong in the graph.

    Context:
    external libraries would dominate the graph while saying little about the
    repository's own architecture, so this filter narrows the edge set to
    project-owned modules and packages.
    """
    if not module_name:
        return None
    prefix = module_name.split(".")[0]
    if prefix not in INTERNAL_PREFIXES:
        return None
    return module_name


def discover_nodes() -> tuple[dict[str, ModuleNode], dict[Path, str]]:
    """
    Discover every graph node and remember which file produced which module.

    Context:
    the returned path-to-module map is reused by edge discovery so the AST pass
    can translate imports back into the normalized module names used by the
    generated artifacts.
    """
    nodes: dict[str, ModuleNode] = {}
    path_to_module: dict[Path, str] = {}
    for path in _iter_python_files(REPO_ROOT):
        is_test = "tests" in path.parts
        module_name = _normalize_module_name(path)
        package = module_name.split(".")[0]
        category = _node_category(module_name, path, is_test=is_test)
        layer = "test" if is_test else "production"
        nodes[module_name] = ModuleNode(
            name=module_name,
            path=str(path.relative_to(REPO_ROOT)),
            package=package,
            category=category,
            layer=layer,
        )
        path_to_module[path] = module_name
    return nodes, path_to_module


def discover_edges(path_to_module: dict[Path, str]) -> list[DependencyEdge]:
    """
    Parse repository files and collect the internal import edges they declare.

    Context:
    this pass reads import statements statically from the AST. It does not
    execute the code, which keeps graph generation robust even for optional
    dependency paths or modules that need runtime setup.
    """
    edges: list[DependencyEdge] = []
    for path, source_module in path_to_module.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _import_target(alias.name)
                    if target:
                        edges.append(
                            DependencyEdge(
                                source=source_module,
                                target=target,
                                edge_type="import",
                                line=node.lineno,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                target = _import_target(node.module)
                if target:
                    edge_type = (
                        "re_export"
                        if path.name == "__init__.py" and target.split(".")[0] == source_module.split(".")[0]
                        else "import"
                    )
                    edges.append(
                        DependencyEdge(
                            source=source_module,
                            target=target,
                            edge_type=edge_type,
                            line=node.lineno,
                        )
                    )
    deduped: dict[tuple[str, str, str], DependencyEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.edge_type)
        existing = deduped.get(key)
        if existing is None or (edge.line is not None and (existing.line or 0) > edge.line):
            deduped[key] = edge
    return sorted(deduped.values(), key=lambda edge: (edge.source, edge.target, edge.edge_type))


def _production_nodes(nodes: dict[str, ModuleNode]) -> dict[str, ModuleNode]:
    """Return only production-layer modules from the full node map."""
    return {name: node for name, node in nodes.items() if node.layer == "production"}


def _test_nodes(nodes: dict[str, ModuleNode]) -> dict[str, ModuleNode]:
    """Return only test-layer modules from the full node map."""
    return {name: node for name, node in nodes.items() if node.layer == "test"}


def build_package_edges(
    nodes: dict[str, ModuleNode],
    edges: Iterable[DependencyEdge],
) -> list[DependencyEdge]:
    """
    Collapse module-level production edges into package-to-package edges.

    Context:
    the package graph is meant to summarize architectural boundaries, so
    multiple imports between modules in the same package pair are represented as
    one package-level relationship.
    """
    package_edges: dict[tuple[str, str, str], DependencyEdge] = {}
    for edge in edges:
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
        if source_node is None or target_node is None:
            continue
        if source_node.layer != "production" or target_node.layer != "production":
            continue
        source_package = source_node.package
        target_package = target_node.package
        if source_package == target_package:
            continue
        key = (source_package, target_package, edge.edge_type)
        package_edges[key] = DependencyEdge(
            source=source_package,
            target=target_package,
            edge_type=edge.edge_type,
        )
    return sorted(package_edges.values(), key=lambda edge: (edge.source, edge.target, edge.edge_type))


def build_test_edges(
    nodes: dict[str, ModuleNode],
    edges: Iterable[DependencyEdge],
) -> list[DependencyEdge]:
    """
    Build the derived graph showing which tests touch which production modules.

    Context:
    this is not read directly from imports as a separate source; it is a view
    derived from the full edge set so the docs can answer "what production
    surface is covered directly by the tests?"
    """
    test_edges: dict[tuple[str, str], DependencyEdge] = {}
    for edge in edges:
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
        if source_node is None or target_node is None:
            continue
        if source_node.layer == "test" and target_node.layer == "production":
            test_edges[(edge.source, edge.target)] = DependencyEdge(
                source=edge.source,
                target=edge.target,
                edge_type="test_dependency",
                line=edge.line,
            )
    return sorted(test_edges.values(), key=lambda edge: (edge.source, edge.target))


def _quote(value: str) -> str:
    """Escape one Graphviz attribute value safely for DOT output."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _node_style(node: ModuleNode) -> dict[str, str]:
    """
    Choose Graphviz styling based on a node's architectural role.

    Context:
    the generated graphs are meant to be read quickly in docs and PR review, so
    shape/color choices carry semantic meaning instead of being cosmetic only.
    """
    if node.category == "entrypoint":
        return {"shape": "box", "style": "filled,rounded", "fillcolor": "#f4d35e", "color": "#7f6000"}
    if node.category == "package":
        return {"shape": "folder", "style": "filled", "fillcolor": "#d9eaf4", "color": "#0b3954"}
    if node.layer == "test":
        return {"shape": "box", "style": "filled,rounded", "fillcolor": "#ece4ff", "color": "#5e548e"}
    if node.package in {"workflows", "train", "data", "models"}:
        return {"shape": "box", "style": "filled,rounded", "fillcolor": "#e8f5e9", "color": "#2d6a4f"}
    return {"shape": "ellipse", "style": "filled", "fillcolor": "#f7f7f7", "color": "#6c757d"}


def write_dot(
    nodes: dict[str, ModuleNode],
    edges: Iterable[DependencyEdge],
    output_path: Path,
    *,
    title: str,
    rankdir: str,
) -> None:
    """
    Write one Graphviz DOT file for the provided node and edge set.

    Context:
    the script keeps DOT generation separate from SVG rendering so the repo
    preserves both the editable source graph and the rendered asset checked
    into documentation.
    """
    lines = [
        "digraph dependencies {",
        f'  label={_quote(title)};',
        "  labelloc=t;",
        '  fontname="Helvetica";',
        f"  rankdir={rankdir};",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#7d8597"];',
    ]
    for node_name in sorted(nodes):
        node = nodes[node_name]
        style = _node_style(node)
        attrs = {
            "label": node.name,
            **style,
        }
        attr_text = ", ".join(f"{key}={_quote(value)}" for key, value in attrs.items())
        lines.append(f"  {_quote(node.name)} [{attr_text}];")
    for edge in edges:
        if edge.source not in nodes or edge.target not in nodes:
            continue
        color = "#4f5d75" if edge.edge_type == "re_export" else "#7d8597"
        style = "dashed" if edge.edge_type in {"re_export", "test_dependency"} else "solid"
        attrs = {"color": color, "style": style}
        attr_text = ", ".join(f"{key}={_quote(value)}" for key, value in attrs.items())
        lines.append(f"  {_quote(edge.source)} -> {_quote(edge.target)} [{attr_text}];")
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n")


def render_svg(dot_path: Path, svg_path: Path) -> None:
    """
    Render a DOT file to SVG using the local Graphviz `dot` executable.

    Context:
    rendering stays in a separate helper so callers can write all DOT files
    first and then render the exact subsets they want as checked-in assets.
    """
    subprocess.run(
        ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
        cwd=REPO_ROOT,
    )


def _reachable_subgraph(
    entrypoints: set[str],
    edges: Iterable[DependencyEdge],
) -> set[str]:
    """
    Compute the nodes reachable from the given entrypoint modules.

    Context:
    this produces the narrower "entrypoint flow" graph so readers can follow
    the main runnable surfaces without the noise of the full production graph.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.source].add(edge.target)
    visited = set(entrypoints)
    queue = deque(entrypoints)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _fan_counts(nodes: dict[str, ModuleNode], edges: Iterable[DependencyEdge]) -> tuple[Counter[str], Counter[str]]:
    """
    Count incoming and outgoing production dependencies per node.

    Context:
    fan-in and fan-out are used only for the summary report, where they help
    identify central modules and broad orchestrators at a glance.
    """
    fan_out: Counter[str] = Counter()
    fan_in: Counter[str] = Counter()
    for edge in edges:
        if edge.source in nodes and edge.target in nodes:
            fan_out[edge.source] += 1
            fan_in[edge.target] += 1
    return fan_in, fan_out


def _cycle_components(nodes: dict[str, ModuleNode], edges: Iterable[DependencyEdge]) -> list[list[str]]:
    """
    Detect strongly connected components in the production dependency graph.

    Context:
    a cycle here indicates an architectural knot in the import structure. The
    summary surfaces those knots explicitly so refactors can target them later.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.source in nodes and edge.target in nodes:
            adjacency[edge.source].add(edge.target)
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    onstack: set[str] = set()
    components: list[list[str]] = []

    def strongconnect(node_name: str) -> None:
        nonlocal index
        indices[node_name] = index
        lowlink[node_name] = index
        index += 1
        stack.append(node_name)
        onstack.add(node_name)
        for neighbor in adjacency[node_name]:
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node_name] = min(lowlink[node_name], lowlink[neighbor])
            elif neighbor in onstack:
                lowlink[node_name] = min(lowlink[node_name], indices[neighbor])
        if lowlink[node_name] == indices[node_name]:
            component: list[str] = []
            while True:
                current = stack.pop()
                onstack.remove(current)
                component.append(current)
                if current == node_name:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node_name in sorted(nodes):
        if node_name not in indices:
            strongconnect(node_name)
    return sorted(components, key=lambda item: (-len(item), item))


def write_canonical_json(
    nodes: dict[str, ModuleNode],
    edges: Iterable[DependencyEdge],
    package_edges: Iterable[DependencyEdge],
    test_edges: Iterable[DependencyEdge],
    output_path: Path,
) -> None:
    """
    Persist the full graph in a deterministic JSON format.

    Context:
    the canonical JSON acts as the machine-readable source of truth behind the
    rendered assets and summary markdown, which makes later tooling or diffs
    easier to build.
    """
    payload = {
        "graph_version": 1,
        "nodes": [node.__dict__ for node in sorted(nodes.values(), key=lambda item: item.name)],
        "edges": [edge.__dict__ for edge in edges],
        "package_edges": [edge.__dict__ for edge in package_edges],
        "test_edges": [edge.__dict__ for edge in test_edges],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(
    production_nodes: dict[str, ModuleNode],
    production_edges: list[DependencyEdge],
    package_edges: list[DependencyEdge],
    test_edges: list[DependencyEdge],
    output_path: Path,
) -> None:
    """
    Write a human-readable markdown summary of the generated graph artifacts.

    Context:
    the SVGs are useful visually, but maintainers often want quick text answers
    about dependency counts, cycles, and high fan-in/fan-out modules without
    opening a graph viewer.
    """
    fan_in, fan_out = _fan_counts(production_nodes, production_edges)
    cycle_components = _cycle_components(production_nodes, production_edges)
    cross_package = Counter((edge.source, edge.target) for edge in package_edges)
    lines = [
        "# Static Dependency Graph Summary",
        "",
        "## Production Overview",
        "",
        f"- Production modules: {len(production_nodes)}",
        f"- Production dependency edges: {len(production_edges)}",
        f"- Cross-package edges: {len(package_edges)}",
        f"- Test-to-production edges: {len(test_edges)}",
        "",
        "## Cycle Status",
        "",
    ]
    if cycle_components:
        lines.append(f"- Cycles detected: {len(cycle_components)}")
        lines.extend(f"- {' -> '.join(component)}" for component in cycle_components)
    else:
        lines.append("- Cycles detected: none")
    lines.extend(["", "## Highest Fan-In", ""])
    for name, count in fan_in.most_common(8):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Highest Fan-Out", ""])
    for name, count in fan_out.most_common(8):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Package Dependencies", ""])
    for (source, target), _count in sorted(cross_package.items()):
        lines.append(f"- `{source}` -> `{target}`")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """
    Generate all dependency graph artifacts and documentation summaries.

    Context:
    this is intentionally an all-in-one regeneration command so docs authors do
    not have to remember a sequence of subcommands for JSON, DOT, SVG, and
    summary outputs.
    """
    parser = argparse.ArgumentParser(
        description="Generate static dependency graph artifacts for the repository."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated graph artifacts.",
    )
    parser.add_argument(
        "--package-asset",
        type=Path,
        default=DEFAULT_PACKAGE_ASSET,
        help="Package graph SVG path used by README/docs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    package_asset = args.package_asset.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_asset.parent.mkdir(parents=True, exist_ok=True)

    nodes, path_to_module = discover_nodes()
    all_edges = discover_edges(path_to_module)
    production_nodes = _production_nodes(nodes)
    production_edges = [
        edge
        for edge in all_edges
        if edge.source in production_nodes and edge.target in production_nodes
    ]
    package_nodes = {
        package: ModuleNode(
            name=package,
            path="",
            package=package,
            category="entrypoint" if package in ENTRYPOINT_MODULES else "package",
            layer="production",
        )
        for package in sorted({node.package for node in production_nodes.values()})
    }
    package_edges = build_package_edges(nodes, production_edges)
    test_nodes = _test_nodes(nodes)
    test_edges = build_test_edges(nodes, all_edges)
    touched_prod_nodes = sorted({edge.target for edge in test_edges})
    test_graph_nodes = {
        **test_nodes,
        **{name: production_nodes[name] for name in touched_prod_nodes if name in production_nodes},
    }
    entrypoint_reachable = _reachable_subgraph(ENTRYPOINT_MODULES, production_edges)
    entrypoint_nodes = {name: production_nodes[name] for name in sorted(entrypoint_reachable)}
    entrypoint_edges = [
        edge
        for edge in production_edges
        if edge.source in entrypoint_nodes and edge.target in entrypoint_nodes
    ]

    canonical_json = output_dir / "dependency_graph.json"
    package_dot = output_dir / "package_graph.dot"
    package_svg = output_dir / "package_graph.svg"
    module_dot = output_dir / "production_module_graph.dot"
    module_svg = output_dir / "production_module_graph.svg"
    test_dot = output_dir / "test_dependency_graph.dot"
    test_svg = output_dir / "test_dependency_graph.svg"
    entrypoint_dot = output_dir / "entrypoint_flow_graph.dot"
    entrypoint_svg = output_dir / "entrypoint_flow_graph.svg"
    summary_path = output_dir / "summary.md"

    write_canonical_json(nodes, all_edges, package_edges, test_edges, canonical_json)
    write_dot(package_nodes, package_edges, package_dot, title="Package Dependency Graph", rankdir="LR")
    write_dot(
        production_nodes,
        production_edges,
        module_dot,
        title="Production Module Dependency Graph",
        rankdir="LR",
    )
    write_dot(
        test_graph_nodes,
        test_edges,
        test_dot,
        title="Test to Production Dependency Graph",
        rankdir="LR",
    )
    write_dot(
        entrypoint_nodes,
        entrypoint_edges,
        entrypoint_dot,
        title="Entrypoint Flow Dependency Graph",
        rankdir="TB",
    )
    render_svg(package_dot, package_svg)
    render_svg(module_dot, module_svg)
    render_svg(test_dot, test_svg)
    render_svg(entrypoint_dot, entrypoint_svg)
    package_asset.write_text(package_svg.read_text())
    write_summary(production_nodes, production_edges, package_edges, test_edges, summary_path)


if __name__ == "__main__":
    main()
