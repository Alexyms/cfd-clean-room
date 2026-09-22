#!/usr/bin/env python
"""Regenerate the machine-derived regions of docs/SYSTEM.md from the source tree.

PROVENANCE. Ported from the Agora repository
(http://alex@asustor-nas.tailacfe35.ts.net:3000/Hermes/agora.git), file
scripts/gen_system_map.py, taken at commit 8c15330fa8869805bfc2210f5410db6a63bc064d
(the file itself was last changed in 270200e). The core is carried across
unchanged: AST module discovery, the internal edge graph, the depth-first
cycle search, the content-hash fingerprint, annotation loading and
validation, marker location with strict and loose patterns, and the splice
by byte offset with newline="" on read and write. What differs from the
origin: the tool-surface and http-surface regions and the code feeding them
are dropped (no MCP tools, no HTTP routes here), the --verify-live check is
dropped with them, the requirement identifier pattern is widened to this
repository's letter-plus-number scheme, and the default paths are retargeted.
When a defect is fixed in one copy, diff against that commit to decide
whether the fix transfers.

Section 3 of SYSTEM.md describes the shape of src/: which modules exist,
which module imports which, and what is bound at runtime rather than by an
import. A hand-pasted description of a tree is a claim about the tree that
ages silently, and a stale generated table is byte-identical in kind to a
fresh one. This script makes the generated regions verifiable instead.

Run:

    python scripts/gen_system_map.py            # rewrite the regions
    python scripts/gen_system_map.py --check    # CI: does it match?

FOUR PROPERTIES THIS SCRIPT IS BUILT AROUND. Each is asserted in
tests/test_system_map.py.

1. It parses; it never imports the code under analysis. A guard that runs
   underneath the thing it guards is not a guard. Importing src.solver_ns to
   describe it would make this generator fail exactly when the solver is
   broken, which is the moment its output matters most. ast reads the file
   whatever state it is in.

2. It rewrites regions and never appends. A missing or malformed marker is a
   hard error naming the marker, not an invitation to create one. A generator
   that appends produces a document with two dsm sections, both plausible,
   one stale, and nothing in the document says which.

3. Everything outside the markers is preserved byte for byte, whitespace and
   line endings included. Section 2 of SYSTEM.md is the hand-authored
   requirements register and is the most valuable text in the file; a
   generator that reflowed it would be worse than no generator at all. The
   splice is done on the raw string by marker offsets, and the file is read
   and written with newline="" so Python never translates anything.

4. Output is deterministic, so a regeneration with no source change is a
   zero-line diff. Every table is sorted by an explicit key. The provenance
   fingerprint is a content hash rather than a timestamp for exactly this
   reason: a timestamp changes on every run, which destroys idempotence and
   turns each regeneration into a diff nobody reads.

WHAT THIS SCRIPT CANNOT KNOW, AND WHERE THAT COMES FROM INSTEAD.

Two columns in SYSTEM.md are editorial rather than measurable: a module's
one-line responsibility and the requirement identifiers a module serves.
They live in docs/system_map_annotations.toml and are checked here rather
than trusted: every module must have an entry, every entry must name a
module that exists, and every serves identifier must exist in the section 2
register. So a new module does not quietly acquire a blank cell; it fails
this script until someone writes the sentence.

The serves list is a DECLARATION and the rendered document says so. Whether
a requirement is actually met is answered by the tests named in the
register's Verified By column, not by this file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DOCUMENT = REPO_ROOT / "docs" / "SYSTEM.md"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src"
DEFAULT_ANNOTATIONS = REPO_ROOT / "docs" / "system_map_annotations.toml"

# The regions this script owns. An allow-list rather than "whatever markers are
# in the file": a marker naming a region that does not appear here is a typo or
# a half-finished edit, and answering it with silence is how a document grows a
# section nothing maintains.
REGIONS = (
    "components",
    "dsm",
    "runtime-edges",
    "source-fingerprint",
)

BEGIN_TEMPLATE = "<!-- BEGIN GENERATED: {name} -->"
END_TEMPLATE = "<!-- END GENERATED: {name} -->"

# Strict: exactly the spelling the templates produce. Loose: anything a human
# might have meant as a marker. Every loose match that is not also a strict
# match is a malformed marker, which is reported rather than skipped. A marker
# with a stray space is invisible to a strict search, so a generator that only
# looked strictly would report the region missing and, worse, a --check would
# compare against a region it never wrote.
_STRICT_MARKER_RE = re.compile(r"<!-- (BEGIN|END) GENERATED: ([a-z0-9-]+) -->")
_LOOSE_MARKER_RE = re.compile(
    r"<!--\s*(BEGIN|END)[ _\t-]*GENERATED\s*:?[^>]*-->", re.IGNORECASE
)

# Requirement identifiers, as they appear in the section 2 register and in the
# annotations file. Derived from the register itself: every row identifier is
# REQ-, one uppercase subsystem letter (S, T, C, A, V, N today), exactly two
# digits, and optionally a dot and a derived-requirement number that carries
# the parent in the identifier (REQ-S12.1 is derived from REQ-S12). A single
# letter class rather than the six seen, so a new subsystem prefix is parsed
# instead of silently dropped from the register.
REQ_RE = re.compile(r"REQ-([A-Z])(\d{2})(?:\.(\d+))?")

# Decorators that bind nothing into a dispatch registry. THIS IS A DENY-LIST AND
# THAT IS DELIBERATE, because the two directions fail differently here. An
# allow-list of known registries drops an unrecognised decorator silently, so a
# new registration mechanism would be missing from a table whose whole purpose
# is to name registration mechanisms that imports cannot see. A deny-list of
# inert decorators makes an unrecognised one show up as an extra row: noise,
# which someone reads and resolves, rather than silence, which nobody can.
INERT_DECORATORS = frozenset(
    {
        "dataclass",
        "dataclasses.dataclass",
        "property",
        "cached_property",
        "functools.cached_property",
        "staticmethod",
        "classmethod",
        "abstractmethod",
        "abc.abstractmethod",
        "overload",
        "typing.overload",
        "override",
        "typing.override",
        "functools.wraps",
    }
)


class GeneratorError(Exception):
    """A structural problem: a bad marker, a missing annotation, an unparsable file.

    Distinct from a --check disagreement, which is a legitimate answer rather
    than a fault, and which exits with a different code.
    """


# ---------------------------------------------------------------------------
# Source analysis: ast only, never import
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoratorFacts:
    dotted: str
    module_rel: str
    function: str


@dataclass
class ModuleFacts:
    rel_path: str  # posix, relative to the repository root
    dotted: str  # e.g. "src.solver_ns"
    lines: int
    source: bytes
    imports: frozenset[str]
    decorators: tuple[DecoratorFacts, ...]

    @property
    def short(self) -> str:
        return self.dotted.rsplit(".", 1)[-1]


def _dotted_name(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _decorator_dotted(node: ast.expr) -> str:
    """`dataclass(frozen=True)` -> `dataclass`; bare name unchanged."""
    target = node.func if isinstance(node, ast.Call) else node
    try:
        return ast.unparse(target)
    except Exception:  # pragma: no cover - ast.unparse handles every expr we can meet
        return "<unparsable>"


def _analyse_module(path: Path, repo_root: Path) -> ModuleFacts:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - would be a broken checkout
        raise GeneratorError(f"{path}: not valid UTF-8 ({exc}).") from exc

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise GeneratorError(
            f"{path}: cannot be parsed ({exc.msg} at line {exc.lineno}). The system map "
            "is generated from source text, so a file that does not parse stops the "
            "generation rather than being described from a stale reading of it."
        ) from exc

    dotted = _dotted_name(path, repo_root)
    package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]

    imports: set[str] = set()
    decorators: list[DecoratorFacts] = []
    rel_path = path.resolve().relative_to(repo_root.resolve()).as_posix()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports resolved against the containing package, so a
            # future `from . import mesh` is an edge in the matrix rather than
            # a hole in it.
            if node.level:
                base_parts = package.split(".")
                base = ".".join(
                    base_parts[: len(base_parts) - (node.level - 1)] or base_parts
                )
                root = f"{base}.{node.module}" if node.module else base
            else:
                root = node.module or ""
            if root:
                imports.add(root)
                for alias in node.names:
                    imports.add(f"{root}.{alias.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                dotted_dec = _decorator_dotted(decorator)
                if dotted_dec in INERT_DECORATORS:
                    continue
                decorators.append(
                    DecoratorFacts(
                        dotted=dotted_dec, module_rel=rel_path, function=node.name
                    )
                )

    return ModuleFacts(
        rel_path=rel_path,
        dotted=dotted,
        lines=len(text.splitlines()),
        source=raw,
        imports=frozenset(imports),
        decorators=tuple(decorators),
    )


def discover_modules(
    source_root: Path, repo_root: Path
) -> tuple[list[ModuleFacts], list[ModuleFacts]]:
    """Every .py under source_root, split into (described, empty).

    Scope is the package only. scripts/ and tests/ are tooling and must not
    appear in the package's own dependency matrix. A matrix that included its
    own generator would describe the toolchain rather than the system.

    Empty __init__.py files are counted and named but not given a row: a
    package marker with no code has no responsibility to record and no edges
    to draw. A NON-empty __init__.py is described like any other module,
    because at that point it is one.
    """
    paths = sorted(source_root.rglob("*.py"), key=lambda p: p.as_posix())
    if not paths:
        raise GeneratorError(
            f"{source_root}: no Python files found. Refusing to write a system map that "
            "describes an empty tree. An empty result is indistinguishable from a "
            "misconfigured source root."
        )
    modules = [_analyse_module(p, repo_root) for p in paths]
    described = [m for m in modules if m.source.strip()]
    empty = [m for m in modules if not m.source.strip()]
    return described, empty


def internal_edges(modules: list[ModuleFacts]) -> dict[str, set[str]]:
    """Module -> the described modules it imports. Keyed by dotted name."""
    known = {m.dotted for m in modules}
    edges: dict[str, set[str]] = {m.dotted: set() for m in modules}
    for module in modules:
        for imported in module.imports:
            if imported in known and imported != module.dotted:
                edges[module.dotted].add(imported)
    return edges


def find_cycles(edges: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """Every simple cycle, canonicalised and sorted, so the answer is deterministic.

    Not "cycles at depth 2". A two-module cycle is the one people look for and
    the one a matrix makes visible by eye; a three-module cycle is the one that
    actually happens, and a check that reports only pairs would answer "none"
    in its presence.
    """
    found: set[tuple[str, ...]] = set()
    stack: list[str] = []
    on_stack: set[str] = set()

    def walk(node: str) -> None:
        stack.append(node)
        on_stack.add(node)
        for neighbour in sorted(edges.get(node, ())):
            if neighbour in on_stack:
                cycle = stack[stack.index(neighbour) :]
                pivot = cycle.index(min(cycle))
                found.add(tuple(cycle[pivot:] + cycle[:pivot]))
            else:
                walk(neighbour)
        stack.pop()
        on_stack.discard(node)

    for start in sorted(edges):
        walk(start)
    return sorted(found)


# ---------------------------------------------------------------------------
# Editorial annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotations:
    modules: dict[str, dict]


def load_annotations(path: Path) -> Annotations:
    if not path.exists():
        raise GeneratorError(
            f"{path}: annotations file missing. It carries the two columns of SYSTEM.md "
            "that cannot be measured: a module's responsibility and the requirements "
            "it declares it serves."
        )
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return Annotations(modules=data.get("modules", {}))


def parse_requirement_ids(document_text: str) -> list[str]:
    """Requirement identifiers declared in the section 2 register, in document order.

    Only rows of the register count, not every mention in the file. A
    requirement is declared by having a row of its own; referring to one in
    prose does not create it, and a parser that could not tell the difference
    would report a typo as a requirement.
    """
    section = _register_section(document_text)
    ids: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue
        if REQ_RE.fullmatch(cells[0]):
            ids.append(cells[0])
    if not ids:
        raise GeneratorError(
            "SYSTEM.md: no requirement rows found in section 2. Either the register is "
            "empty or the table format changed, and the second is a documentation "
            "decision rather than something this script may paper over."
        )
    return ids


def _register_section(document_text: str) -> str:
    start = re.search(r"^##\s*2\.", document_text, re.MULTILINE)
    if start is None:
        raise GeneratorError(
            "SYSTEM.md: cannot find the '## 2.' requirements register heading."
        )
    end = re.search(r"^##\s*3\.", document_text[start.end() :], re.MULTILINE)
    return (
        document_text[start.end() : start.end() + end.start()]
        if end
        else document_text[start.end() :]
    )


def validate_annotations(
    annotations: Annotations,
    modules: list[ModuleFacts],
    requirement_ids: list[str],
) -> None:
    """Fail loudly rather than render a blank cell.

    A missing annotation is the moment someone added a module without saying
    what it is for. Rendering an empty cell there would produce a document
    that looked complete and was not, which is the failure mode the whole file
    exists to close.
    """
    problems: list[str] = []
    known_reqs = set(requirement_ids)

    described = {m.rel_path for m in modules}
    for rel_path in sorted(described - set(annotations.modules)):
        problems.append(
            f"module {rel_path} has no entry in the annotations file. Add "
            f'[modules."{rel_path}"] with a `responsibility` and a `serves` list.'
        )
    for rel_path in sorted(set(annotations.modules) - described):
        problems.append(
            f"annotations name module {rel_path}, which is not a non-empty file under the "
            "source root. Delete the entry or restore the module."
        )
    for rel_path, entry in sorted(annotations.modules.items()):
        if rel_path not in described:
            continue
        if not str(entry.get("responsibility", "")).strip():
            problems.append(f"module {rel_path} has an empty `responsibility`.")
        for req in entry.get("serves", []):
            if req not in known_reqs:
                problems.append(
                    f"module {rel_path} declares {req}, which is not a requirement in "
                    "the section 2 register. Either the identifier is a typo or the "
                    "requirement was deleted without its references."
                )

    if problems:
        raise GeneratorError(
            "annotations do not match the source tree:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    """A pipe or a bare `|` union type would end the table cell it sits in."""
    return text.replace("|", r"\|")


def _requirement_shorthand(reqs: list[str]) -> str:
    """`REQ-S01, REQ-T03` -> `S01, T03`; an empty list renders as `none`."""
    short = sorted(r.rsplit("-", 1)[-1] for r in reqs)
    return ", ".join(short) if short else "none"


def render_components(
    modules: list[ModuleFacts], empty: list[ModuleFacts], annotations: Annotations
) -> list[str]:
    lines = [
        "| Module | Lines | Responsibility | Declares it serves |",
        "|---|---|---|---|",
    ]
    for module in sorted(modules, key=lambda m: m.rel_path):
        entry = annotations.modules[module.rel_path]
        lines.append(
            f"| `{module.rel_path}` | {module.lines} | "
            f"{_md_escape(str(entry['responsibility']).strip())} | "
            f"{_requirement_shorthand(list(entry.get('serves', [])))} |"
        )
    total_files = len(modules) + len(empty)
    total_lines = sum(m.lines for m in modules) + sum(m.lines for m in empty)
    lines.append("")
    lines.append(
        f"Total {total_files} Python files, {total_lines} lines. {len(empty)} empty "
        "`__init__.py` carry no row: a package marker with no code has no responsibility "
        "to record."
    )
    lines.append("")
    lines.append(
        "`Declares it serves` is an EDITORIAL CLAIM read from "
        "`docs/system_map_annotations.toml`. It says which requirements a module is "
        "meant to satisfy, not that it does. Whether a requirement is met is answered "
        "by the tests named in the register's `Verified By` column."
    )
    return lines


def render_dsm(modules: list[ModuleFacts], edges: dict[str, set[str]]) -> list[str]:
    names = sorted(m.short for m in modules)
    by_dotted = {m.dotted: m.short for m in modules}
    short_edges = {
        by_dotted[dotted]: sorted(by_dotted[t] for t in targets)
        for dotted, targets in edges.items()
    }

    label_width = max(len(n) for n in names) + 3
    header = " " * label_width
    starts: dict[str, int] = {}
    for name in names:
        starts[name] = len(header)
        header += name + "  "
    lines = ["```", header.rstrip()]

    for row in names:
        rendered = row.ljust(label_width)
        for column in names:
            mark = "X" if column in short_edges.get(row, ()) else "."
            position = starts[column] + (len(column) - 1) // 2
            rendered = rendered.ljust(position) + mark
        lines.append(rendered.rstrip())

    total = sum(len(v) for v in short_edges.values())
    lines.append("```")
    lines.append("")
    lines.append(f"Rows import columns. Edges, {total} total:")
    lines.append("")
    lines.append("```")
    importer_width = max(len(n) for n in names) + 1
    for name in names:
        targets = short_edges.get(name, [])
        if targets:
            lines.append(f"{name.ljust(importer_width)}-> {', '.join(targets)}")
    lines.append("```")
    lines.append("")

    cycles = find_cycles({m.dotted: edges[m.dotted] for m in modules})
    if cycles:
        rendered_cycles = "; ".join(
            " -> ".join(by_dotted[n] for n in cycle) + f" -> {by_dotted[cycle[0]]}"
            for cycle in cycles
        )
        lines.append(f"Cycles of any length: **{len(cycles)}**. {rendered_cycles}.")
    else:
        lines.append(
            "Cycles of any length: **none**. Checked by depth-first search over the "
            "whole graph, not by looking for mutual pairs. A three-module cycle is the "
            "one that actually happens and a pair check answers 'none' in its presence."
        )
    lines.append("")
    lines.append(
        "This matrix is generated from the import statements in `src/` and describes "
        "the modules that exist today. Section 3.2 (cascade rules) is hand-authored "
        "and stays that way: the generator owns what the structure is, and the cascade "
        "rules are a human judgment about what follows from it."
    )
    return lines


def render_runtime_edges(modules: list[ModuleFacts]) -> list[str]:
    """Decorator-bound registrations: what the import graph structurally cannot see.

    A decorator that puts a function into a registry dispatched at call time
    is an edge no import expresses. When this table is empty, that emptiness
    is the finding: the import graph is the complete dispatch story.
    """
    grouped: dict[tuple[str, str], list[DecoratorFacts]] = {}
    for module in modules:
        for decorator in module.decorators:
            grouped.setdefault((decorator.dotted, decorator.module_rel), []).append(
                decorator
            )

    lines = ["| Registry | Bound in | Applications | Functions |", "|---|---|---|---|"]
    for (dotted, module_rel), applications in sorted(grouped.items()):
        functions = {d.function for d in applications}
        lines.append(
            f"| `{dotted}` | `{module_rel}` | {len(applications)} | {len(functions)} |"
        )
    lines.append("")
    if not grouped:
        lines.append(
            "**No runtime-bound edges.** Every decorator in `src/` is one of the inert "
            "set (`dataclass`, `staticmethod`, `property` and their kin), which bind "
            "nothing into a dispatch registry. The dependency matrix above is therefore "
            "the complete dispatch story for this system: nothing is bound at runtime "
            "that the import graph cannot see. This table exists so that the day a "
            "registry appears, it shows up here as a row rather than as silence."
        )
        lines.append("")
    lines.append(
        "**Source-derived. Nothing here is evidence about a running process.** These "
        "are decorator applications counted in the source text. `Applications` exceeds "
        "`Functions` where one function carries several non-inert decorators."
    )
    return lines


def source_fingerprint(modules: list[ModuleFacts], empty: list[ModuleFacts]) -> str:
    """SHA-256 over the analysed files, path and length included.

    A CONTENT HASH, NOT A GIT SHA AND NOT A TIMESTAMP, and both alternatives are
    traps rather than preferences. A git sha describes the last commit while
    the working tree here is routinely dirty, so the document would claim a
    currency it does not have, the exact class of defect it is meant to
    detect. A timestamp changes on every run, which destroys idempotence and
    makes every regeneration a diff.

    Path and byte length are hashed alongside the content so that renaming a
    module, or moving a line between two files, changes the digest.
    Concatenating contents alone would not.

    Line endings are normalised to LF before hashing, and the length is taken
    from the normalised bytes. Without that the digest records the checkout
    configuration of whichever machine generated it rather than the content,
    so the same tree hashes differently on Windows and on a Linux runner and
    the check fails for a reason unrelated to what it checks. A check that
    fails spuriously gets deleted, and the staleness detection goes with it.
    """
    digest = hashlib.sha256()
    for module in sorted(modules + empty, key=lambda m: m.rel_path):
        source = module.source.replace(b"\r\n", b"\n")
        digest.update(module.rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(source)).encode("ascii"))
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def render_fingerprint(
    modules: list[ModuleFacts],
    empty: list[ModuleFacts],
    source_root: Path,
    repo_root: Path,
) -> list[str]:
    rel_root = source_root.resolve().relative_to(repo_root.resolve()).as_posix()
    return [
        "| Property | Value |",
        "|---|---|",
        f"| Scope | `{rel_root}/**/*.py` |",
        f"| Files hashed | {len(modules) + len(empty)} |",
        f"| Digest | `sha256:{source_fingerprint(modules, empty)}` |",
        "",
        "This is what lets the document answer whether it is current, which is the one "
        "question a stale table cannot be asked. `python scripts/gen_system_map.py "
        "--check` recomputes the whole set of generated regions, this digest included, "
        "and exits non-zero on any disagreement.",
        "",
        "A disagreement means the source tree moved and this document did not. It does "
        "not mean any hand-authored section is wrong, and in particular it says nothing "
        "about section 2, which no generator touches.",
    ]


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------


def _dominant_newline(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def locate_regions(document_text: str) -> dict[str, tuple[int, int]]:
    """Marker name -> (offset after BEGIN marker, offset of END marker).

    Every failure below is a hard error rather than a repair. A generator that
    fixes a malformed marker, or that creates a missing region, silently takes
    ownership of a piece of the document nobody asked it to own.
    """
    strict_spans: dict[tuple[str, str], list[re.Match]] = {}
    for match in _STRICT_MARKER_RE.finditer(document_text):
        strict_spans.setdefault((match.group(1), match.group(2)), []).append(match)

    strict_offsets = {m.span() for matches in strict_spans.values() for m in matches}
    malformed = [
        match.group(0)
        for match in _LOOSE_MARKER_RE.finditer(document_text)
        if match.span() not in strict_offsets
    ]
    if malformed:
        raise GeneratorError(
            "malformed generated-region marker(s): "
            + "; ".join(repr(m) for m in malformed)
            + f". The exact spellings are {BEGIN_TEMPLATE.format(name='<name>')!r} and "
            f"{END_TEMPLATE.format(name='<name>')!r}. Refusing to guess, because a marker "
            "this script cannot see is a region it would report as missing while "
            "`--check` compared against text it never wrote."
        )

    unknown = sorted({name for (_, name) in strict_spans if name not in REGIONS})
    if unknown:
        raise GeneratorError(
            f"marker(s) name unknown region(s): {', '.join(unknown)}. Known regions are "
            f"{', '.join(REGIONS)}. Refusing to leave a marked region unmaintained."
        )

    located: dict[str, tuple[int, int]] = {}
    for name in REGIONS:
        begins = strict_spans.get(("BEGIN", name), [])
        ends = strict_spans.get(("END", name), [])
        if not begins or not ends:
            missing = []
            if not begins:
                missing.append(BEGIN_TEMPLATE.format(name=name))
            if not ends:
                missing.append(END_TEMPLATE.format(name=name))
            raise GeneratorError(
                f"region {name!r}: missing marker(s) {', '.join(repr(m) for m in missing)}. "
                "This script rewrites regions and never creates them. Appending would "
                "produce a document with two sections of the same name, both plausible, "
                "one stale."
            )
        if len(begins) > 1 or len(ends) > 1:
            raise GeneratorError(
                f"region {name!r}: marker appears {len(begins)} time(s) as BEGIN and "
                f"{len(ends)} time(s) as END. Exactly one of each is required; a "
                "duplicated region has no single answer to what is generated and what "
                "is not."
            )
        begin, end = begins[0], ends[0]
        if end.start() < begin.end():
            raise GeneratorError(
                f"region {name!r}: END marker appears before BEGIN marker."
            )
        located[name] = (begin.end(), end.start())

    ordered = sorted(located.values())
    for (_, first_end), (second_start, _) in pairwise(ordered):
        if second_start < first_end:
            raise GeneratorError(
                "generated regions overlap; markers are nested or interleaved."
            )
    return located


def splice(document_text: str, blocks: dict[str, list[str]]) -> str:
    """Rewrite each region's interior; leave every other byte exactly as it was.

    Done by offset on the raw string rather than by rebuilding the document
    line by line, because rebuilding is how a generator quietly normalises the
    trailing whitespace, the final newline, or the line endings of text it was
    told not to touch.
    """
    located = locate_regions(document_text)
    newline = _dominant_newline(document_text)
    result = document_text
    for name, (start, end) in sorted(
        located.items(), key=lambda item: item[1][0], reverse=True
    ):
        body = newline + newline.join(blocks[name]) + newline
        result = result[:start] + body + result[end:]
    return result


def build_blocks(
    document_text: str, source_root: Path, repo_root: Path, annotations_path: Path
) -> dict[str, list[str]]:
    modules, empty = discover_modules(source_root, repo_root)
    annotations = load_annotations(annotations_path)
    requirement_ids = parse_requirement_ids(document_text)
    validate_annotations(annotations, modules, requirement_ids)
    edges = internal_edges(modules)
    return {
        "components": render_components(modules, empty, annotations),
        "dsm": render_dsm(modules, edges),
        "runtime-edges": render_runtime_edges(modules),
        "source-fingerprint": render_fingerprint(
            modules, empty, source_root, repo_root
        ),
    }


def render_document(
    document_path: Path, source_root: Path, repo_root: Path, annotations_path: Path
) -> tuple[str, str]:
    """(current text, regenerated text). Both read with newline='', no translation."""
    with document_path.open("r", encoding="utf-8", newline="") as handle:
        current = handle.read()
    blocks = build_blocks(current, source_root, repo_root, annotations_path)
    return current, splice(current, blocks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


EXIT_OK = 0
EXIT_DISAGREES = 1
# 2 is reserved. The origin used it for a live check that could not run; kept
# unassigned here so the two copies' exit codes stay comparable.
EXIT_STRUCTURAL = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the machine-derived regions of docs/SYSTEM.md.",
    )
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit 1 if the document does not match the tree.",
    )
    args = parser.parse_args(argv)

    try:
        current, regenerated = render_document(
            args.document, args.source_root, args.repo_root, args.annotations
        )
    except GeneratorError as exc:
        print(f"gen_system_map: {exc}", file=sys.stderr)
        return EXIT_STRUCTURAL

    if args.check:
        if current == regenerated:
            print(f"{args.document.name} matches the source tree.")
            return EXIT_OK
        print(
            f"{args.document.name} DOES NOT match the source tree. Run "
            "`python scripts/gen_system_map.py` and commit the result.",
            file=sys.stderr,
        )
        return EXIT_DISAGREES

    if current == regenerated:
        print(f"{args.document.name} already current; nothing written.")
        return EXIT_OK

    with args.document.open("w", encoding="utf-8", newline="") as handle:
        handle.write(regenerated)
    print(f"{args.document.name} regenerated.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
