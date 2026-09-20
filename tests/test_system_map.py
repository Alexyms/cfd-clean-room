"""The system map generator, checked at the layer its failures actually occur at.

Ported with scripts/gen_system_map.py from the Agora repository
(tests/test_system_map.py at commit 8c15330fa8869805bfc2210f5410db6a63bc064d).
The classes covering the tool-surface and http-surface regions, the live
server check, the traceability checker and the import-linter contracts are
not carried across because none of those artifacts exist here. Everything
that asserts one of the generator's four stated properties is.

The interesting failures are the quiet ones: a region that regenerated into
the wrong place, a --check that agreed because it compared nothing, a
fingerprint that stayed the same because it hashed nothing. Every check that
could pass vacuously therefore carries a control: a generator that emits
nothing is idempotent, so the idempotence test sits beside one that asserts
the first run changed something; a --check that always agrees is stable, so
it is paired with a source edit that must make it disagree.

The fixture trees are temporary directories, never src/. The one exception is
the small set of read-only assertions about the real repository at the end.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_system_map  # noqa: E402 -- scripts/ is not a package; path set above

pytestmark = pytest.mark.unit


def requires_repo_checkout(*paths: Path):
    """Skip when the named repo-relative paths are absent."""
    missing = [p for p in paths if not (REPO_ROOT / p).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"Not in a repository checkout: {', '.join(str(m) for m in missing)} absent.",
    )


# ---------------------------------------------------------------------------
# Fixture trees
# ---------------------------------------------------------------------------


HAND_AUTHORED = """\
## 2. Requirements register

| ID | Requirement | Source | V |
|---|---|---|---|
| REQ-S01 | The first one. | D1 | T |
| REQ-T02 | The second one. | D2 | I |

   A line with leading spaces and trailing ones.\t

An empty line above and a tab in the middle:\tlike this.
"""


def write_tree(
    root: Path,
    *,
    modules: dict[str, str] | None = None,
    newline: str = "\n",
    extra_regions: str = "",
) -> dict[str, Path]:
    """A miniature repository: a src/ package, a document with markers, annotations.

    Deliberately not a copy of the real one. A fixture that mirrored src/
    would fail every time somebody added a module, which teaches people to
    edit the fixture rather than to read the failure.
    """
    modules = modules or {
        "src/alpha.py": "import src.beta\n\n\ndef f():\n    return 1\n",
        "src/beta.py": "VALUE = 2\n",
        "src/__init__.py": "",
    }
    for rel, body in modules.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="")

    document = root / "SYSTEM.md"
    regions = "\n\n".join(
        f"<!-- BEGIN GENERATED: {name} -->\n<!-- END GENERATED: {name} -->"
        for name in gen_system_map.REGIONS
    )
    text = (
        "# Fixture\n\n"
        + HAND_AUTHORED
        + "\n## 3. Generated\n\n"
        + regions
        + extra_regions
        + "\n\n## 4. Tail\n"
    )
    with document.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text.replace("\n", newline) if newline != "\n" else text)

    annotations = root / "annotations.toml"
    entries = []
    for rel, body in sorted(modules.items()):
        if not body.strip():
            continue
        entries.append(
            f'[modules."{rel}"]\nresponsibility = "Does {Path(rel).stem}."\nserves = []\n'
        )
    annotations.write_text("\n".join(entries), encoding="utf-8")
    return {"document": document, "annotations": annotations, "root": root}


def run_generator(tree: dict[str, Path], *flags: str) -> int:
    return gen_system_map.main(
        [
            "--document",
            str(tree["document"]),
            "--source-root",
            str(tree["root"] / "src"),
            "--repo-root",
            str(tree["root"]),
            "--annotations",
            str(tree["annotations"]),
            *flags,
        ]
    )


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def document_text(tree: dict[str, Path]) -> str:
    with tree["document"].open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


class TestIdempotence:
    """The correctness property, and the one that catches almost everything else.

    A generator that inserts its own markers, mis-splices an offset, appends a
    newline, or renders a set in iteration order fails this and usually fails
    nothing else.
    """

    def test_a_second_run_against_an_unchanged_tree_is_a_zero_byte_diff(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        first = read_bytes(tree["document"])
        assert run_generator(tree) == 0
        assert read_bytes(tree["document"]) == first

    def test_the_first_run_actually_changed_something(self, tmp_path):
        """The control. Idempotence is trivially true of a generator that writes nothing."""
        tree = write_tree(tmp_path)
        before = read_bytes(tree["document"])
        assert run_generator(tree) == 0
        assert read_bytes(tree["document"]) != before
        assert b"| Module | Lines |" in read_bytes(tree["document"])

    def test_ordering_is_stable_when_the_files_arrive_in_a_different_order(
        self, tmp_path
    ):
        """Two trees, identical content, written in opposite order.

        Directory iteration order is not a promise on any filesystem, and a
        generator that leaned on it would produce a document that reordered
        itself on somebody else's machine.
        """
        modules = {
            "src/__init__.py": "",
            "src/zulu.py": "import src.alpha\n",
            "src/alpha.py": "X = 1\n",
            "src/mike.py": "import src.zulu\n",
        }
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        first = write_tree(first_root, modules=dict(modules))
        second = write_tree(second_root, modules=dict(reversed(list(modules.items()))))
        assert run_generator(first) == 0
        assert run_generator(second) == 0
        assert read_bytes(first["document"]) == read_bytes(second["document"])


class TestTheGeneratorParsesAndNeverImports:
    """A guard that runs underneath the thing it guards is not a guard.

    Importing src.solver_ns to describe it would make the generator fail
    exactly when the solver is broken, the moment its output matters most.
    """

    def test_a_module_that_explodes_on_import_is_still_described(self, tmp_path):
        """The behavioural control, not a grep for the word `import`.

        This module cannot be imported without raising. It parses perfectly
        well, so a parsing generator describes it and an importing one dies.
        """
        tree = write_tree(
            tmp_path,
            modules={
                "src/__init__.py": "",
                "src/landmine.py": (
                    "raise RuntimeError('this module cannot be imported')\n"
                    "\n"
                    "\ndef unreachable():\n    return 1\n"
                ),
            },
        )
        assert run_generator(tree) == 0
        assert "`src/landmine.py`" in document_text(tree)
        assert "src.landmine" not in sys.modules

    def test_a_module_that_does_not_parse_stops_the_run_and_names_the_file(
        self, tmp_path, capsys
    ):
        """The other half. Parsing is not a way of tolerating anything.

        A file that will not parse cannot be described, and describing it
        from the previous reading would be a document claiming currency it
        does not have.
        """
        tree = write_tree(
            tmp_path,
            modules={
                "src/__init__.py": "",
                "src/broken.py": "def f(:\n    pass\n",
            },
        )
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "broken.py" in capsys.readouterr().err


class TestMarkersAreStructureAndNotSuggestions:
    """Every one of these is a hard error naming the marker, never a repair.

    A generator that creates a missing region produces a document with two
    sections of the same name, both plausible, one stale, and nothing in the
    document says which is which.
    """

    def test_a_missing_begin_marker_fails_and_names_it(self, tmp_path, capsys):
        tree = write_tree(tmp_path)
        text = document_text(tree).replace("<!-- BEGIN GENERATED: dsm -->\n", "")
        tree["document"].write_text(text, encoding="utf-8", newline="")
        before = read_bytes(tree["document"])

        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "dsm" in capsys.readouterr().err
        assert read_bytes(tree["document"]) == before, "a failed run must write nothing"

    def test_a_missing_end_marker_fails_and_names_it(self, tmp_path, capsys):
        tree = write_tree(tmp_path)
        text = document_text(tree).replace("<!-- END GENERATED: dsm -->\n", "")
        tree["document"].write_text(text, encoding="utf-8", newline="")

        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "END GENERATED: dsm" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "broken",
        [
            "<!--BEGIN GENERATED: dsm -->",
            "<!-- BEGIN GENERATED:dsm -->",
            "<!-- begin generated: dsm -->",
            "<!-- BEGIN  GENERATED: dsm -->",
        ],
    )
    def test_a_malformed_marker_fails_rather_than_reading_as_absent(
        self, tmp_path, capsys, broken
    ):
        """The subtle one, and the reason the loose pattern exists.

        A marker with a stray space is invisible to a strict search. Without
        the loose pass this would be reported as a MISSING region, which is at
        least loud, but under --check it would compare the document against a
        rendering of a region that was never spliced, so the answer would be
        wrong rather than merely unhelpful.
        """
        tree = write_tree(tmp_path)
        text = document_text(tree).replace("<!-- BEGIN GENERATED: dsm -->", broken)
        tree["document"].write_text(text, encoding="utf-8", newline="")

        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "malformed" in capsys.readouterr().err.lower()

    def test_a_duplicated_region_fails(self, tmp_path, capsys):
        tree = write_tree(
            tmp_path,
            extra_regions="\n\n<!-- BEGIN GENERATED: dsm -->\n<!-- END GENERATED: dsm -->",
        )
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "dsm" in capsys.readouterr().err

    def test_a_marker_naming_an_unknown_region_fails(self, tmp_path, capsys):
        tree = write_tree(
            tmp_path,
            extra_regions=(
                "\n\n<!-- BEGIN GENERATED: invented -->\n<!-- END GENERATED: invented -->"
            ),
        )
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "invented" in capsys.readouterr().err

    def test_a_marker_for_a_dropped_origin_region_fails(self, tmp_path, capsys):
        """The port removed tool-surface. A leftover marker for it must not be silent."""
        tree = write_tree(
            tmp_path,
            extra_regions=(
                "\n\n<!-- BEGIN GENERATED: tool-surface -->"
                "\n<!-- END GENERATED: tool-surface -->"
            ),
        )
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "tool-surface" in capsys.readouterr().err

    def test_nothing_is_ever_appended_to_the_end_of_the_document(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        assert document_text(tree).endswith("## 4. Tail\n")


class TestEverythingOutsideTheMarkersSurvivesByteForByte:
    """Section 2 is the hand-authored register and the most valuable text in the file.

    A generator that reflowed it, stripped its trailing spaces, or normalised
    its line endings would be doing damage that no diff reviewer would think
    to look for, in the one section a generator has no business touching.
    """

    def test_the_hand_authored_block_is_unchanged_including_whitespace(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        assert HAND_AUTHORED in document_text(tree)

    def test_a_crlf_document_stays_crlf(self, tmp_path):
        """A generator that wrote LF into a CRLF document would produce mixed
        line endings, which git normalises on the next touch and which then
        shows up as a whole-file diff attributed to whoever saved it next."""
        tree = write_tree(tmp_path, newline="\r\n")
        assert run_generator(tree) == 0
        raw = read_bytes(tree["document"])
        assert b"\r\n" in raw
        assert re.search(rb"[^\r]\n", raw) is None, "a bare LF was introduced"


class TestCheckModeCanDisagreeAndCanAgree:
    """Both directions, because either alone is satisfied by a broken implementation.

    A --check that always agrees passes the agreement test. A --check that
    always disagrees passes the disagreement test. Only the pair says anything.
    """

    def test_it_agrees_immediately_after_a_generation(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        assert run_generator(tree, "--check") == gen_system_map.EXIT_OK

    def test_it_disagrees_once_a_source_file_changes(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        (tree["root"] / "src" / "beta.py").write_text(
            "VALUE = 3\nEXTRA = 4\n", encoding="utf-8"
        )
        assert run_generator(tree, "--check") == gen_system_map.EXIT_DISAGREES

    def test_it_disagrees_when_a_source_file_changes_without_changing_its_length(
        self, tmp_path
    ):
        """Same byte count, different bytes. A length-only fingerprint would agree."""
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        (tree["root"] / "src" / "beta.py").write_text("VALUE = 9\n", encoding="utf-8")
        assert run_generator(tree, "--check") == gen_system_map.EXIT_DISAGREES

    def test_it_still_agrees_when_a_file_outside_the_source_root_changes(
        self, tmp_path
    ):
        """Scope is src/. Tooling and notes are not the system being described."""
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        (tree["root"] / "notes.py").write_text("# outside src/\n", encoding="utf-8")
        (tree["root"] / "scripts").mkdir()
        (tree["root"] / "scripts" / "tool.py").write_text(
            "# tooling\n", encoding="utf-8"
        )
        assert run_generator(tree, "--check") == gen_system_map.EXIT_OK


class TestTheFingerprintIsAContentHash:
    def _digest(self, tree) -> str:
        match = re.search(r"sha256:([0-9a-f]{64})", document_text(tree))
        assert match, "no digest found in the source-fingerprint region"
        return match.group(1)

    def test_two_runs_produce_the_same_digest(self, tmp_path):
        """A timestamp would fail this, which is exactly why it is not a timestamp."""
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        first = self._digest(tree)
        assert run_generator(tree) == 0
        assert self._digest(tree) == first

    def test_editing_a_source_file_changes_the_digest(self, tmp_path):
        tree = write_tree(tmp_path)
        assert run_generator(tree) == 0
        before = self._digest(tree)
        (tree["root"] / "src" / "beta.py").write_text("VALUE = 3\n", encoding="utf-8")
        assert run_generator(tree) == 0
        assert self._digest(tree) != before

    def test_moving_a_line_between_two_files_changes_the_digest(self, tmp_path):
        """The control for hashing paths and lengths alongside content.

        Concatenating file contents and hashing the result would answer
        identically here: the same bytes, in the same order, in a differently
        shaped tree.
        """
        tree = write_tree(
            tmp_path,
            modules={
                "src/__init__.py": "",
                "src/one.py": "A = 1\nB = 2\n",
                "src/two.py": "C = 3\n",
            },
        )
        assert run_generator(tree) == 0
        before = self._digest(tree)

        (tree["root"] / "src" / "one.py").write_text("A = 1\n", encoding="utf-8")
        (tree["root"] / "src" / "two.py").write_text("B = 2\nC = 3\n", encoding="utf-8")
        assert run_generator(tree) == 0
        assert self._digest(tree) != before


class TestTheAnnotationsAreCheckedRatherThanTrusted:
    def test_a_module_with_no_entry_stops_the_generation(self, tmp_path, capsys):
        tree = write_tree(tmp_path)
        (tree["root"] / "src" / "gamma.py").write_text("G = 1\n", encoding="utf-8")
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "gamma.py" in capsys.readouterr().err

    def test_an_entry_naming_no_module_stops_the_generation(self, tmp_path, capsys):
        tree = write_tree(tmp_path)
        with tree["annotations"].open("a", encoding="utf-8") as handle:
            handle.write(
                '\n[modules."src/ghost.py"]\nresponsibility = "Haunts."\nserves = []\n'
            )
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "ghost.py" in capsys.readouterr().err

    def test_a_serves_entry_naming_no_requirement_stops_the_generation(
        self, tmp_path, capsys
    ):
        tree = write_tree(tmp_path)
        text = tree["annotations"].read_text(encoding="utf-8")
        text = text.replace("serves = []", 'serves = ["REQ-S99"]', 1)
        tree["annotations"].write_text(text, encoding="utf-8")
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        assert "REQ-S99" in capsys.readouterr().err

    def test_a_serves_entry_naming_a_registered_requirement_is_accepted(self, tmp_path):
        """The other direction: the validator must accept what the register declares."""
        tree = write_tree(tmp_path)
        text = tree["annotations"].read_text(encoding="utf-8")
        text = text.replace("serves = []", 'serves = ["REQ-T02", "REQ-S01"]', 1)
        tree["annotations"].write_text(text, encoding="utf-8")
        assert run_generator(tree) == gen_system_map.EXIT_OK
        assert "| S01, T02 |" in document_text(tree)

    @pytest.mark.parametrize("empty", ['""', '"   "', '"\\t"'])
    def test_an_empty_responsibility_stops_the_generation(
        self, tmp_path, capsys, empty
    ):
        """Whitespace-only is parametrised alongside empty because .strip() is
        what the validator applies, and a rule that only caught "" would pass a tab."""
        tree = write_tree(tmp_path)
        text = tree["annotations"].read_text(encoding="utf-8")
        text = re.sub(
            r'responsibility = "[^"]*"', f"responsibility = {empty}", text, count=1
        )
        tree["annotations"].write_text(text, encoding="utf-8")
        assert run_generator(tree) == gen_system_map.EXIT_STRUCTURAL
        error = capsys.readouterr().err
        assert "responsibility" in error
        assert "src/alpha.py" in error


class TestTheRequirementPatternMatchesThisRegister:
    """The identifier scheme was widened from the origin's REQ-AGORA-nnn.

    Derived from the section 2 register rather than from the four examples in
    the porting brief, so the pattern is asserted against what the register
    actually contains, in both directions.
    """

    @pytest.mark.parametrize(
        "identifier", ["REQ-S01", "REQ-T10", "REQ-C04", "REQ-A06", "REQ-V03", "REQ-N03"]
    )
    def test_every_prefix_in_the_register_is_accepted(self, identifier):
        assert gen_system_map.REQ_RE.fullmatch(identifier)

    @pytest.mark.parametrize(
        "identifier",
        ["REQ-AGORA-001", "REQ-S1", "REQ-S001", "REQ-s01", "REQ-01", "VAL-001"],
    )
    def test_shapes_outside_the_scheme_are_rejected(self, identifier):
        assert gen_system_map.REQ_RE.fullmatch(identifier) is None

    def test_a_requirement_mentioned_in_prose_is_not_a_row(self, tmp_path):
        tree = write_tree(tmp_path)
        text = document_text(tree).replace(
            "## 3. Generated",
            "REQ-N09 is discussed here in prose only.\n\n## 3. Generated",
        )
        tree["document"].write_text(text, encoding="utf-8", newline="")
        assert gen_system_map.parse_requirement_ids(text) == ["REQ-S01", "REQ-T02"]


class TestTheRuntimeEdgesTableSaysWhenItIsEmpty:
    """An empty table is a finding, and the document has to say so in words."""

    def test_an_inert_only_tree_renders_the_no_edges_statement(self, tmp_path):
        tree = write_tree(
            tmp_path,
            modules={
                "src/__init__.py": "",
                "src/plain.py": (
                    "from dataclasses import dataclass\n\n\n@dataclass\nclass P:\n"
                    "    x: int\n\n    @property\n    def y(self) -> int:\n"
                    "        return self.x\n"
                ),
            },
        )
        assert run_generator(tree) == 0
        text = document_text(tree)
        start, end = gen_system_map.locate_regions(text)["runtime-edges"]
        region = text[start:end]
        assert "No runtime-bound edges" in region
        assert "complete dispatch story" in region
        assert "Source-derived" in region

    def test_a_non_inert_decorator_becomes_a_row_instead(self, tmp_path):
        """The control: the statement must disappear the moment a registry appears."""
        tree = write_tree(
            tmp_path,
            modules={
                "src/__init__.py": "",
                "src/registry.py": (
                    "import functools\n\n\ndef register(fn):\n    return fn\n\n\n"
                    "@register\ndef bound():\n    return 1\n"
                ),
            },
        )
        assert run_generator(tree) == 0
        text = document_text(tree)
        start, end = gen_system_map.locate_regions(text)["runtime-edges"]
        region = text[start:end]
        assert "| `register` | `src/registry.py` | 1 | 1 |" in region
        assert "No runtime-bound edges" not in region


# ---------------------------------------------------------------------------
# The real document
# ---------------------------------------------------------------------------


@requires_repo_checkout(
    Path("docs/SYSTEM.md"), Path("docs/system_map_annotations.toml")
)
class TestTheRealDocumentMatchesTheRealTree:
    def test_check_mode_agrees(self):
        """The whole point, asserted on every commit rather than at phase close.

        If this fails, docs/SYSTEM.md describes a tree that no longer exists.
        Run `python scripts/gen_system_map.py` and read the diff before
        committing it.
        """
        assert gen_system_map.main(["--check"]) == gen_system_map.EXIT_OK

    def test_the_register_parses_to_the_expected_identifiers(self):
        """A per-prefix range rather than a count, so a gap in the numbering fails."""
        with (REPO_ROOT / "docs" / "SYSTEM.md").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            ids = gen_system_map.parse_requirement_ids(handle.read())
        expected = (
            [f"REQ-S{n:02d}" for n in range(1, 13)]
            + [f"REQ-T{n:02d}" for n in range(1, 11)]
            + [f"REQ-C{n:02d}" for n in range(1, 5)]
            + [f"REQ-A{n:02d}" for n in range(1, 7)]
            + [f"REQ-V{n:02d}" for n in range(1, 4)]
            + [f"REQ-N{n:02d}" for n in range(1, 4)]
        )
        assert ids == expected

    def test_the_runtime_edges_table_labels_itself_source_derived(self):
        """A count of decorators is a statement about source text. The table
        has to say so in the document, because the reader who acts on it never
        sees this script."""
        with (REPO_ROOT / "docs" / "SYSTEM.md").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            text = handle.read()
        start, end = gen_system_map.locate_regions(text)["runtime-edges"]
        assert "Source-derived" in text[start:end]
