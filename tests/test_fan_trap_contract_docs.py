"""Keep the documented fan-trap contract in step with the code.

The verdict's shape and severity rules changed several times while this was
built, and each change left one description behind -- the tool docstring, then
the skill file, each caught only in review. The skill in particular is served
to agents over ``skill://``, so a stale copy is not a docs nit: it is what the
model reads before deciding how to call the tool.

These tests assert the properties that actually went wrong, not prose.
"""

import re
import unittest
from pathlib import Path

from src.obqc_validator import FAN_TRAP_KINDS, KIND_CONDITIONAL_ROW_COUNT, OBQCResult

ROOT = Path(__file__).resolve().parent.parent

# Every file that describes the verdict to a human or to a model.
CONTRACT_DOCS = (
    ROOT / "docs" / "obqc.md",
    ROOT / "docs" / "tools-reference.md",
    ROOT / "docs" / "fan-trap-prevention.md",
    ROOT / ".claude" / "skills" / "fan-trap-prevention.md",
    ROOT / "src" / "main.py",
)

VERDICT_KEYS = ("evaluated", "detected", "blocking", "findings")


class TestFanTrapContractIsDocumentedAccurately(unittest.TestCase):
    def test_every_written_shape_lists_the_whole_verdict(self):
        """A partial key list is how each stale description read.

        Checked against the braces actually written, not against the file as a
        whole: the stale skill listed ``{detected, blocking, findings}`` and
        then mentioned ``evaluated`` in the next breath, so a file-wide search
        for the word found it and proved nothing.
        """
        # The prose form -- "{evaluated, detected, blocking, findings}" -- is
        # what went stale each time. A worked JSON example is skipped: its
        # nested braces defeat a flat match, and it is not where the key list
        # gets summarised.
        shape = re.compile(r"\{([a-z_, ]*\b(?:detected|findings)\b[a-z_, ]*)\}")

        for path in CONTRACT_DOCS:
            text = path.read_text()
            if "obqc_fan_trap" not in text:
                continue

            # Wrapped across lines in the tool docstring, so compare on one.
            written = [
                group
                for group in shape.findall(re.sub(r"\s+", " ", text))
                if sum(key in group for key in VERDICT_KEYS) >= 2
            ]
            with self.subTest(doc=path.relative_to(ROOT)):
                self.assertTrue(
                    written,
                    f"{path.relative_to(ROOT)} describes obqc_fan_trap but "
                    "never writes out its shape",
                )
                for group in written:
                    missing = [key for key in VERDICT_KEYS if key not in group]
                    self.assertEqual(
                        missing,
                        [],
                        f"{path.relative_to(ROOT)} writes the verdict as "
                        f"'{{{group}}}', missing {missing}",
                    )

    def test_no_description_claims_every_fan_trap_blocks(self):
        """A conditional_row_count is reported without blocking.

        Saying otherwise teaches a caller to treat a successful response as
        proof that no fan-trap was found.
        """
        overbroad = re.compile(
            r"fan-trap is a \*\*blocking error\*\*|every (detected )?fan-trap blocks",
            re.IGNORECASE,
        )
        for path in CONTRACT_DOCS:
            with self.subTest(doc=path.relative_to(ROOT)):
                self.assertIsNone(
                    overbroad.search(path.read_text()),
                    f"{path.relative_to(ROOT)} says all fan-traps block, but "
                    "conditional_row_count findings do not",
                )

    def test_documented_keys_match_the_result_object(self):
        """The list above is only useful while it mirrors the code."""
        self.assertEqual(
            set(OBQCResult(is_valid=True).to_dict()["obqc_fan_trap"]),
            set(VERDICT_KEYS),
        )

    def test_every_finding_kind_is_documented(self):
        """A kind the code can emit but no page explains is undocumented.

        Adding the fourth kind left the reference describing three.
        """
        reference = (ROOT / "docs" / "obqc.md").read_text()
        for kind in FAN_TRAP_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, reference)

    def test_stated_counts_match_the_number_of_kinds(self):
        """Prose that counts the rules has to count them correctly.

        "Three findings, strongest first" outlived the third revision of this
        list, and a reader has no way to tell it is stale.
        """
        words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        expected = words[len(FAN_TRAP_KINDS)]
        # Plural only: "One finding below is reported without blocking" counts
        # a single rule, not the set.
        counted = re.compile(
            r"\b(two|three|four|five|six)\s+(findings|rules)\b", re.IGNORECASE
        )

        for path in CONTRACT_DOCS:
            with self.subTest(doc=path.relative_to(ROOT)):
                for word, noun in counted.findall(path.read_text()):
                    self.assertEqual(
                        word.lower(),
                        expected,
                        f"{path.relative_to(ROOT)} says '{word} {noun}' but "
                        f"there are {len(FAN_TRAP_KINDS)} fan-trap findings",
                    )

    def test_the_fan_trap_section_is_not_labelled_error_only(self):
        """One kind never blocks, so a bare (ERROR) heading misdescribes it."""
        heading = re.compile(r"^#+ Fan-trap detection \((.+)\)\s*$", re.MULTILINE)

        for path in CONTRACT_DOCS:
            for severities in heading.findall(path.read_text()):
                with self.subTest(doc=path.relative_to(ROOT)):
                    self.assertIn(
                        "WARNING",
                        severities,
                        f"{path.relative_to(ROOT)} labels fan-trap detection "
                        f"'({severities})', but {KIND_CONDITIONAL_ROW_COUNT} "
                        "is warning-only",
                    )


if __name__ == "__main__":
    unittest.main()
