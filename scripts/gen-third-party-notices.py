#!/usr/bin/env python3
"""Generate ``THIRD_PARTY_NOTICES.md`` from the resolved production tree.

OrionBelt Analytics itself is BUSL-1.1. The PyPI wheel only *declares* its
dependencies -- pip fetches them from PyPI, so nothing third-party is
redistributed there and no attribution is owed. The Docker image is different:
it bakes in the whole ``uv sync --no-dev`` closure plus a Debian base, which is
redistribution of binaries, and MIT/BSD/Apache-2.0 all require the copyright
notice and licence text to travel with them.

This script produces the notice file that satisfies that, and the CI ``--check``
mode keeps it honest when Dependabot moves the tree underneath us.

Two deliberate design choices, both about keeping the committed file stable:

1. **No versions, and PyPI URLs rather than declared home pages.** A notice
   owes attribution, not a version number, and a declared ``Home-page`` drifts
   between releases. Deriving every row from the package *name* alone means a
   routine version bump produces no diff at all, so the ``--check`` gate stays
   quiet until the thing worth a human look actually happens: a package
   entering or leaving the tree, or changing licence. Red CI on every bump
   would train everyone to regenerate without reading.

2. **No verbatim licence texts in the Markdown.** Inlined and deduplicated they
   run to ~810 KB, they trip ``check-added-large-files``, and -- worse -- they
   are not reproducible: numpy and friends ship different vendored-library
   texts per platform wheel, so a macOS dev and Linux CI would generate
   different files forever. The texts belong with the binaries instead. They
   already ship inside the image under each ``*.dist-info/``; ``--dump-texts``
   collects them into one file at Docker build time so that stays true by
   construction rather than by accident.

Usage::

    python scripts/gen-third-party-notices.py            # write the file
    python scripts/gen-third-party-notices.py --check    # fail if stale (CI)
    python scripts/gen-third-party-notices.py --dump-texts OUT  # image bundle
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTICES_PATH = REPO_ROOT / "THIRD_PARTY_NOTICES.md"

# Packages that are never installed on every platform, so their metadata cannot
# be read locally on at least one of Linux/macOS. Listed here so the output does
# not depend on which OS ran the generator -- these entries win over anything
# found on disk, which is what makes a Linux CI run and a macOS dev run agree.
PLATFORM_ONLY: dict[str, str] = {
    "colorama": "BSD-3-Clause",
    "jeepney": "MIT",
    "pywin32": "PSF-2.0",
    "pywin32-ctypes": "BSD-3-Clause",
    "secretstorage": "BSD-3-Clause",
    "tzdata": "Apache-2.0",
}

# Packages whose declared metadata is absent, ambiguous ("BSD", "Dual
# License"), or a copyright line rather than a licence. Each value below was
# read out of the package's own shipped licence file -- e.g. BSD 2- vs 3-clause
# was decided by whether the text carries the "neither the name" clause.
AMBIGUOUS: dict[str, str] = {
    "caio": "Apache-2.0",
    "docutils": "LicenseRef-Docutils-Mixed",
    "google-crc32c": "Apache-2.0",
    "html5rdf": "MIT",
    "idna": "BSD-3-Clause",
    "locate": "MIT",
    "lz4": "BSD-3-Clause",
    "mpmath": "BSD-3-Clause",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "psycopg2-binary": "LGPL-3.0-or-later WITH psycopg-exception",
    "pyasn1-modules": "BSD-2-Clause",
    "pybreaker": "BSD-3-Clause",
    "pyperclip": "BSD-3-Clause",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "sympy": "BSD-3-Clause",
    "zstd": "BSD-2-Clause",
}

# Free-text licence strings seen in the wild, mapped to SPDX. Anything not
# listed here and not overridden above is a hard error rather than a guess: an
# unrecognised string is exactly the case that deserves a human read.
# Deliberately absent: bare "BSD" and "BSD License", which do not say whether
# the 3rd clause is present -- those must go through AMBIGUOUS.
RAW_TO_SPDX: dict[str, str] = {
    "# mit license": "MIT",
    "# released under mit license": "MIT",
    "3-clause bsd license": "BSD-3-Clause",
    "apache 2.0": "Apache-2.0",
    "apache license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license version 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache-2.0 and cnri-python": "Apache-2.0 AND CNRI-Python",
    "apache-2.0 and mit": "Apache-2.0 AND MIT",
    "apache-2.0 or bsd-3-clause": "Apache-2.0 OR BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-3-clause and 0bsd and mit and zlib and cc0-1.0": (
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"
    ),
    "isc": "ISC",
    "isc license": "ISC",
    "mit": "MIT",
    "mit and python-2.0": "MIT AND Python-2.0",
    "mit license": "MIT",
    "mit or apache-2.0": "MIT OR Apache-2.0",
    "mit-cmu": "MIT-CMU",
    "mpl-2.0": "MPL-2.0",
    "mpl-2.0 and (apache-2.0 or mit)": "MPL-2.0 AND (Apache-2.0 OR MIT)",
    "mpl-2.0 and mit": "MPL-2.0 AND MIT",
    "psf-2.0": "PSF-2.0",
    "the mit license (mit)": "MIT",
    "unlicense": "Unlicense",
    "w3c-20150513": "W3C-20150513",
}

# Licence families that carry an obligation beyond "keep the notice". A package
# whose SPDX expression matches must be listed in REVIEWED_COPYLEFT, otherwise
# --check fails: that is the gate that stops a copyleft dependency arriving
# unnoticed on the back of a routine bump.
COPYLEFT_RE = re.compile(
    r"\b(GPL|LGPL|AGPL|MPL|CC-BY-SA|EUPL|OSL|CDDL|EPL|SSPL|Docutils-Mixed)",
    re.IGNORECASE,
)

REVIEWED_COPYLEFT: frozenset[str] = frozenset(
    {"certifi", "docutils", "orjson", "psycopg2-binary", "tqdm"}
)

# Hand-written notices, rendered only when the package is actually in the tree.
# Each is (heading, packages it covers, body).
SPECIAL_NOTICES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "psycopg2-binary — LGPL-3.0-or-later, with exceptions",
        ("psycopg2-binary",),
        """\
`psycopg2` is the only strongly copyleft component in the tree. OrionBelt
Analytics imports it unmodified as a separate Python module and does not link
it statically or derive from its source, so LGPL section 5 ("works that use the
library") applies and the LGPL does **not** reach OrionBelt Analytics' own
BUSL-1.1 licensed code.

Distributing the Docker image does convey `psycopg2` itself, which obliges us
to:

- ship its licence text (present in the image under
  `psycopg2_binary-*.dist-info/licenses/LICENSE`, and in the collected bundle
  described below);
- state where its complete corresponding source can be obtained;
- not prevent a recipient from replacing it with a modified version.

Complete source: <https://github.com/psycopg/psycopg2> and
<https://pypi.org/project/psycopg2-binary/#files>. `psycopg2` inside the image
is unmodified upstream, installed from the published wheel, and a recipient may
replace it in place under `/opt/venv`.

Note that psycopg 3 is LGPL-3.0 as well, so switching drivers would not remove
this obligation.""",
    ),
    (
        "wordfreq — Apache-2.0 code, CC-BY-SA 4.0 data",
        ("wordfreq",),
        """\
`wordfreq` (used for cryptic-name detection during schema analysis) is Apache
2.0, but the frequency **data files** bundled in its wheel are redistributable
under [Creative Commons Attribution-ShareAlike
4.0](https://creativecommons.org/licenses/by-sa/4.0/), and carry attribution
requirements of their own. Share-alike binds that data and works derived from
it; it does not reach OrionBelt Analytics' own code, which neither
redistributes the corpora nor derives new frequency lists from them.

Required credits, per the upstream terms:

- Data extracted from **Google Books Ngrams**
  (<http://books.google.com/ngrams>) and **Google Books Syntactic Ngrams**
  (<http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html>).
- Data from **SUBTLEX**, whose terms require crediting the SUBTLEX authors and
  keeping it clear that SUBTLEX is freely available data.
- `wordfreq` itself: <https://github.com/rspeer/wordfreq>.""",
    ),
    (
        "MPL-2.0 components — certifi, orjson, tqdm",
        ("certifi", "orjson", "tqdm"),
        """\
MPL-2.0 is file-level copyleft: it attaches to the licensed files themselves,
not to a larger work that merely uses them. All three are shipped unmodified,
so the obligation is limited to keeping the licence text with the binary and
making the source of those files available:

- `certifi` — <https://github.com/certifi/python-certifi>
- `orjson` — <https://github.com/ijl/orjson> (MPL-2.0 parts; the remainder is
  Apache-2.0 OR MIT)
- `tqdm` — <https://github.com/tqdm/tqdm> (MPL-2.0 parts; the remainder is
  MIT)

Should any of these ever be patched rather than vendored as-is, the modified
files themselves must be published under MPL-2.0.""",
    ),
    (
        "docutils — mixed public domain / BSD / GPL",
        ("docutils",),
        """\
`docutils` arrives transitively (fastmcp → cyclopts → rich-rst → docutils) and
is licensed per-file rather than as a whole: the bulk is public domain, with
some files under a 2-clause BSD licence and a small number under the GPL. It is
not GPL as a work, and the GPL-licensed files are auxiliary tooling rather than
anything on OrionBelt Analytics' import path — the dependency is pulled in for
reStructuredText rendering in the FastMCP CLI's help output.

Per-file terms are in `docutils/COPYING.txt` inside the distribution;
upstream is <https://docutils.sourceforge.io/>.""",
    ),
]

BUNDLE_PATH_IN_IMAGE = "/app/licenses/THIRD_PARTY_LICENSES.txt"

LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE)", re.IGNORECASE)


def normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503.

    Args:
        name: A distribution name as written in metadata or a requirement.

    Returns:
        The lowercased name with runs of ``-``, ``_`` and ``.`` collapsed to
        a single hyphen.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def production_packages() -> dict[str, str]:
    """Resolve the production dependency closure from the lockfile.

    ``uv export`` is universal: it emits the union across every supported
    platform, with environment markers, so the package *set* does not depend on
    the machine running this script.

    Returns:
        Mapping of normalized name to the display name as locked.

    Raises:
        SystemExit: If ``uv export`` fails (most often a stale ``uv.lock``).
    """
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--format",
            "requirements-txt",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        sys.exit(f"uv export failed:\n{result.stderr}")

    packages: dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e", "-", ".")):
            continue
        name = re.split(r"[=<>!~; \[]", line)[0]
        if name:
            packages[normalize(name)] = name
    return packages


def site_packages_dirs() -> list[Path]:
    """Return the site-packages directories to scan for installed metadata.

    Returns:
        Existing ``purelib``/``platlib`` paths for the running interpreter,
        deduplicated and in a stable order.
    """
    paths = sysconfig.get_paths()
    seen: list[Path] = []
    for key in ("purelib", "platlib"):
        candidate = Path(paths[key])
        if candidate.is_dir() and candidate not in seen:
            seen.append(candidate)
    return seen


def _read_metadata(dist_info: Path) -> tuple[str, str] | None:
    """Extract the distribution name and its raw licence string.

    Args:
        dist_info: A ``*.dist-info`` directory.

    Returns:
        ``(display_name, raw_license)`` where ``raw_license`` may be empty, or
        ``None`` if the directory holds no readable ``METADATA``.
    """
    metadata = dist_info / "METADATA"
    if not metadata.is_file():
        return None

    name = ""
    expression = ""
    legacy = ""
    classifiers: list[str] = []
    with metadata.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                break  # end of headers; the long description follows
            if line.startswith("Name: "):
                name = line[len("Name: ") :].strip()
            elif line.startswith("License-Expression: "):
                expression = line[len("License-Expression: ") :].strip()
            elif line.startswith("License: ") and not legacy:
                value = line[len("License: ") :].strip()
                # Some projects paste their whole licence into this field.
                if value and len(value) < 80:
                    legacy = value
            elif line.startswith("Classifier: License ::"):
                classifiers.append(line.split("::")[-1].strip())

    if not name:
        return None
    raw = expression or legacy or " / ".join(classifiers)
    return name, raw


def installed_licenses() -> dict[str, tuple[str, str]]:
    """Collect declared licence strings for everything installed.

    Returns:
        Mapping of normalized name to ``(display_name, raw_license)``.
    """
    found: dict[str, tuple[str, str]] = {}
    for directory in site_packages_dirs():
        for dist_info in sorted(directory.glob("*.dist-info")):
            parsed = _read_metadata(dist_info)
            if parsed is None:
                continue
            name, raw = parsed
            found.setdefault(normalize(name), (name, raw))
    return found


def resolve(packages: dict[str, str]) -> dict[str, str]:
    """Map every production package to an SPDX licence expression.

    Args:
        packages: Mapping of normalized name to display name.

    Returns:
        Mapping of display name to SPDX expression.

    Raises:
        SystemExit: If any package cannot be resolved, listing what to add to
            ``AMBIGUOUS`` or ``RAW_TO_SPDX``.
    """
    installed = installed_licenses()
    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    for key, display in sorted(packages.items(), key=lambda item: item[0]):
        # Curated entries win over anything on disk: that is what makes the
        # output identical on Linux CI and a macOS workstation.
        override = PLATFORM_ONLY.get(key) or AMBIGUOUS.get(key)
        if override:
            resolved[display] = override
            continue

        entry = installed.get(key)
        if entry is None:
            unresolved.append(
                f"  {display}: not installed here and not curated -- add it to "
                f"PLATFORM_ONLY (if it is OS-specific) or AMBIGUOUS"
            )
            continue

        name, raw = entry
        spdx = RAW_TO_SPDX.get(raw.strip().lower())
        if spdx is None:
            unresolved.append(
                f"  {name}: unrecognised licence string {raw!r} -- add it to "
                f"RAW_TO_SPDX, or pin the package in AMBIGUOUS if the string "
                f"does not identify a specific licence"
            )
            continue
        resolved[name] = spdx

    if unresolved:
        sys.exit(
            "Cannot determine a licence for every dependency. Read the "
            "package's own licence file, then record the result:\n"
            + "\n".join(unresolved)
        )
    return resolved


def copyleft_packages(resolved: dict[str, str]) -> list[str]:
    """List packages whose licence carries obligations beyond attribution.

    Args:
        resolved: Mapping of display name to SPDX expression.

    Returns:
        Display names, sorted case-insensitively.
    """
    return sorted(
        (name for name, spdx in resolved.items() if COPYLEFT_RE.search(spdx)),
        key=str.lower,
    )


def render(resolved: dict[str, str]) -> str:
    """Render the notices Markdown.

    Args:
        resolved: Mapping of display name to SPDX expression.

    Returns:
        The full file contents, newline-terminated.
    """
    present = {normalize(name) for name in resolved}
    by_license: dict[str, list[str]] = defaultdict(list)
    for name, spdx in resolved.items():
        by_license[spdx].append(name)

    out: list[str] = []
    out.append("# Third-Party Notices")
    out.append("")
    out.append(
        "<!-- Generated by scripts/gen-third-party-notices.py. Do not edit by "
        "hand; run the script instead. -->"
    )
    out.append("")
    out.append(
        "OrionBelt Analytics is licensed under the Business Source License "
        "1.1 (see [`LICENSE`](LICENSE)). This file covers the third-party "
        "software it depends on."
    )
    out.append("")
    out.append(
        "The published Docker image bundles the full production dependency "
        "closure, so distributing it redistributes these packages and their "
        "licence terms travel with them. The PyPI wheel does not: it only "
        "declares its dependencies, which the installer fetches from PyPI, so "
        "nothing listed here is redistributed by the wheel."
    )
    out.append("")
    out.append(
        f"**Verbatim licence texts** are shipped inside the image, both under "
        f"each package's own `*.dist-info/` directory in `/opt/venv` and "
        f"collected into `{BUNDLE_PATH_IN_IMAGE}`. This file records which "
        f"packages are present and under what terms; it omits version numbers "
        f"deliberately, so that a routine dependency bump does not churn it."
    )
    out.append("")
    out.append(
        "The image also contains a Debian base with system packages "
        "(chromium, libpq5, fonts). Their copyright files remain in place "
        "under `/usr/share/doc/*/copyright` and are not duplicated here."
    )
    out.append("")

    out.append("## Licences requiring specific attention")
    out.append("")
    rendered_any = False
    for heading, covered, body in SPECIAL_NOTICES:
        if not any(normalize(pkg) in present for pkg in covered):
            continue
        rendered_any = True
        out.append(f"### {heading}")
        out.append("")
        out.append(body)
        out.append("")
    if not rendered_any:
        out.append("None: every dependency is under a permissive licence.")
        out.append("")

    out.append("## Summary by licence")
    out.append("")
    out.append("| Licence | Packages |")
    out.append("| --- | --- |")
    out.extend(
        f"| {spdx} | {len(by_license[spdx])} |"
        for spdx in sorted(by_license, key=str.lower)
    )
    out.append("")
    out.append(f"{len(resolved)} packages in total.")
    out.append("")

    out.append("## Packages")
    out.append("")
    out.append("| Package | Licence | Project |")
    out.append("| --- | --- | --- |")
    out.extend(
        f"| {name} | {resolved[name]} | "
        f"<https://pypi.org/project/{normalize(name)}/> |"
        for name in sorted(resolved, key=str.lower)
    )
    out.append("")

    # Emitted already stripped of trailing whitespace so the pre-commit
    # trailing-whitespace hook cannot rewrite the file and make --check
    # disagree with what the generator produces.
    return "\n".join(line.rstrip() for line in out).rstrip() + "\n"


def dump_texts(packages: dict[str, str], destination: Path) -> None:
    """Collect every installed licence and NOTICE file into one bundle.

    Intended for the Docker builder stage, where the environment holds exactly
    the production closure. Texts are deduplicated by content, so the many
    packages shipping an unmodified Apache-2.0 text share a single copy.

    Args:
        packages: Mapping of normalized name to display name.
        destination: File to write; parent directories are created.
    """
    by_hash: dict[str, tuple[str, list[str]]] = {}
    for directory in site_packages_dirs():
        for dist_info in sorted(directory.glob("*.dist-info")):
            parsed = _read_metadata(dist_info)
            if parsed is None:
                continue
            name, _ = parsed
            if normalize(name) not in packages:
                continue
            for path in sorted(dist_info.rglob("*")):
                if not path.is_file() or not LICENSE_FILE_RE.match(path.name):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                entry = by_hash.setdefault(digest, (text, []))
                if name not in entry[1]:
                    entry[1].append(name)

    chunks: list[str] = [
        "THIRD-PARTY LICENCE TEXTS",
        "=========================",
        "",
        "Verbatim licence and NOTICE files for every third-party package",
        "bundled in this OrionBelt Analytics distribution. Identical texts are",
        "listed once against every package that ships them. See",
        "THIRD_PARTY_NOTICES.md for the package/licence index and for the",
        "notices that carry obligations beyond attribution.",
        "",
        "OrionBelt Analytics itself is licensed under the Business Source",
        "License 1.1; see LICENSE.",
        "",
    ]
    for _, (text, names) in sorted(
        by_hash.items(), key=lambda item: sorted(item[1][1], key=str.lower)
    ):
        chunks.append("=" * 78)
        chunks.append(
            "The following applies to: " + ", ".join(sorted(names, key=str.lower))
        )
        chunks.append("=" * 78)
        chunks.append("")
        chunks.append(text.rstrip())
        chunks.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {destination} ({len(by_hash)} distinct texts)")


def main() -> int:
    """Entry point.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if THIRD_PARTY_NOTICES.md is out of date instead of writing it",
    )
    parser.add_argument(
        "--dump-texts",
        metavar="PATH",
        type=Path,
        help="write the verbatim licence-text bundle (used by the Docker build)",
    )
    args = parser.parse_args()

    packages = production_packages()

    if args.dump_texts is not None:
        dump_texts(packages, args.dump_texts)
        return 0

    resolved = resolve(packages)

    unreviewed = [
        name
        for name in copyleft_packages(resolved)
        if normalize(name) not in REVIEWED_COPYLEFT
    ]
    if unreviewed:
        sys.exit(
            "New dependencies under a copyleft or share-alike licence need a "
            "human read before they ship:\n"
            + "\n".join(f"  {name}: {resolved[name]}" for name in unreviewed)
            + "\n\nAdd a SPECIAL_NOTICES entry describing the obligation, then "
            "list the package in REVIEWED_COPYLEFT."
        )

    content = render(resolved)

    if args.check:
        current = (
            NOTICES_PATH.read_text(encoding="utf-8") if NOTICES_PATH.exists() else ""
        )
        if current != content:
            sys.exit(
                f"{NOTICES_PATH.name} is out of date. Regenerate it with:\n"
                f"  uv run python scripts/gen-third-party-notices.py"
            )
        print(f"{NOTICES_PATH.name} is up to date ({len(resolved)} packages).")
        return 0

    NOTICES_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {NOTICES_PATH} ({len(resolved)} packages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
