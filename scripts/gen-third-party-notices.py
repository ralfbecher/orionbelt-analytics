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
import textwrap
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
    "mit and psf-2.0": "MIT AND PSF-2.0",
    "mit and python-2.0": "MIT AND Python-2.0",
    "mit license": "MIT",
    "mit or afl-2.1": "MIT OR AFL-2.1",
    "mit or apache-2.0": "MIT OR Apache-2.0",
    "mit-0": "MIT-0",
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

# Canonical licence texts kept in-tree, for packages that ship none of their
# own. Only licences with a single verbatim upstream text belong here -- MIT
# and BSD are templates carrying a per-project copyright line, so they cannot
# be supplied generically and must come from the package itself.
VENDORED_TEXTS_DIR = Path(__file__).resolve().parent / "license-texts"

# Recognised as attribution material wherever it is installed. ThirdPartyNotices
# is included because a package that vendors its own dependencies (onnxruntime
# ships 325 KB of them) passes their notices on to us along with the binary.
LICENSE_FILE_RE = re.compile(
    r"^(LICEN[CS]E|COPYING|NOTICE|THIRD[-_ ]?PARTY[-_ ]?NOTICES)",
    re.IGNORECASE,
)

# A file named like a licence but carrying code is a module, not a notice.
NON_TEXT_SUFFIXES = frozenset({".py", ".pyc", ".pyi", ".pyd", ".so", ".dll", ".dylib"})

# Packages that ship no licence file anywhere in their distribution. Every
# entry is (SPDX licence to reproduce, upstream source, attribution note), and
# each fact below is sourced from the installed distribution itself -- its
# METADATA, or the licence headers in the source files it installs -- rather
# than from memory. Without an entry here a package is dropped from the bundle
# silently, which is why collect_texts() refuses to proceed instead.
MISSING_TEXT_FALLBACK: dict[str, tuple[str, str, str]] = {
    "fastmcp-slim": (
        "Apache-2.0",
        "https://github.com/PrefectHQ/fastmcp",
        "Author, per package metadata: Jeremiah Lowin.",
    ),
    "flatbuffers": (
        "Apache-2.0",
        "https://github.com/google/flatbuffers",
        "Copyright 2014 Google Inc. All rights reserved. (from the Apache "
        "licence headers on the source files in this distribution)",
    ),
    "py-key-value-aio": (
        "Apache-2.0",
        "https://pypi.org/project/py-key-value-aio/",
        "",
    ),
    "pyoxigraph": (
        "Apache-2.0",
        "https://github.com/oxigraph/oxigraph/tree/main/python",
        "Author, per package metadata: Tpt <thomas@pellissier-tanon.fr>. "
        "Dual-licensed MIT OR Apache-2.0; Apache-2.0 is elected and "
        "reproduced here, the MIT alternative being a template whose "
        "copyright line the distribution does not carry.",
    ),
    "thrift": (
        "Apache-2.0",
        "https://github.com/apache/thrift",
        "Licensed to the Apache Software Foundation (ASF) under one or more "
        "contributor licence agreements (from the licence headers on the "
        "source files in this distribution).",
    ),
    "tokenizers": (
        "Apache-2.0",
        "https://github.com/huggingface/tokenizers",
        "Authors, per package metadata: Nicolas Patry, Anthony Moi.",
    ),
}


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
        "A few packages ship no licence file of their own. Rather than being "
        "dropped from the bundle, each carries a notice naming its licence, "
        "the attribution recorded in its own metadata or source headers, and "
        "the upstream source; the generator refuses to run if a package is "
        "covered by neither."
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


def _license_files(dist_info: Path) -> list[Path]:
    """Find every licence-like file a distribution installs.

    Searching only ``*.dist-info`` misses packages that put their licence
    inside the package directory instead -- onnxruntime is the example in this
    tree -- so the installed-file manifest is consulted as well.

    Args:
        dist_info: A ``*.dist-info`` directory.

    Returns:
        Existing files, deduplicated, in a stable order.
    """
    found: list[Path] = []

    def consider(path: Path) -> None:
        if path.suffix.lower() in NON_TEXT_SUFFIXES:
            return
        if not LICENSE_FILE_RE.match(path.name):
            return
        if path.is_file() and path not in found:
            found.append(path)

    for path in sorted(dist_info.rglob("*")):
        consider(path)

    record = dist_info / "RECORD"
    if record.is_file():
        site_dir = dist_info.parent
        with record.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                relative = line.split(",", 1)[0].strip()
                if not relative:
                    continue
                candidate = (site_dir / relative).resolve()
                # A RECORD entry may point outside site-packages (scripts,
                # data files); anything beyond the tree is not ours to ship.
                if site_dir.resolve() in candidate.parents:
                    consider(candidate)

    return sorted(found)


def _fallback_entry(display: str) -> str:
    """Build a notice for a package that ships no licence file.

    Args:
        display: The package's display name.

    Returns:
        The notice body, including the vendored canonical licence text.

    Raises:
        SystemExit: If the vendored text for the licence is missing.
    """
    spdx, source, attribution = MISSING_TEXT_FALLBACK[normalize(display)]
    text_path = VENDORED_TEXTS_DIR / f"{spdx}.txt"
    if not text_path.is_file():
        sys.exit(
            f"{display} needs the canonical {spdx} text, but "
            f"{text_path} does not exist."
        )

    # Wrapped to match the width of the licence texts it sits alongside.
    lines = textwrap.wrap(
        f"{display} ships no licence file in its distribution. Its declared "
        f"licence is {spdx}; the canonical text is reproduced below.",
        width=78,
    )
    lines.append("")
    if attribution:
        lines.extend(textwrap.wrap(attribution, width=78))
        lines.append("")
    lines.extend([f"Source and licence notice: {source}", "", "-" * 78, ""])
    lines.append(text_path.read_text(encoding="utf-8").rstrip())
    return "\n".join(lines)


def collect_texts(
    packages: dict[str, str],
) -> tuple[dict[str, tuple[str, list[str]]], list[str]]:
    """Gather the licence text of every installed production package.

    Args:
        packages: Mapping of normalized name to display name.

    Returns:
        ``(texts, uncovered)`` where ``texts`` maps a content hash to
        ``(text, packages)`` -- identical texts are shared, so the many
        packages carrying an unmodified Apache-2.0 copy collapse into one
        entry -- and ``uncovered`` lists installed packages that ship no
        licence file and have no curated fallback.
    """
    by_hash: dict[str, tuple[str, list[str]]] = {}
    uncovered: list[str] = []

    def record_text(text: str, display: str) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = by_hash.setdefault(digest, (text, []))
        if display not in entry[1]:
            entry[1].append(display)

    for directory in site_packages_dirs():
        for dist_info in sorted(directory.glob("*.dist-info")):
            parsed = _read_metadata(dist_info)
            if parsed is None:
                continue
            display, _ = parsed
            key = normalize(display)
            if key not in packages:
                continue

            files = _license_files(dist_info)
            if files:
                for path in files:
                    record_text(
                        path.read_text(encoding="utf-8", errors="replace"),
                        display,
                    )
            elif key in MISSING_TEXT_FALLBACK:
                record_text(_fallback_entry(display), display)
            else:
                uncovered.append(display)

    return by_hash, sorted(uncovered, key=str.lower)


def require_full_coverage(uncovered: list[str]) -> None:
    """Abort if any installed production package has no licence text.

    Silently dropping such a package is the failure that matters here: the
    bundle would still be written, the image would still build, and the
    missing attribution would ship.

    Args:
        uncovered: Display names with neither a shipped text nor a fallback.

    Raises:
        SystemExit: If ``uncovered`` is non-empty.
    """
    if not uncovered:
        return
    sys.exit(
        "These bundled packages ship no licence file and have no curated "
        "fallback, so their attribution would be missing from the collected "
        "bundle:\n"
        + "\n".join(f"  {name}" for name in uncovered)
        + "\n\nRead the package's own metadata and source headers, then add an "
        "entry to MISSING_TEXT_FALLBACK (and, if its licence has no vendored "
        "canonical text yet, add one under scripts/license-texts/)."
    )


def dump_texts(packages: dict[str, str], destination: Path) -> None:
    """Write the verbatim licence-text bundle shipped inside the image.

    Args:
        packages: Mapping of normalized name to display name.
        destination: File to write; parent directories are created.
    """
    by_hash, uncovered = collect_texts(packages)
    require_full_coverage(uncovered)

    chunks: list[str] = [
        "THIRD-PARTY LICENCE TEXTS",
        "=========================",
        "",
        "Verbatim licence and NOTICE files for every third-party package",
        "bundled in this OrionBelt Analytics distribution. Identical texts are",
        "listed once against every package that ships them. Packages that ship",
        "no licence file of their own carry a notice naming their licence and",
        "upstream source instead. See THIRD_PARTY_NOTICES.md for the",
        "package/licence index and for the notices that carry obligations",
        "beyond attribution.",
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
    covered = sorted({name for _, names in by_hash.values() for name in names})
    print(
        f"Wrote {destination} ({len(by_hash)} distinct texts, "
        f"{len(covered)} packages)"
    )


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

    # Coverage is validated here too, not just in --dump-texts, so a package
    # shipping no licence file fails in this 17-second job rather than in the
    # six-minute image build -- or, worse, at release time.
    _, uncovered = collect_texts(packages)
    require_full_coverage(uncovered)

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
        print(
            f"{NOTICES_PATH.name} is up to date ({len(resolved)} packages, "
            f"licence texts complete)."
        )
        return 0

    NOTICES_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {NOTICES_PATH} ({len(resolved)} packages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
