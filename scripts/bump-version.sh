#!/usr/bin/env bash
# Bump the OrionBelt Analytics version everywhere it is recorded, in one step.
#
# Updates: src/__init__.py, server.json (both version fields), the README
# badge, and prepends a CHANGELOG stub. Then runs `uv lock` to keep the
# lockfile in step.
#
# pyproject.toml is NOT touched: [project] declares the version dynamic and
# hatch reads it from src/__init__.py. That also keeps it, and uv.lock, out of
# the churn — both are inputs to the Docker dependency layer, so a bump that
# edited them forced a full cold reinstall on every release. `uv lock` is still
# run because a bump can coincide with other manifest edits; it is now
# expected to be a no-op for the version alone.
#
# Usage:
#   ./scripts/bump-version.sh X.Y.Z
#
# After running: review the diff, fill in the CHANGELOG bullets, commit, and
# continue with the release steps (PR -> squash merge -> tag -> PyPI -> Docker).

set -euo pipefail

NEW_VERSION="${1:-}"
if [[ -z "${NEW_VERSION}" ]]; then
    echo "usage: $0 X.Y.Z" >&2
    exit 1
fi
if [[ ! "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: '${NEW_VERSION}' is not a semantic version (expected X.Y.Z)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Source of truth for the current version.
OLD_VERSION="$(grep -E '^__version__' src/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "${OLD_VERSION}" ]]; then
    echo "error: could not read __version__ from src/__init__.py" >&2
    exit 1
fi
if [[ "${OLD_VERSION}" == "${NEW_VERSION}" ]]; then
    echo "error: version is already ${NEW_VERSION}" >&2
    exit 1
fi

echo "Bumping ${OLD_VERSION} -> ${NEW_VERSION}"

# Replace a literal old->new string in a file, but fail if the expected text is
# absent — so a format change surfaces immediately instead of silently skipping.
replace() {
    local file="$1" search="$2" replacement="$3"
    if ! grep -qF -- "${search}" "${file}"; then
        echo "error: expected to find '${search}' in ${file} — file format may have changed" >&2
        exit 1
    fi
    # Use perl for literal (\Q..\E) matching, consistent across macOS/Linux.
    SEARCH="${search}" REPLACE="${replacement}" perl -i -pe \
        's/\Q$ENV{SEARCH}\E/$ENV{REPLACE}/g' "${file}"
    echo "  updated ${file}"
}

# pyproject.toml is deliberately absent: [project] declares the version
# dynamic and hatch reads it from src/__init__.py above, so there is nothing
# to replace. Keeping it out is what lets a bump leave the Docker dependency
# layer's inputs untouched -- see the comment in pyproject.toml.
replace src/__init__.py  "__version__ = \"${OLD_VERSION}\"" "__version__ = \"${NEW_VERSION}\""
replace server.json      "\"version\": \"${OLD_VERSION}\""   "\"version\": \"${NEW_VERSION}\""
replace README.md        "version-${OLD_VERSION}"            "version-${NEW_VERSION}"
replace README.md        "Version ${OLD_VERSION}"            "Version ${NEW_VERSION}"

# Prepend a CHANGELOG entry stub above the most recent version section.
CHANGELOG="CHANGELOG.md"
if [[ -f "${CHANGELOG}" ]]; then
    TODAY="$(date +%Y-%m-%d)"
    STUB="## [${NEW_VERSION}] - ${TODAY}\n\n### Changed\n- TODO: describe changes\n\n"
    # Insert before the first existing '## [' version heading.
    NEW_VERSION_HEADING="${NEW_VERSION}" STUB="${STUB}" perl -i -pe '
        if (!$done && /^## \[/) {
            my $s = $ENV{STUB}; $s =~ s/\\n/\n/g; print $s; $done = 1;
        }
    ' "${CHANGELOG}"
    echo "  inserted CHANGELOG stub for ${NEW_VERSION} (${TODAY}) — fill in the bullets"
else
    echo "  warning: ${CHANGELOG} not found, skipping changelog stub" >&2
fi

# Regenerate the lockfile so its pinned project version matches — the step that
# is easy to forget by hand and the whole reason this script exists.
echo "Running uv lock..."
uv lock

echo
echo "Done. Next steps:"
echo "  1. Edit ${CHANGELOG} — replace the TODO bullets with the real changes."
echo "  2. Review:  git diff"
echo "  3. Commit (include uv.lock), open a PR, squash-merge, tag, publish."
