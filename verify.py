#!/usr/bin/env python3
"""
Verify this repository.

Two jobs, and the first one is the point of the file.

NUMBERS. Every figure quoted in README.md is listed below with the code path
that produces it. If a number in the README and a number in the code ever
disagree, this script fails. A README is a claim; this makes it a testable one.

HYGIENE. Checks that no data file is about to be committed, that no absolute
path or credential-shaped string has crept in, and that no em dash has appeared
in prose.

    python verify.py              # everything, about a minute
    python verify.py --fast       # skip the Cox models
    python verify.py --hygiene    # hygiene only, no data needed

Exit code 0 means every check passed.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The numbers README.md quotes. Left side is a path into the results structure,
# right side is (expected value, tolerance).
#
# Tolerances are set to the precision the README prints at: a rate quoted as
# "3.02%" is checked to 5e-3, a hazard ratio quoted as "1.774" to 5e-4. They are
# not there to absorb a real change, and a real change should fail this file and
# then be corrected in BOTH places.
# ---------------------------------------------------------------------------
EXPECTED = [
    ("cohort.n_patients", 233647, 0),
    ("cohort.deaths", 8341, 0),
    ("cohort.death_rate_pct", 3.57, 5e-3),
    ("cohort.male_pct", 40.5, 5e-2),
    ("cohort.mean_age", 50.74, 5e-3),
    ("cohort.mean_predicted_age", 51.90, 5e-3),
    ("cohort.mae_years", 8.45, 5e-3),
    ("cohort.pearson_r_predicted_vs_chronological", 0.8381, 5e-5),

    ("confound.pearson_r_pad_vs_age", -0.3842, 5e-5),
    ("confound.ols_slope", -0.2131, 5e-5),
    ("confound.ols_intercept", 11.9755, 5e-5),

    ("raw.older.n", 56400, 0),
    ("raw.older.deaths", 1702, 0),
    ("raw.older.death_rate_pct", 3.02, 5e-3),
    ("raw.older.mean_age", 42.2, 5e-2),
    ("raw.younger.n", 43121, 0),
    ("raw.younger.death_rate_pct", 4.57, 5e-3),
    ("raw.younger.mean_age", 62.9, 5e-2),

    ("corrected.older.n", 47263, 0),
    ("corrected.older.deaths", 2431, 0),
    ("corrected.older.death_rate_pct", 5.14, 5e-3),
    ("corrected.older.mean_age", 52.7, 5e-2),
    ("corrected.younger.death_rate_pct", 2.57, 5e-3),
    ("corrected.younger.mean_age", 52.2, 5e-2),

    ("pooled_correction_by_sex.male.mean_padbc", 0.372, 5e-4),
    ("pooled_correction_by_sex.female.mean_padbc", -0.254, 5e-4),
    ("pooled_correction_by_sex.gap_years", 0.626, 5e-4),

    ("within_sex_correction.male.slope", -0.2100, 5e-5),
    ("within_sex_correction.female.slope", -0.2159, 5e-5),
    ("within_sex_correction.rate_ratio_male", 1.89, 5e-3),
    ("within_sex_correction.rate_ratio_female", 1.96, 5e-3),

    ("normal_ecg_only.n", 92138, 0),
    ("normal_ecg_only.pearson_r_pad_vs_age", -0.4160, 5e-5),
    ("normal_ecg_only.raw.older_vs_younger_rate_ratio", 0.59, 5e-3),
    ("normal_ecg_only.corrected_within_normal.older_vs_younger_rate_ratio", 2.04, 5e-3),
]

EXPECTED_COX = [
    ("models.raw_categorical.terms.PAD > +8 (older).hr", 1.774, 5e-4),
    ("models.raw_categorical.terms.PAD < -8 (younger).hr", 0.791, 5e-4),
    ("models.raw_categorical.terms.age (per year).hr", 1.074, 5e-4),
    ("models.raw_categorical.terms.male.hr", 1.611, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc > +8 (older).hr", 1.586, 5e-4),
    ("models.corrected_categorical.terms.PAD_bc < -8 (younger).hr", 0.771, 5e-4),
    ("models.raw_continuous.terms.PAD (per year).hr", 1.0293, 5e-5),
    ("models.corrected_continuous.terms.PAD_bc (per year).hr", 1.0293, 5e-5),
]


def dig(obj, path):
    """Walk a dotted path, tolerating keys that themselves contain dots."""
    cur = obj
    rest = path
    while rest:
        for key in sorted(cur.keys(), key=len, reverse=True):
            if rest == key:
                return cur[key]
            if rest.startswith(key + "."):
                cur = cur[key]
                rest = rest[len(key) + 1:]
                break
        else:
            raise KeyError("no such path: %s (stuck at %r)" % (path, rest))
    return cur


def check_values(results, expected, label):
    ok = True
    print("\n%s" % label)
    for path, want, tol in expected:
        try:
            got = dig(results, path)
        except KeyError as exc:
            print("  MISSING  %-62s %s" % (path, exc))
            ok = False
            continue
        good = abs(got - want) <= tol
        print("  %-8s %-62s want %-12s got %s"
              % ("ok" if good else "MISMATCH", path, want,
                 got if isinstance(got, int) else "%.6f" % got))
        ok = ok and good
    return ok


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
TEXT_SUFFIXES = {".py", ".md", ".txt", ".cff", ".yml", ".yaml", ".toml", ".cfg",
                 ".sh", ".ps1", ".json", ".gitignore", ".in", ""}
SKIP_DIRS = {".git", "data", "results", "__pycache__", ".venv", "venv"}

# EVERY quantifier below is BOUNDED, and that is not a style preference.
# The email pattern was first written with unbounded "+" quantifiers either side
# of the @. On a single 200,000 character line that takes 170 seconds, because
# the engine tries every start position and backtracks across the whole line:
# the cost is quadratic in line length. Bounding the quantifiers brings the same
# line down to 0.09 seconds, a factor of about 1900. A scanner that hangs on a
# minified file or an embedded blob is a scanner nobody runs.
SECRET_PATTERNS = [
    ("generic api key assignment",
     re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|token)\b[ \t]{0,8}"
                r"[=:][ \t]{0,8}['\"][A-Za-z0-9_\-]{12,200}['\"]")),
    ("bearer token", re.compile(r"(?i)\bBearer[ \t]{1,4}[A-Za-z0-9._\-]{20,400}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,255}\b")),
    ("openai style key", re.compile(r"\bsk-[A-Za-z0-9]{20,255}\b")),
    ("groq style key", re.compile(r"\bgsk_[A-Za-z0-9]{20,255}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,4000}\.[A-Za-z0-9_\-]{10,4000}\.")),
    ("absolute windows path", re.compile(r"[A-Za-z]:\\{1,2}[Uu]sers\\")),
    ("unix home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._\-]{1,64}/")),
    ("email address",
     re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}")),
]

# Lines longer than this are not pattern scanned, because the cost of scanning
# them is not worth paying and a very long line in a source repository is itself
# worth looking at. It is REPORTED rather than skipped silently, so this is a
# visible limitation and not a hole.
MAX_SCAN_LINE = 4000

# Em dash, as a literal codepoint and as all three HTML entity spellings. On
# some systems grep will not match U+2014 at all, which is why this is done in
# Python rather than in a shell one-liner.
#
# The pattern is ASSEMBLED rather than written out, so that this file does not
# itself contain an em dash or an entity and therefore does not match itself.
# The alternative, excluding this file from the scan, would mean a real em dash
# typed into the prose here would never be caught. A scanner that cannot check
# itself is one file short of a scanner.
_EM_DASH_ALTERNATIVES = [chr(0x2014)] + ["&" + name for name in
                                         ("mdash;", "#8212;", "#x2014;")]
EM_DASH = re.compile("|".join(_EM_DASH_ALTERNATIVES), re.IGNORECASE)


def walk_every_file(root):
    """
    Every file at every depth. Written as an explicit walk rather than a glob
    because a shallow scan is exactly how a credential gets missed: a scan that
    stops at four directories deep reports CLEAN on a secret that is five deep.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


# ---------------------------------------------------------------------------
# Proving the scanner can fire
#
# A clean scan is worth nothing on its own. This builds a throwaway tree in a
# temporary directory, plants one violation of every class, at five directories
# deep, and asserts the scanner finds all of them. Nothing is planted in the
# repository itself and nothing is left behind.
#
# The planted strings are assembled from fragments so that THIS file does not
# contain any of them literally and therefore does not report itself. They are
# invalid by construction and are not credentials.
# ---------------------------------------------------------------------------
def scanner_selftest():
    import shutil
    import tempfile

    j = "".join
    planted = {
        "github token": j(["gh", "p_", "A" * 36]),
        "aws access key id": j(["AKIA", "ABCDEFGHIJKLMNOP"]),
        "openai style key": j(["sk", "-", "A" * 32]),
        "groq style key": j(["gsk", "_", "B" * 32]),
        "slack token": j(["xox", "b-", "1234567890-abcdef"]),
        "bearer token": j(["Bearer", " ", "C" * 30]),
        "private key block": j(["-----BEGIN", " RSA PRIVATE KEY", "-----"]),
        "jwt": j(["eyJ", "abcdefghijk", ".", "eyJabcdefghijk", ".", "sig"]),
        "generic api key assignment": j(['api_key', ' = ', '"', "D" * 24, '"']),
        "absolute windows path": j(["C", ":", chr(92), "Users", chr(92), "someone"]),
        "unix home path": j(["/home", "/", "someone", "/"]),
        "email address": j(["someone", "@", "example", ".", "org"]),
        "em dash": chr(0x2014),
    }

    root = tempfile.mkdtemp(prefix="ecgscan_")
    try:
        deep = os.path.join(root, "a", "b", "c", "d", "e")
        os.makedirs(deep)
        for i, (label, text) in enumerate(sorted(planted.items())):
            # .md so that the em dash rule applies too.
            with open(os.path.join(deep, "planted_%02d.md" % i), "w",
                      encoding="utf-8") as fh:
                fh.write("a line that should be caught: %s\n" % text)
        # And a data file, plus something over the size limit.
        with open(os.path.join(deep, "leaked.csv"), "w", encoding="utf-8") as fh:
            fh.write("a,b\n1,2\n")
        with open(os.path.join(deep, "big.txt"), "w", encoding="utf-8") as fh:
            fh.write("x" * 1_200_000)

        found = set()
        for path in walk_every_file(root):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if os.path.getsize(path) > 1_000_000:
                found.add("file over 1 MB")
            if rel.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
                found.add("data file inside the repository")
            try:
                text = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                if len(line) > MAX_SCAN_LINE:
                    found.add("long line")
                    continue
                for label, pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        found.add(label)
                if EM_DASH.search(line):
                    found.add("em dash")

        expected = set(planted) | {"file over 1 MB", "long line",
                                   "data file inside the repository"}
        missed = sorted(expected - found)
        return {"expected": len(expected), "found": len(found & expected),
                "missed": missed, "passed": not missed,
                "depth_tested": 5}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def hygiene(root):
    ok = True
    print("\nHYGIENE")

    st = scanner_selftest()
    print("  scanner self-test: planted %d violation classes at %d directories "
          "deep, caught %d" % (st["expected"], st["depth_tested"], st["found"]))
    if st["passed"]:
        print("    every class fires, so a clean result below means something")
    else:
        print("    MISSED: %s" % ", ".join(st["missed"]))
        ok = False

    n_scanned = 0
    n_binary = 0
    findings = []

    for path in walk_every_file(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        suffix = os.path.splitext(path)[1].lower()
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            findings.append((rel, 0, "unreadable", str(exc)))
            continue
        if b"\x00" in raw:
            n_binary += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            n_binary += 1
            continue
        n_scanned += 1

        for lineno, line in enumerate(text.splitlines(), 1):
            if len(line) > MAX_SCAN_LINE:
                findings.append((rel, lineno,
                                 "line over %d characters, NOT pattern scanned"
                                 % MAX_SCAN_LINE, "%d characters" % len(line)))
                continue
            for label, pattern in SECRET_PATTERNS:
                for m in pattern.finditer(line):
                    # Deliberately no exemption for this file. The patterns are
                    # written so that they do not match their own source, which
                    # was checked by running the scan over this file with no
                    # exemption in place. If a credential were ever pasted in
                    # here, it would be reported like any other.
                    findings.append((rel, lineno, label, m.group(0)[:70]))
            if suffix in (".md", ".txt", ".py", ".cff"):
                for m in EM_DASH.finditer(line):
                    findings.append((rel, lineno, "em dash", line.strip()[:70]))

    print("  files scanned as text: %d   binary or undecodable skipped: %d"
          % (n_scanned, n_binary))

    # No data file anywhere in the tree outside data/, and nothing large tracked.
    for path in walk_every_file(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        size = os.path.getsize(path)
        if size > 1_000_000:
            findings.append((rel, 0, "file over 1 MB", "%d bytes" % size))
        if rel.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
            findings.append((rel, 0, "data file inside the repository", ""))

    # And ask git directly, which is the only authority on what would be pushed.
    try:
        tracked = subprocess.run(
            ["git", "-C", root, "ls-files"],
            capture_output=True, text=True, timeout=30)
        if tracked.returncode == 0:
            for line in tracked.stdout.splitlines():
                if line.lower().endswith((".csv", ".hdf5", ".h5", ".parquet", ".zip")):
                    findings.append((line, 0, "data file TRACKED BY GIT", ""))
            print("  files tracked by git: %d"
                  % len([l for l in tracked.stdout.splitlines() if l]))
        else:
            print("  git reports: not a repository yet (nothing tracked)")
    except (OSError, subprocess.SubprocessError) as exc:
        print("  git unavailable, skipped the tracked-file check: %s" % exc)

    if findings:
        ok = False
        print("  %d FINDING(S):" % len(findings))
        for rel, lineno, label, detail in findings:
            print("    %s:%s  %s  %s" % (rel, lineno, label, detail))
    else:
        print("  no secrets, no absolute paths, no email addresses, no em dashes,")
        print("  no data file in the tree, nothing over 1 MB.")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify this repository.")
    ap.add_argument("--fast", action="store_true", help="skip the Cox models")
    ap.add_argument("--hygiene", action="store_true", help="hygiene checks only")
    args = ap.parse_args(argv)

    all_ok = True

    if not args.hygiene:
        import ecg_age_bias as base
        base.ensure_data(base.DEFAULT_CSV)
        rows, _ = base.load(base.DEFAULT_CSV)
        results = base.analyse(rows)
        all_ok &= check_values(results, EXPECTED,
                               "NUMBERS QUOTED IN README.md (core analysis)")

        if not args.fast:
            import cox_model
            sys.stderr.write("  fitting Cox models, about 40 seconds ...\n")
            code = cox_model.main(["--json",
                                   os.path.join(HERE, "results", "cox.json")])
            if code != 0:
                print("\ncox_model.py exited %d" % code)
                all_ok = False
            else:
                import json
                with open(os.path.join(HERE, "results", "cox.json"),
                          encoding="utf-8") as fh:
                    cox = json.load(fh)
                all_ok &= check_values(cox, EXPECTED_COX,
                                       "NUMBERS QUOTED IN README.md (Cox models)")
                # The Cox implementation's own self-tests print PASS or FAIL
                # inside cox_model.py, which exits 0 either way outside
                # --selftest. Read their verdicts from the JSON so that a
                # failed self-test fails this script too.
                print("\nCOX IMPLEMENTATION SELF-TESTS")
                for name, chk in sorted(cox["checks"].items()):
                    if isinstance(chk, dict) and "passed" in chk:
                        good = chk["passed"] is True
                        print("  %-8s %s" % ("ok" if good else "FAILED", name))
                        all_ok &= good
                ident = cox["checks"]["continuous_identity"]
                good = ident["identical_to_6dp"]
                print("  %-8s continuous PAD and PAD_bc hazard ratios identical"
                      % ("ok" if good else "MISMATCH"))
                all_ok &= good

                # The three attenuation figures the README quotes are DERIVED
                # from the two hazard ratios above, so they cannot drift on
                # their own. They are still checked, because a number can be
                # mistyped in prose without any code changing, and that is the
                # failure mode this whole file exists to catch.
                import math
                raw = dig(cox, "models.raw_categorical.terms.PAD > +8 (older).hr")
                cor = dig(cox,
                          "models.corrected_categorical.terms."
                          "PAD_bc > +8 (older).hr")
                derived = [
                    ("attenuation, hazard ratio scale (%)",
                     100 * (1 - cor / raw), 10.6, 5e-2),
                    ("attenuation, log hazard ratio scale (%)",
                     100 * (1 - math.log(cor) / math.log(raw)), 19.6, 5e-2),
                    ("attenuation, excess hazard scale (%)",
                     100 * (1 - (cor - 1) / (raw - 1)), 24.3, 5e-2),
                    ("excess hazard, raw (%)", 100 * (raw - 1), 77.4, 5e-2),
                    ("excess hazard, corrected (%)", 100 * (cor - 1), 58.6, 5e-2),
                ]
                print("\nATTENUATION FIGURES QUOTED IN README.md")
                for label, got, want, tol in derived:
                    good = abs(got - want) <= tol
                    print("  %-8s %-42s want %-7s got %.4f"
                          % ("ok" if good else "MISMATCH", label, want, got))
                    all_ok &= good

    all_ok &= hygiene(HERE)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOMETHING FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
