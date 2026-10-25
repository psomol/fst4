#!/bin/sh
# FST4 golden-output regression harness.
#
# Usage:
#   tests/golden.sh capture [demo...]   (re)record golden outputs
#   tests/golden.sh check   [demo...]   run demos and diff against goldens
#   tests/golden.sh check-values [demo...]  compare RESULT LINES at print
#       precision instead of whole raw bytes. A cross-platform smoke, NOT a
#       replacement for per-platform byte-exact baselines: it tolerates
#       low-order FP digits (libm/codegen differences) while still catching a
#       changed subset, criterion value or trajectory. Floating-point tokens are
#       compared rounded to FST_VALUE_PRECISION significant digits (default 6);
#       integers (feature indices, counts) are compared exactly.
#   tests/golden.sh time    [demo...]   run demos, report duration + exit code only
#
# With no demo arguments, operates on every non-skip entry in tests/manifest.tsv,
# excluding the "slow" tier; set FST_GOLDEN_SLOW=1 to include slow demos, or name
# them explicitly.
# Executables are looked up in the repo root by default; set FST_BIN=<dir>
# (absolute or repo-root-relative, forward slashes) to gate a CMake build tree,
# e.g. FST_BIN=build-win/bin (build-mac/bin on macOS, build-linux/bin on Linux).
# The golden baseline directory is chosen PER PLATFORM: tests/golden-win-x64 on
# native Windows x86-64, tests/golden-macos-arm64 on Apple Silicon,
# tests/golden-linux-arm64 on arm64 Linux (tests/golden-linux-x64 on x86-64
# Linux, for which no baseline ships - capture your own); any other platform
# must set FST_GOLDEN_DIR=<dir> (absolute or repo-root-relative) explicitly.
# FST_GOLDEN_DIR always overrides the default.
# OPTIONAL EXTERNAL DEPENDENCIES. Neither one's absence is an FST4 defect, and a
# scenario that cannot run must never be reported as a failure of the codebase.
# The two behave in OPPOSITE ways, so they are handled differently:
#
#  * LibSVM — scenarios that need it CANNOT RUN without it. All of them request it
#    via --criterion wrapper-svm, which the criterion factory refuses at startup
#    with its own message; that refusal is detected here and the scenario is
#    SKIPPED rather than failed. Detecting the message (rather than tagging the
#    manifest) means a new wrapper-svm scenario is covered automatically. Such
#    skips are counted and reported SEPARATELY, so a run with unrunnable
#    scenarios never looks identical to a fully-green run.
#    NOTE the coupling: the detection matches the wording of the fst_error thrown
#    in fst4.cpp. If that message is reworded, update the grep below — otherwise
#    these scenarios silently go back to reporting FAIL.
#
#  * Eigen — NOTHING here depends on it, so there is nothing to skip. FST4_USE_EIGEN
#    defaults OFF, the goldens were captured with the hand-rolled LU path, and
#    CMake hard-errors at configure time if Eigen is requested without a valid
#    FST4_EIGEN_DIR. A MISSING Eigen therefore cannot fail a scenario. The hazard
#    is the REVERSE: an Eigen-ENABLED build swaps Indexed_Matrix's
#    LU/inverse/determinant and shifts low-order FP bits BY DESIGN, so every
#    normal-model scenario can legitimately mismatch. Those are NOT regressions,
#    and this script warns up front when it detects such a build.
#
# Exit code: 0 = all pass, 1 = at least one failure, 2 = usage/setup error.
# Scenarios skipped for a missing optional dependency do NOT affect the exit code.
#
# See tests/README.md for the per-demo tier/mode rationale.

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
MANIFEST="$HERE/manifest.tsv"
# Per-platform golden baseline default (see header); FST_GOLDEN_DIR overrides.
if [ -n "${FST_GOLDEN_DIR:-}" ]; then
  GOLDEN=$FST_GOLDEN_DIR
else
  case "$(uname -s)-$(uname -m)" in
    MINGW*-x86_64|MSYS*-x86_64) GOLDEN="$HERE/golden-win-x64" ;;
    Darwin-arm64)               GOLDEN="$HERE/golden-macos-arm64" ;;
    # Linux: capture creates the directory, so a first capture works with no
    # FST_GOLDEN_DIR. arm64 Linux builds under the same golden-preset flags as
    # macOS arm64 and still gets its OWN baseline — never point it at
    # golden-macos-arm64. (The two arm64 baselines are in fact byte-identical on
    # all but one scenario; the separate baseline is cheap insurance against
    # musl, other glibc versions and x86-64 Linux.)
    Linux-x86_64)               GOLDEN="$HERE/golden-linux-x64" ;;
    Linux-aarch64|Linux-arm64)  GOLDEN="$HERE/golden-linux-arm64" ;;
    *) echo "no default golden baseline for $(uname -s)-$(uname -m); set FST_GOLDEN_DIR" >&2; exit 2 ;;
  esac
fi
# Per-run scratch dir for .raw/.out/.result intermediates AND for the side-file
# artifacts two scenarios write (--dump-scores, --curves-svg; the manifest reaches
# it as $FST_LASTRUN, which is why it is exported). Override FST_LASTRUN to run
# two golden runs against ONE tree without them clobbering each other's files —
# e.g. a container gate alongside a host gate. The scratch path leaks into those
# two scenarios' invocation echo, so normalize() rewrites whatever directory was
# used back to the canonical token below; goldens therefore stay byte-identical
# and machine-independent no matter where the scratch dir lives.
LASTRUN="${FST_LASTRUN:-$HERE/last_run}"
export FST_LASTRUN="$LASTRUN"
LASTRUN_CANON="tests/last_run"
# BRE-escaped form of $LASTRUN for the normalize() rewrite (| is the delimiter).
LASTRUN_RE=$(printf '%s' "$LASTRUN" | sed 's/[\\.^$*[]/\\&/g; s/|/\\|/g')
# On native Windows (Git Bash / MSYS) the same directory reaches the native
# fst4 process in mixed Windows form (C:/Users/...), not the MSYS form
# (/c/Users/...) that $LASTRUN holds — so the invocation echo prints a path
# the pattern above cannot match, and the machine path would leak into the
# golden. Compute the mixed form too and normalize() rewrites BOTH.
if command -v cygpath >/dev/null 2>&1; then
  LASTRUN_WIN=$(cygpath -m "$LASTRUN" 2>/dev/null || printf '')
else
  LASTRUN_WIN=""
fi
LASTRUN_WIN_RE=$(printf '%s' "$LASTRUN_WIN" | sed 's/[\\.^$*[]/\\&/g; s/|/\\|/g')

# On native Windows (Git Bash / MSYS) the MinGW runtime DLLs must win over
# Git Bash's own incompatible libstdc++-6.dll (otherwise every process dies at
# load time with a bogus segfault and no output).
case "$(uname -s)" in
  MINGW*|MSYS*) [ -d /c/mingw64/bin ] && PATH="/c/mingw64/bin:$PATH" ;;
esac

# Stock macOS ships no timeout(1): fall back to gtimeout (brew coreutils), or,
# if neither exists, run unguarded (timeouts then unenforced — only hangs suffer).
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN=timeout
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN=gtimeout
else TIMEOUT_BIN=""
fi
run_bounded() { # $1=timeout-seconds, $2...=command
  _t=$1; shift
  if [ -n "$TIMEOUT_BIN" ]; then "$TIMEOUT_BIN" "$_t" "$@"; else "$@"; fi
}

cd "$ROOT" || exit 2
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST" >&2; exit 2; }
mkdir -p "$GOLDEN" "$LASTRUN"

# Executable directory: repo root unless FST_BIN points at a build tree
# (CMake: build-*/bin). Demos run with CWD=$ROOT either way,
# so data/ and tests/ relative paths in commands keep working.
BIN=${FST_BIN:-.}

MODE=${1:-}
case "$MODE" in capture|check|check-values|time) ;; *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;; esac
shift

# --- helpers ---------------------------------------------------------------

# Normalization applied to all captured output before storing/comparing:
#  - drop CR (Windows line endings)
#  - mask elapsed-time reports (the only timing-dependent text demos emit)
normalize() {
  sed -E -e 's/\r$//' \
         -e 's/StopWatch time: [0-9:]+[[:space:]]+\(equals [0-9]+ seconds\)/StopWatch time: <ELAPSED>/g' \
  | if [ -n "$LASTRUN_WIN_RE" ]; then
      sed -e "s|$LASTRUN_RE|$LASTRUN_CANON|g" -e "s|$LASTRUN_WIN_RE|$LASTRUN_CANON|g"
    else
      sed -e "s|$LASTRUN_RE|$LASTRUN_CANON|g"
    fi
}

# Result-signature extraction for demos whose full log is not run-stable
# (thread interleaving): keep only the final-result lines.
result_sig() {
  grep -E '^(Search result:|Subset<|Criterion value|Validated )' || true
}

# Print-precision canonicalization for `check-values`. Rewrites every
# FLOATING-POINT token to a fixed number of
# significant digits so that low-order libm/codegen differences between
# platforms compare equal, while everything else — feature indices, subset
# cardinalities, counts — stays byte-exact. Only tokens containing a decimal
# point (optionally with an exponent) are rounded, so bare integers are never
# touched and a changed subset still fails loudly.
#
# Deliberately applied to the WHOLE normalized output, not to `result_sig` lines:
# most goldens are the CLI's JSON output, which `result_sig`'s demo-era text
# patterns (`Search result:` / `Subset<` / ...) do not match at all — filtering
# first would silently compare nothing on most of the suite.
# Rounding the whole stream is format-agnostic and works for JSON and text alike.
#
# Caveat, accepted: a dotted token like a "4.10.0" version string has its first
# two components rounded as if numeric ("4.1" + ".0"). Harmless here because the
# SAME transform is applied to both sides before diffing, so a real difference
# still differs; it could in principle mask a version change that collides after
# rounding. This is a cross-platform smoke, not a byte-exact baseline.
VALUE_PRECISION=${FST_VALUE_PRECISION:-6}
values_sig() {
  awk -v p="$VALUE_PRECISION" '
    {
      out = ""; rest = $0
      while (match(rest, /-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?/)) {
        out = out substr(rest, 1, RSTART-1) sprintf("%.*g", p, substr(rest, RSTART, RLENGTH) + 0)
        rest = substr(rest, RSTART + RLENGTH)
      }
      print out rest
    }'
}

manifest_entries() {
  # prints: name<TAB>mode<TAB>timeout
  grep -Ev '^[[:space:]]*(#|$)' "$MANIFEST"
}

lookup() { # $1=demo -> sets M_MODE M_TIMEOUT M_CMD, returns 1 if not found
  line=$(manifest_entries | awk -v d="$1" '$1==d{print; exit}')
  [ -n "$line" ] || return 1
  M_MODE=$(echo "$line" | awk '{print $2}')
  M_TIMEOUT=$(echo "$line" | awk '{print $3}')
  # optional 5th+ column: full command line to run instead of ./<name>
  # (used by fst4 CLI scenarios — see the manifest header); strip trailing comment
  M_CMD=$(echo "$line" | awk '{cmd=""; for(i=5;i<=NF;i++) {if($i=="#") break; cmd=cmd (cmd==""?"":" ") $i}; print cmd}')
}

run_one() { # $1=name $2=timeout -> writes $LASTRUN/$1.raw, sets R_STATUS R_SECS
  if [ -n "$M_CMD" ]; then
    # manifest commands invoke ./fst4 — redirect to $BIN when overridden
    cmd=$M_CMD
    [ "$BIN" != "." ] && cmd=$(printf '%s' "$M_CMD" | sed "s|^\./fst4|$BIN/fst4|")
    start=$(date +%s)
    run_bounded "$2" sh -c "$cmd" >"$LASTRUN/$1.raw" 2>&1
    R_STATUS=$?
    R_SECS=$(( $(date +%s) - start ))
    return
  fi
  exe="$BIN/$1"
  [ -x "$BIN/$1.exe" ] && exe="$BIN/$1.exe"
  [ -x "$exe" ] || { echo "  $1: executable not found (build it first)" >&2; R_STATUS=127; R_SECS=0; return; }
  start=$(date +%s)
  run_bounded "$2" "$exe" >"$LASTRUN/$1.raw" 2>&1
  R_STATUS=$?
  R_SECS=$(( $(date +%s) - start ))
}

# --- main loop -------------------------------------------------------------

# Default set: every non-skip demo, minus the slow tier unless FST_GOLDEN_SLOW=1.
# Explicitly named demos are always run regardless of tier.
if [ $# -gt 0 ]; then
  DEMOS="$*"
else
  DEMOS=$(manifest_entries | awk -v slow="${FST_GOLDEN_SLOW:-0}" '$2!="skip" && (slow=="1" || $4!="slow"){print $1}')
fi

# Eigen-enabled builds mismatch the goldens BY DESIGN (see the header). There is no
# runtime marker for it — FST_USE_EIGEN is compile-time only — so probe the build
# tree's CMakeCache. Only meaningful when FST_BIN points at a CMake tree; with the
# default BIN=. there is no cache to consult and nothing is claimed.
if [ "$BIN" != "." ] && [ -f "$BIN/../CMakeCache.txt" ] &&
   grep -q '^FST4_USE_EIGEN:BOOL=ON' "$BIN/../CMakeCache.txt" 2>/dev/null; then
  echo "WARNING: the build under test was configured with FST4_USE_EIGEN=ON."
  echo "         Eigen replaces the LU/inverse/determinant backend, so criterion"
  echo "         values differ from the goldens in low-order FP bits BY DESIGN."
  echo "         Any mismatch below is EXPECTED and is NOT an FST4 regression."
  echo "         Gate with an Eigen-OFF build before concluding anything."
  echo
fi

# Native-optimized builds (-DFST4_NATIVE_OPT=ON) are machine-specific codegen and
# NOT gate-valid: goldens are captured under the fixed "golden" preset. Same
# CMakeCache probe as the Eigen warning above. The asymmetry is
# deliberate: a bad CHECK wastes one run (warn), a bad CAPTURE poisons the shared
# baseline for every future run on this platform (refuse; FST_GOLDEN_ALLOW_NATIVE=1
# overrides for forensic experiments that know what they are doing).
if [ "$BIN" != "." ] && [ -f "$BIN/../CMakeCache.txt" ] &&
   grep -q '^FST4_NATIVE_OPT:BOOL=ON' "$BIN/../CMakeCache.txt" 2>/dev/null; then
  if [ "$MODE" = "capture" ] && [ "${FST_GOLDEN_ALLOW_NATIVE:-0}" != "1" ]; then
    echo "ERROR: refusing to CAPTURE goldens from a -DFST4_NATIVE_OPT=ON build." >&2
    echo "       Baselines must come from the fixed golden preset (default build)," >&2
    echo "       or every machine with a different CPU mismatches them. Reconfigure" >&2
    echo "       without FST4_NATIVE_OPT, or set FST_GOLDEN_ALLOW_NATIVE=1 to force." >&2
    exit 2
  fi
  echo "WARNING: the build under test was configured with FST4_NATIVE_OPT=ON."
  echo "         Native codegen is machine-specific; goldens are captured under the"
  echo "         fixed golden preset, so mismatches below may be codegen, not bugs."
  echo
fi

fail=0; pass=0; skipped=0; optskip=0
: >"$LASTRUN/timings.tsv"

for d in $DEMOS; do
  if ! lookup "$d"; then echo "$d: not in manifest" >&2; fail=$((fail+1)); continue; fi
  if [ "$M_MODE" = "skip" ] && [ "$MODE" != "time" ]; then
    echo "SKIP  $d (manifest: skip)"; skipped=$((skipped+1)); continue
  fi

  run_one "$d" "$M_TIMEOUT"
  printf '%s\t%s\t%s\n' "$d" "$R_SECS" "$R_STATUS" >>"$LASTRUN/timings.tsv"

  if [ "$R_STATUS" -eq 124 ]; then
    echo "FAIL  $d (timeout after ${M_TIMEOUT}s)"; fail=$((fail+1)); continue
  elif [ "$R_STATUS" -ne 0 ]; then
    # LibSVM is an OPTIONAL dependency (README third-party policy): a build configured
    # -DFST4_WITH_SVM=OFF, or one where libsvm/ was never populated, cannot run the
    # wrapper-svm scenarios. The criterion factory refuses at startup with its own
    # message, so detect that rather than marking scenarios in the manifest — a new
    # wrapper-svm scenario is then covered automatically. A missing optional dependency
    # is NOT a regression: it means the scenario could not be run at all, which the
    # summary reports separately from both passes and manifest skips.
    # 2>/dev/null: the .raw is absent when the executable itself was not found.
    if grep -q 'no LibSVM support' "$LASTRUN/$d.raw" 2>/dev/null; then
      echo "SKIP  $d (NOT RUNNABLE: this build has no LibSVM support)"
      optskip=$((optskip+1)); continue
    fi
    echo "FAIL  $d (exit code $R_STATUS after ${R_SECS}s)"
    tail -3 "$LASTRUN/$d.raw" | sed 's/^/      /'
    fail=$((fail+1)); continue
  fi

  normalize <"$LASTRUN/$d.raw" >"$LASTRUN/$d.out"

  case "$MODE" in
    time)
      echo "OK    $d (${R_SECS}s, exit 0)"; pass=$((pass+1)) ;;
    capture)
      case "$M_MODE" in
        exact)  cp "$LASTRUN/$d.out" "$GOLDEN/$d.out" ;;
        result) result_sig <"$LASTRUN/$d.out" >"$GOLDEN/$d.result" ;;
        smoke)  ;; # nothing to record; validity is checked at check time
      esac
      echo "CAP   $d (${R_SECS}s, mode=$M_MODE)"; pass=$((pass+1)) ;;
    check)
      if [ "$M_MODE" = "smoke" ]; then
        if grep -q 'Criterion value' "$LASTRUN/$d.out"; then
          echo "PASS  $d (${R_SECS}s, smoke: exit 0 + result present)"; pass=$((pass+1))
        else
          echo "FAIL  $d (smoke: exit 0 but no 'Criterion value' in output)"; fail=$((fail+1))
        fi
        continue
      fi
      case "$M_MODE" in
        exact)
          ref="$GOLDEN/$d.out"; cur="$LASTRUN/$d.out" ;;
        result)
          ref="$GOLDEN/$d.result"; cur="$LASTRUN/$d.result"
          result_sig <"$LASTRUN/$d.out" >"$cur" ;;
      esac
      if [ ! -f "$ref" ]; then
        echo "FAIL  $d (no golden recorded — run capture first)"; fail=$((fail+1))
      else
        # goldens are stored with LF, but a fresh checkout with core.autocrlf=true
        # rewrites them to CRLF (see .gitattributes); compare CR-insensitively
        refn="$LASTRUN/$d.golden"
        tr -d '\r' <"$ref" >"$refn"
        if diff -q "$refn" "$cur" >/dev/null 2>&1; then
          echo "PASS  $d (${R_SECS}s)"; pass=$((pass+1))
        else
          echo "FAIL  $d (output differs from golden, ${R_SECS}s)"
          diff "$refn" "$cur" | head -12 | sed 's/^/      /'
          fail=$((fail+1))
        fi
      fi ;;
    check-values)
      # Cross-platform smoke: same comparison `check` makes, but with every
      # floating-point token rounded to $VALUE_PRECISION significant digits, so
      # low-order libm/codegen differences between platforms compare equal.
      # smoke-mode scenarios have no recorded values, so they fall back to the
      # same validity check `check` applies.
      if [ "$M_MODE" = "smoke" ]; then
        if grep -q 'Criterion value' "$LASTRUN/$d.out"; then
          echo "PASS  $d (${R_SECS}s, smoke: exit 0 + result present)"; pass=$((pass+1))
        else
          echo "FAIL  $d (smoke: exit 0 but no 'Criterion value' in output)"; fail=$((fail+1))
        fi
        continue
      fi
      case "$M_MODE" in
        exact)  ref="$GOLDEN/$d.out" ;;
        result) ref="$GOLDEN/$d.result" ;;
      esac
      if [ ! -f "$ref" ]; then
        echo "FAIL  $d (no golden recorded — run capture first)"; fail=$((fail+1))
      else
        refv="$LASTRUN/$d.golden.values"; curv="$LASTRUN/$d.values"
        tr -d '\r' <"$ref" | values_sig >"$refv"
        case "$M_MODE" in
          exact)  values_sig <"$LASTRUN/$d.out" >"$curv" ;;
          result) result_sig <"$LASTRUN/$d.out" | values_sig >"$curv" ;;
        esac
        if [ ! -s "$refv" ]; then
          echo "FAIL  $d (golden yielded no result lines to compare)"; fail=$((fail+1))
        elif diff -q "$refv" "$curv" >/dev/null 2>&1; then
          echo "PASS  $d (${R_SECS}s, values @ ${VALUE_PRECISION} sig digits)"; pass=$((pass+1))
        else
          echo "FAIL  $d (result values differ at ${VALUE_PRECISION} sig digits, ${R_SECS}s)"
          diff "$refv" "$curv" | head -12 | sed 's/^/      /'
          fail=$((fail+1))
        fi
      fi ;;
  esac
done

echo
summary="$MODE summary: $pass ok, $fail failed, $skipped skipped"
if [ "$optskip" -gt 0 ]; then
  summary="$summary, $optskip NOT RUNNABLE"
fi
echo "$summary"
if [ "$optskip" -gt 0 ]; then
  echo "  ($optskip scenario(s) could not run because the OPTIONAL LibSVM dependency is"
  echo "   absent from this build. That is not an FST4 failure — those scenarios were"
  echo "   not exercised at all, so this run covers less than a full green run does.)"
fi
[ "$fail" -eq 0 ] || exit 1
exit 0
