# FST4 Golden-Output Regression Harness

This is the release gate of FST4: every change is gated on this harness staying
green. It answers one question with zero judgment required: **did the change alter
any scenario's observable behavior?**

## Usage

Run from anywhere (scripts locate the repo root themselves); Git Bash on Windows,
any POSIX sh on macOS/Linux:

```bash
sh tests/golden.sh check              # the gate: fast tier (~10 min), diff vs goldens
FST_GOLDEN_SLOW=1 sh tests/golden.sh check   # full gate incl. slow tier (run before a release)
sh tests/golden.sh check demo10 demo26   # subset only (named demos run even if slow-tier)
sh tests/golden.sh capture demo10     # re-record golden after an INTENDED behavior change
sh tests/golden.sh time               # timing/exit-code survey only, no comparison
FST_BIN=build-win/bin sh tests/golden.sh check   # gate a CMake build tree (build-mac/bin on macOS, build-linux/bin on Linux); default is the repo root
```

The fast/default tier (106 scenarios) is what you run after every work item. The
`slow` tier (6 scenarios — demo52t, fst4_bifos_demo33, fst4_ensemble_demo50,
fst4_exhaustive_demo40, fst4_margin_demo60, fst4_search2_demo25; minutes to
half an hour each) must additionally be green before a release. 112 scenarios in
total.

Exit code 0 = all pass. Any failure prints the first lines of the diff and exits 1.
Raw and normalized outputs of the last run stay in `tests/last_run/` (gitignored)
for inspection; goldens live in per-platform baseline dirs (tracked) —
`tests/golden-win-x64/` on native Windows, `tests/golden-macos-arm64/` on Apple
Silicon, `tests/golden-linux-arm64/` on arm64 Linux, auto-selected by golden.sh
from `uname` (`FST_GOLDEN_DIR` overrides; a platform without a baseline of its
own captures one and points there). All three tracked baselines are complete at
112 scenarios and each is verified green on its own platform's binaries, slow
tier included.

Set `FST_LASTRUN=<dir>` to move the scratch directory when two gate runs share
one checkout (a container and its host, say) — without it they clobber each
other's intermediates. Output is normalized back to the canonical path, so the
goldens compare byte-identically wherever the scratch directory lives.

Executables must be built first (see the build quickstart in `README.md`).
CMake is the only build system, and build trees are per-platform by
convention, so gate the tree you built:
`cmake --build build-mac --parallel --target demos_all` then
`FST_BIN=build-mac/bin` (`build-win` on Windows, `build-linux` on Linux). The
`demos_all` target also builds the non-demo harness entries: the two smoke
tests (`smoke_tests` group) and the `fst4` command-line tool. The `FST_BIN`
directory applies to the manifest's fst4 command lines too (their leading
`./fst4` is rewritten). On Windows the script prepends `C:\mingw64\bin` to PATH
itself so the correct runtime DLLs are found.

## fst4 CLI scenarios

Manifest entries whose 5th column carries a command line (names `fst4_*`) run
that command instead of `./<name>`; everything else (normalization, modes,
tiers) works identically. The scenarios pin the CLI surface: every splitter,
scaler, criterion family, search algorithm, all output-format paths
(JSON/CSV/text), the data readers, the seeded OS initialization, and threaded
runs. All commands use `--stable-output` (strips the timing/stats block) and
`--verbosity silent` (suppresses library progress prose), which makes `exact`
mode applicable — byte-stability was established by repeated runs on identical
binaries, threaded scenarios included. `fst4_demo10` is the demo-parity anchor:
it reproduces demo10's selected subset and criterion value exactly.

**Purpose-built fixtures live in `tests/data/`** (tracked), as opposed to the
real datasets in `data/`: they are hand-written minimal files that pin a
specific loader or parser contract, and they are small enough to read and
reason about in full. `mixedtypes_dense.arff` / `mixedtypes_sparse.arff` carry
unsupported attribute types (STRING, DATE) in first/middle/last positions plus
quoted values containing commas, pinning the ARFF loader's graceful
column-skipping (scenarios `fst4_arff_skiptypes_dense` / `_sparse`). Both
fixtures encode the same data as a hand-cleaned file with those columns
physically removed, so their goldens' criterion value is exactly what the
cleaned file produces — if a future change breaks the column remapping, the
value moves. `csv_basic.csv` / `csv_semicolon.csv` carry the same 20 records in
two CSV dialects (scenarios `fst4_csv_basic` / `_semicolon` must report
identical result lines).

## What is compared, exactly

Captured output is stdout+stderr merged, normalized by:

1. dropping CR characters (Windows line endings),
2. masking elapsed-time reports: `StopWatch time: 0:00:01:26  (equals 86 seconds)`
   → `StopWatch time: <ELAPSED>` (the only timing-dependent text demos emit),
3. rewriting the scratch-directory path back to the canonical `tests/last_run`
   token in the two scenarios that write side files.

Per-demo comparison modes (third column of `tests/manifest.tsv`):

| Mode | Meaning | Used when |
|---|---|---|
| `exact` | normalized output must byte-match `<baseline dir>/<demo>.out` | output proved run-stable |
| `result` | only the final result signature must match `<baseline dir>/<demo>.result` — lines starting `Search result:`, `Subset<`, `Criterion value`, `Validated ` | full log is not run-stable (thread interleaving) but results are |
| `smoke` | exit 0 within timeout and ≥1 `Criterion value` line in output; no golden | results are load-dependent by design (wall-clock-budgeted searches) |
| `skip` | not run | run time is hours by design |

> **Two `golden.sh` invocations must not share one scratch directory.** Both
> would write `tests/last_run/` — the second truncates `timings.tsv` and
> overwrites `<scenario>.raw`/`.out` under the first, so two runs touching the
> same scenario at the same moment can produce a spurious failure. Point the
> second run at its own scratch directory with `FST_LASTRUN=<dir>` (above), or
> gate one build at a time.

A timeout or a nonzero exit code is always a failure (demos exit 1 on fatal
errors), with ONE exception: a `wrapper-svm` scenario run against a binary
built without the optional LibSVM dependency (`-DFST4_WITH_SVM=OFF`, or
`libsvm/` never populated) is reported `SKIP`, not `FAIL`. `golden.sh`
recognizes it from the binary's own startup refusal, so no manifest bookkeeping
is needed and new `wrapper-svm` scenarios are covered automatically.

### The two optional external dependencies behave OPPOSITELY

Neither one's absence is an FST4 defect, but they need opposite handling — worth
stating explicitly, because the gate is used to judge whether the codebase itself
is healthy.

- **LibSVM absent → 7 scenarios CANNOT RUN.** All of them request it through
  `--criterion wrapper-svm`, so all 7 route through the one criterion-factory
  refusal that `golden.sh` detects: `fst4_svmalt_demo23`, `fst4_hybrid_demo53`,
  `fst4_hybrid_threads`, `fst4_svmpre_demo12`, `fst4_search2_demo25`,
  `fst4_missing_demo62`, `fst4_metric_aucmacro_svm`. They are counted and
  reported SEPARATELY (`N NOT RUNNABLE`) rather than folded into the skip
  count, because **such a run is NOT full coverage** and must not be mistaken
  for a fully-green one. Exit code is unaffected.
  Two scenarios look SVM-dependent but are not: `fst4_report_tuned` uses
  `--criterion wrapper-rbfsvm` and `fst4_rbfsvm_opt` pins `--rbfsvm-optimize`,
  both the native RBF SVM. No compiled demo needs LibSVM either.
  Caveat worth knowing: the detection matches the wording of the `fst_error`
  thrown in `fst4.cpp`. Reword that message and these scenarios silently revert
  to reporting `FAIL`.
- **Eigen absent → nothing happens; NOTHING here depends on it.**
  `FST4_USE_EIGEN` defaults OFF, the goldens were captured with the hand-rolled
  LU path, and CMake hard-errors at configure time if Eigen is requested without
  a valid `FST4_EIGEN_DIR`. The hazard is the REVERSE: an Eigen-**enabled**
  build swaps `Indexed_Matrix`'s LU/inverse/determinant and shifts low-order FP
  bits *by design*, so normal-model scenarios legitimately mismatch. Those are
  not regressions. Since `FST_USE_EIGEN` is compile-time only with no runtime
  marker, `golden.sh` probes `$FST_BIN/../CMakeCache.txt` and prints a warning
  banner before running anything.

## Why each scenario is classified the way it is

> Most FST3 demos live on as `fst4_*` scenarios (their protocols recast as
> command lines of the fst4 tool), which inherited the exact/result/smoke/skip
> classifications of the compiled demos they replaced. The rationale below
> governs those scenarios: the determinism facts are properties of the library,
> not of the demo executables.

Determinism facts:

- **"Random" splits are deterministic.** `Data_Splitter_RandomRandom` seeds from
  `time()` only when constructed with `randomize=true`; the demos and the fst4
  tool never do, so the splits are drawn from FST4's own generator
  (`FST::fst_rand()`, unseeded = `--seed 1`), which yields the identical sequence
  on every run and on every platform. The same holds for the derived randfix
  splitter.
- **Monte Carlo candidate generation is deterministic**, in sequence AND in
  assignment: candidates are drawn on the master thread from the `fst_rand`
  stream and their results are consumed in generation order, so threaded runs
  are bit-identical to sequential ones at the same seed.
- **demo24 / demo24t are wall-clock-budgeted** (`--mc-seconds`): the trial
  count, and therefore the result, depends on machine speed and load. These can
  never be golden-compared → `smoke` mode, and they have no scenario here.
- **demo34 runs for 6 hours by design** (a 6-hour Monte Carlo budget), and
  **demo11t, demo32t, demo35t are compute-bound for hours** on
  `reuters_apte.arff` (healthy, not hung) → no scenario. demo35t additionally
  has a 5-minute DAF probe budget after its hours-long SVM grid search, so it
  would be load-dependent even if it were fast.
- **demo36 / demo37t (the sampled evaluator)**: demo36 runs for hours (gisette,
  d=300) and has no scenario; the sampled path itself is pinned by
  `fst4_sampled`, `fst4_sampled_cold` and `fst4_sampled_threads`, the last of
  them pinning the bit-identical-at-any-thread-count guarantee.
- Threaded scenarios may interleave progress lines nondeterministically even
  when their results are deterministic; any scenario whose full log differs
  between two identical-binary runs is demoted from `exact` to `result` mode.
  The fst4 CLI scenarios avoid the issue entirely by running with `--verbosity
  silent`, and the threaded engines replay results in candidate order, so the
  compiled threaded exemplar demo52t is byte-stable too.

## Portability of the goldens

Golden capture and check are tied to the fixed "golden" build preset — the
DEFAULT CMake build (`-O2 -funroll-loops -ffp-contract=off`, identical on every
architecture, no `-march=native`). Within one platform+arch, golden-preset
baselines are machine-independent; `-DFST4_NATIVE_OPT=ON` builds are
machine-specific, and golden.sh REFUSES to capture from them (and warns on
check) via a CMakeCache probe. Baselines are per-platform: `golden-win-x64/`,
`golden-macos-arm64/` and `golden-linux-arm64/` ship complete at 112
scenarios, captured under the golden preset; `golden-linux-x64/` is
auto-selected but not shipped, so an x86-64 Linux user captures it on first
run.

Measured cross-platform agreement: Linux arm64 and macOS arm64 are
byte-for-byte identical on 111 of the 112 scenarios, and both differ from
Windows x64 on two (`fst4_report_multiclass`, `fst4_report_tuned`) — in every
case only in low-order digits of a `--report detailed` ROC/PR line, where an
iterative solve printed at ~12 significant digits reaches a marginally
different fixed point per architecture. Selected subsets, criterion values and
every other output line agree everywhere. The `check-values` mode of golden.sh
(result lines compared at print precision) exists for exactly this
cross-platform situation. The per-platform split is kept as insurance for
platforms not yet measured (musl, other glibc versions, x86-64 Linux), not
because of observed divergence.

Two hazards when moving between machines:

1. **Native-optimized executables are not portable.** `-march=native`
   binaries (`-DFST4_NATIVE_OPT=ON`) from a newer CPU die instantly with
   `Illegal instruction` (exit 132) on a CPU lacking those instruction-set
   extensions. This fails loudly, never silently. Default (golden-preset)
   builds do not have this problem.
2. **Synced build artifacts defeat staleness detection.** A build tree carried
   over by a file-synchronization service keeps its original mtimes, so an
   incremental build on the other machine can silently keep foreign-CPU
   objects and binaries. Always run a clean rebuild first when the tree was
   last built on another machine.

Slow-tier timeouts in the manifest are sized to the slowest machine in regular
use.

## Workflow rules

0. **Make sure the binary you check is fresh.** A green `check` against a stale
   binary proves nothing. CMake tracks header dependencies exactly per
   translation unit, so incremental builds are trustworthy; still prefer a
   clean rebuild before capturing new goldens. (The harness was mutation-tested:
   a +0.001 perturbation in the GMahalanobis criterion fails the demo10 gate.)
1. `check` must be green **before** you start (establishes the baseline is intact).
2. `check` must be green **after** every work item; investigate any diff — either
   the change is a bug, or the behavior change is intended and must be explained
   in the commit message together with the re-captured golden.
3. Never re-capture a golden to make a failure go away without understanding the
   diff. The diff IS the finding.
4. New CLI functionality gets its own goldens here: add an entry to the
   manifest and a `capture` run in the same commit that adds the feature.
