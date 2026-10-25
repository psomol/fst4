# Contributing to FST4

Thank you for your interest in FST4. This page covers the practical side of
contributing: how to report problems, what a pull request must bring along,
and the rules the code base follows. The legal side, the Contributor License
Agreement, is explained in [LICENSING.md](LICENSING.md).

## Before you start: the CLA

**Every contribution needs a signed Contributor License Agreement (CLA)
before it can be merged.** FST4 is dual-licensed (AGPL-3.0 plus a commercial
option from ÚTIA); without the CLA your contribution could not be part of the
commercial option, and the model would break down over time. You keep your
copyright; the CLA grants ÚTIA the right to relicense your contribution.
[LICENSING.md](LICENSING.md) explains the model.

The agreement text is ÚTIA's to issue. It will be published in this
repository, together with the signing procedure, before the first external
contribution is merged. Until then you are welcome to open issues, and to
open pull requests for discussion, but no pull request can be merged. If you
contribute on behalf of your employer, make sure you are entitled to do so.

Code written with the help of AI assistants is welcome under the same terms:
you are the contributor, you are responsible for the contribution, and you
must have the right to license it.

## Getting in touch

- **Bugs and feature requests:** open an issue on GitHub.
- **Questions, ideas, anything else technical:** somol@utia.cas.cz, with
  "FST" anywhere in the subject line (it routes the message to the right
  place and gets you a faster answer).
- **Licensing** (commercial license, CLA questions): utia@utia.cas.cz.
- **Security problems:** see [SECURITY.md](SECURITY.md); please do not open a
  public issue for them.

## Reporting a bug

A useful report contains:

- the complete command line, or the config file when you use `--config`;
- the data format and, if at all possible, a minimal data file that
  reproduces the problem (or the name of one of the bundled `data/` files);
- your platform and compiler, and whether the build was Release or Debug;
- the beginning of fst4's output: the setup echo states the version and
  every effective option, and `--seed` makes randomized runs reproducible.

## Pull requests

1. For anything beyond a small fix, open an issue first and describe what you
   intend to change; it saves work on both sides.
2. Fork the repository, work on a branch, one topic per pull request.
3. Build **both** configurations with CMake: `-DCMAKE_BUILD_TYPE=Release` and
   `-DCMAKE_BUILD_TYPE=Debug`. The Debug build enables the assertions and
   `-Wall`; it must stay warning-free (the vendored CLI11 is the only
   exception).
4. Run the golden gate on your Release build:

   ```bash
   FST_BIN=build/bin sh tests/golden.sh check
   ```

   Every scenario must pass on your platform (baselines exist for Windows
   x86-64, macOS arm64 and Linux arm64; on another platform, capture your own
   baseline first, see [tests/README.md](tests/README.md)). If your change
   *intentionally* alters output (a new output line, changed numerics),
   recapture the affected scenarios with `sh tests/golden.sh capture
   <scenario>` and explain every moved golden in the pull request; the
   maintainers recapture the other platforms. Never edit golden files by
   hand. Before a release the slow tier runs too
   (`FST_GOLDEN_SLOW=1`).
5. A new `fst4` flag comes with its `--help` text, its entry in
   [doc/FST4_CLI.md](doc/FST4_CLI.md) and at least one golden scenario in
   `tests/manifest.tsv` (`--stable-output --verbosity silent` keeps the output
   byte-stable).
6. A new component (criterion, classifier, search, splitter, tracker) is
   registered in the fst4 factories so that it is reachable from the command
   line, documented with Doxygen comments (the reference must build
   warning-free: `doxygen doxygen.cfg` run from `doc/`), and covered by a
   golden scenario; a `demos/*.cfg` config when it demonstrates a protocol.
7. Code, goldens, documentation and data belong in the same pull request.

## Coding rules

- **C++17, header-only library.** New library code goes into a header under
  the matching `_src_*` directory and into `namespace FST`; the command-line
  tool's own types live in `namespace fst4_cli` in `fst4.cpp`.
- **Randomness** goes through `FST::fst_rand()`, `fst_srand()` and
  `fst_rand_below()` only, never `rand()` or `<random>`: seed-for-seed
  reproducibility across platforms depends on it. A component that needs a
  private generator seeds it deterministically from its inputs.
- **Threading.** Output from code that runs in worker threads goes through
  `syncout::print`. Every component that threaded searches clone derives
  from `Clonable_Impl`; clone bodies are never hand-written. Threaded results
  must be bit-identical to sequential at any thread count: candidates are
  generated on the master thread and results are replayed in candidate order
  (see `candidate_evaluator_threaded.hpp`).
- **Search progress output** goes through the `Search_Observer` helpers
  `report()` and `report_solution()`, never through bare stream code in a
  `search_*.hpp`.
- **Determinism.** Ties break by index order, nothing depends on the
  iteration order of unordered containers, and no result depends on time.
- **Naming and style.** Template parameters keep the established names
  (`DATATYPE`, `REALTYPE`, `IDXTYPE`, `DIMTYPE`, `RETURNTYPE`, `SUBSET`,
  `CRITERION`, ...); classes are `Class_Name`, methods `method_name`;
  indentation uses tabs; every public class and method carries a Doxygen
  comment.
- **Comments describe what the code does, in the present tense.** History
  (dates, review numbers, why something was changed) belongs in the commit
  message and the pull request, not in the sources.
- **The copyright and license header block** is identical in every source
  file. Do not edit it in one file; it changes by script across all files.
- **No new third-party code or data** without an explicit license review.
  Sample data needs its attribution in `data/NOTICE`.
