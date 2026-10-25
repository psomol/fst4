# Feature Selection Toolbox 4 (FST4)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01502-b31b1b.svg)](https://arxiv.org/abs/2608.01502)

*The arXiv badge links the paper on the budgeted stochastic sequential search,
a new principal component of this version, not a description of the toolbox as
a whole; see [Citing FST4](#citing-fst4).*

Feature Selection Toolbox 4 (FST4) is a standalone C++17 library and
command-line tool for **feature selection** (also known as attribute or
variable selection): it reduces problem dimensionality to maximize the
accuracy of data models and automatic decision rules and to reduce data
acquisition cost. This is FST4 version 4.2, the first published FST4 release;
the last public release before it was FST3 (see
[Version history](#version-history)).

More generally, FST4 is a **subset optimizer**. Every search in it maximizes a
user-supplied set function over a finite universe of candidates, so the same
algorithms that pick features can pick sensors, experiments, portfolio
components or any other subset for which a candidate can be scored. The scorer
can be a built-in criterion, a C++ class of your own, or any external program
(`--criterion extern-value`, with or without a data file). The sequential,
floating and stochastic heuristics return near-optimal subsets at thousands of
candidates; Branch & Bound returns the provably optimal subset when the
criterion is monotone (it never decreases when a candidate is added).

- **Filters, wrappers and optimal search in one toolbox.** Normal-model and
  multinomial-model filter criteria (Bhattacharyya distance, divergence,
  generalized Mahalanobis distance, mutual information); wrapper criteria over
  built-in deterministic classifiers and regressors (k-NN, normal Bayes, LDA,
  logistic regression, Random Forest, linear and RBF-kernel SVM, ridge and k-NN
  regression) scored by accuracy, balanced accuracy, F1, AUC, R² or
  cost-weighted metrics; bridges to external classifiers and external
  criteria. Optimal Branch & Bound (four variants) and exhaustive search;
  sequential, floating, oscillating, retreating, Monte Carlo and budgeted
  stochastic search; result trackers for stability evaluation, process
  similarity and anti-overfitting regularization.
- **Deterministic by contract.** Seeded runs reproduce seed-for-seed across
  Windows, macOS and Linux, threaded runs are bit-identical to sequential at
  any thread count, and every release is gated on byte-exact outputs of 112
  reference scenarios per platform.
- **No C++ required.** The `fst4` command-line tool exposes the whole library
  through flags and config files, with text, JSON and CSV output, and ships
  with 51 ready-to-run demo configurations. The library is header-only with
  no external dependencies (LibSVM and Eigen are optional extensions).

## Quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target fst4
build/bin/fst4 --data data/wdbc_30.trn --criterion gmahalanobis --search sfs --target-d 7 --validate knn
build/bin/fst4 --config demos/demo10.cfg
```

Run `fst4` from the repository root so that the relative `data/` paths
resolve. `build/bin/fst4 --help` lists every flag, `--list` prints the
component registry with validity notes.

## Documentation

- [Introduction to FST4](https://psomol.github.io/fst4/FST4_INTRO.html): a
  self-contained tour of what FST4 does and how to use it. The source is
  [FST4_INTRO.html](FST4_INTRO.html) in this repository; GitHub shows the file
  as HTML source, the link above renders it.
- [fst4 command-line reference](doc/FST4_CLI.md): every flag, the output
  formats, the config-file syntax, Unix pipe usage, the demo parity configs.
- [API reference](https://fst.utia.cz/doc/): the Doxygen reference of the C++
  library, on the FST web site. Regenerate it locally with `doxygen
  doxygen.cfg` run from `doc/` (output `doc/html`).
- [demos/](demos/): 51 ready-to-run config files, one per published FST3 demo
  plus the FST4-native additions, each with a header comment stating the demo
  title, the protocol and any runtime warnings; and the three commented C++
  exemplars [demo10.cpp](demos/demo10.cpp) (basic filter-based selection),
  [demo52t.cpp](demos/demo52t.cpp) (threaded wrapper criterion with result
  regularization) and [demo55.cpp](demos/demo55.cpp) (selection stability
  and similarity evaluation). The [Demos page](https://fst.utia.cz/demos.html)
  of the web site browses the configs.
- [RELEASE_NOTES.md](RELEASE_NOTES.md): the complete record of what changed,
  one detailed section per version.
- [tests/README.md](tests/README.md): the golden-output regression harness,
  which doubles as the build verification (see [Building](#building)).
- [fst.utia.cz](https://fst.utia.cz): the FST web site, with background on
  the feature selection problem, references and the legacy FST3 and FST1
  releases.

## Citing FST4

If FST4 contributes to your research, please cite the toolbox paper:

* Petr Somol and Pavel Pudil. Feature Selection Toolbox. Pattern
  Recognition, 35(12):2749-2759, 2002.
  [doi:10.1016/S0031-3203(01)00245-X](https://doi.org/10.1016/S0031-3203(01)00245-X)

The budgeted stochastic (sampled) sequential search (`--evaluator sampled`,
demos `demo36.cfg` and `demo37t.cfg`) is a new principal component of FST4 and
one of the main contributions of this release: it bounds the evaluation cost
of every sequential step by a candidate budget independent of the
dimensionality, which is what makes floating search feasible at thousands of
features. The following paper describes this component, not the toolbox as a
whole:

* Petr Somol and Jiri Grim. Stochastic Sequential Search in
  Very-High-Dimensional Feature Selection. arXiv:2608.01502 [cs.LG], 2026.
  https://arxiv.org/abs/2608.01502

[CITATION.cff](CITATION.cff) carries the same references in machine-readable
form ("Cite this repository" in the sidebar).

## License

FST4 is **dual-licensed**. You may use it under either of the following:

* **Free software under the GNU Affero General Public License, version 3**
  (the full text is in [LICENSE](LICENSE)): anyone, including commercial
  users, may use, modify and redistribute it, provided that any modified
  version they distribute or make available over a network is released under
  the same license with its complete source code.
* **A commercial license from ÚTIA** for organizations that want to use FST4
  in closed-source products without those obligations. In case of interest,
  write to utia@utia.cas.cz.

FST4 software (with the exception of the bundled third-party components and
the sample data sets listed below) is Copyright © 2000–present Institute of
Information Theory and Automation of the Czech Academy of Sciences (ÚTIA).
[LICENSING.md](LICENSING.md) explains the two options and the Contributor
License Agreement that every contribution requires. FST4 comes without
warranty of any kind, as stated in sections 15 and 16 of the
[LICENSE](LICENSE). Licensing questions go to utia@utia.cas.cz; technical
questions, bug reports and pull requests go to somol@utia.cas.cz, with "FST"
somewhere in the subject line (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Authorship

* FST4 builds on the human-written FST3 codebase. Version 4 was developed by
  Petr Somol with Anthropic Claude models as coding assistants: Claude Fable 5
  did most of the 2026 development work, with minor contributions by Claude
  Opus 5; Claude Fable 5.1 performed the full-codebase reviews in preparation
  for publication; and Claude Opus 4.5 performed the December 2025 audit of
  the FST3 codebase that preceded the FST4 program. All design decisions, the
  acceptance of every change and the verification through the byte-exact
  golden test suite are human work, and the copyright statement above covers
  the resulting work as a whole. Each source file's header repeats this
  attribution.

## Bundled third-party code

* The original FST3 public distribution contained no third-party code (users
  were instructed to download LibSVM themselves). FST4 keeps that policy for
  LibSVM and bundles one component; it remains under its own license, which
  applies to that component only and is retained in the component's directory:
  - CLI11 2.4.2 ([cli11/](cli11/)): Copyright (c) 2017-2024 University of
    Cincinnati, developed by Henry Schreiner under NSF AWARD 1414736,
    BSD-3-Clause license (see [cli11/LICENSE](cli11/LICENSE)). Used by the
    fst4 command-line tool only; the FST4 library itself has no CLI11
    dependency, so library-only use carries no CLI11 obligations. If you
    redistribute FST4 (source or binary), BSD-3 requires the CLI11 notice to
    travel along: keep `cli11/LICENSE` in source form, and reproduce the
    notice (this section or `cli11/LICENSE`) in materials accompanying any
    binary containing the fst4 tool.
* LibSVM is OPTIONAL (the toolbox is fully functional without it: the native
  `wrapper-linsvm` and `wrapper-rbfsvm` criteria cover both SVM kernels of
  practical value with no external library) and is NOT distributed with FST4
  (licensing policy). It is needed only for the LibSVM-backed SVM tools
  (`classifier_svm.hpp`): if you want those, fetch `svm.cpp` and `svm.h`
  (version 3.37, drop-in compatible) into `libsvm/` following the instructions
  in [libsvm/readme.txt](libsvm/readme.txt), including the `info()`
  log-silencing tweak (the SVM test scenarios assume it). LibSVM is Copyright
  (c) 2000-2023 Chih-Chung Chang and Chih-Jen Lin, BSD-3-Clause license.
* Eigen is likewise NOT distributed with FST4 and is not needed for any
  toolbox capability. It backs one OPTIONAL verification feature: building
  with `-DFST4_USE_EIGEN=ON -DFST4_EIGEN_DIR=<eigen-root>` swaps the internal
  matrix numerics (LU/inverse/determinant) to a user-supplied Eigen for
  cross-validation of results (see [Building](#building) and
  [RELEASE_NOTES.md](RELEASE_NOTES.md)). Such builds define
  `EIGEN_MPL2_ONLY`, so only Eigen's Mozilla Public License 2.0 core is used
  (the build fails if anything outside it were pulled in); MPL2 obligations
  attach to Eigen's own files, not to FST4 code. Default builds contain no
  Eigen code at all.

## Bundled sample data sets

* The [data/](data/) directory ships the sample data sets used by the
  exemplar demos, the `demos/*.cfg` parity configs and the golden test
  scenarios. They are third-party research data and are NOT covered by the
  FST4 copyright above: each carries its own terms. Full per-data-set
  attribution (creators, DOI, licence, and the modifications FST4 applied) is
  in [data/NOTICE](data/NOTICE), and each data file repeats its own origin
  and terms in its header (except the `.csv` files, whose format has no room
  for one; NOTICE covers them).
* Most originate at the UCI Machine Learning Repository and are licensed
  CC BY 4.0, which permits redistribution and adaptation for any purpose
  provided credit is given and modifications are indicated. FST4's copies ARE
  modified: every `.trn` file is a format conversion, `mushroom_125.trn` is
  binarized from the original 22 categorical attributes,
  `sonar_60_missing_data.trn` is an FST-made variant with 30% of the values
  marked missing, and the `.csv` files are CSV-format conversions of the
  corresponding `.trn` files (same values, same sample order), included so the
  demos can exercise the CSV reader. `data/NOTICE` records this per file.
* One data set is not under CC BY 4.0 and carries narrower terms:
  - `reuters_apte.arff`: Reuters-21578, Distribution 1.0. Copyright for the
    underlying newswire text and Reuters annotations resides with Reuters Ltd.;
    Reuters Ltd. and Carnegie Group, Inc. have agreed to allow free distribution
    of the collection for research purposes only. The FST4 file contains no
    article text: it is an anonymized sparse term-frequency matrix (attributes
    `a1..a10105`, classes `cls0..cls32`) from which no document can be
    reconstructed.
* If you redistribute FST4, these notices must travel with it: keep
  `data/NOTICE` and the data file headers intact (the CSV format has no
  comment syntax, so the `.csv` files carry no in-file header;
  `data/NOTICE` is their attribution record).

## Building

FST4 requires a C++17 conforming compiler and has no external library
dependencies. Unlike FST3 it needs no Boost: the C++ standard library
replaces it. LibSVM and Eigen are both OPTIONAL and not distributed with
FST4: LibSVM only for the LibSVM-backed SVM tools (fetch it into `libsvm/`
per [libsvm/readme.txt](libsvm/readme.txt)), Eigen only for the verification
build (`-DFST4_USE_EIGEN`, below); a default build needs neither. CLI11 is
bundled ([cli11/](cli11/)) and is used by the fst4 tool only. FST4 was
developed and tested with MinGW-W64 g++ 15.2 on Windows (x86-64), Apple
clang on macOS (arm64) and GCC 13 on Linux (arm64); any C++17 conforming
compiler is expected to work.

**CMake (any platform):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Executables are placed in `build/bin` (fst4, the exemplar demos, smoke
tests). Useful targets: `fst4`, `demos`, `smoke_tests`, `demos_all`.
Options: `-DFST4_WITH_SVM=OFF` skips the LibSVM-dependent targets and builds
an SVM-less fst4; `-DFST4_USE_EIGEN=ON -DFST4_EIGEN_DIR=<eigen-root>` swaps
the internal matrix numerics to Eigen (intended for verification builds, see
[RELEASE_NOTES.md](RELEASE_NOTES.md)). A `-DCMAKE_BUILD_TYPE=Debug` tree
enables the assertions and `-Wall`.

**Windows / Visual C++ without CMake:** create an Empty Project and manually
Add Existing header files (hpp) and source files (cpp). Add only one demo cpp
at a time (or `fst4.cpp`). Set the C++ Language Standard to C++17 or newer.
No external include or library paths are needed. Compiling
`_src_global/global.cpp` is no longer required (it remains as an empty
compatibility stub so that older project files keep working). For threaded
demos add `FST_THREADED` to Preprocessor Definitions under Configuration
Properties - C/C++ - Preprocessor; for a debug target add `DEBUG`, for a
release target `NDEBUG`. For SVM-based code add `libsvm/svm.cpp` (fetched per
[libsvm/readme.txt](libsvm/readme.txt)) to the project and `libsvm` to
Additional Include Directories. Visual C++ produces "unsafe/deprecated"
warnings related solely to the TRN and ARFF file-format readers because of
their old C-style input/output calls (fscanf, strtok etc.); defining
`_CRT_SECURE_NO_WARNINGS` switches them off without any negative consequence.
Some normal-model sources use `M_PI`, which Visual C++ may not define by
default; if so, add `_USE_MATH_DEFINES` to Preprocessor Definitions.

**Library use:** the library is header-only. Include the headers you need
(see the exemplar sources in [demos/](demos/)); nothing else needs to be
compiled.

**Build verification:** the golden-output test harness in [tests/](tests/)
recomputes a set of reference scenarios and compares the outputs byte-exactly
against the recorded reference outputs:

```bash
FST_BIN=build/bin sh tests/golden.sh check
```

The reference baselines are per-platform and auto-selected: baselines for
Windows x86-64 (`tests/golden-win-x64/`), macOS arm64
(`tests/golden-macos-arm64/`) and Linux arm64 (`tests/golden-linux-arm64/`)
are included; on any other platform set `FST_GOLDEN_DIR` to a baseline
directory of your own capture. See [tests/README.md](tests/README.md).

## Thread safety

The library as a whole is not thread safe. Only those portions of the library
that are used inside threaded feature selection routines are thread safe:
mostly the classes derived from `Clonable`, in particular `Criterion` and
`Data_Accessor` derived and related code.

## Known problems

- In rare occurrences the Divergence criterion on some data breaches monotonicity
  (adding a feature to a subset causes Divergence to decrease). This is undesirable
  in connection with Branch & Bound, which assumes monotonicity and may then miss
  the optimum. The reason is numerical instability: Divergence needs two matrix
  inversions per evaluation, and on ill-conditioned data the computed values lose
  accuracy. This is a property of the measure, not a defect that a code fix can
  remove, so the caution stands in 4.2. What did change: a normal-model criterion
  that arrives at a non-finite value (NaN or infinity) now aborts with a
  descriptive error instead of letting a search silently accept the garbage, and
  Bhattacharyya distance forms its covariance term in the log domain, so it no
  longer over- or underflows on badly scaled data (see RELEASE_NOTES.md). The
  recommendation therefore remains to prefer Bhattacharyya distance (one
  inversion, verified against an unrelated library and, in 4.0, against the
  Eigen library) whenever an optimal search is used.

  Remark: numerical problems are expectable with probabilistic criteria in general
  when the number of training samples in data is insufficient with respect to
  dimensionality. This is a theoretical problem - the only reasonable workaround is
  obtaining more training samples.

- LibSVM-backed SVM training/testing (wrapper-svm) freezes on some data. LINEAR
  and SIGMOID kernels seem to be particularly affected, RBF kernel is least
  affected. Evaluating individual features seems to be more prone to freeze than
  evaluating feature groups. Note that this problem could not be located in FST
  codes and is almost certainly the problem in LibSVM. Since 4.2 the native,
  dependency-free wrapper-linsvm and wrapper-rbfsvm criteria cover linear and
  RBF-kernel SVM selection without LibSVM and without this freeze; wrapper-svm
  remains for the FST3 parity configs, --svm-optimize and the poly/sigmoid
  kernels.

- Historical issue, believed resolved: textual output of time consuming multi-threaded
  searches could occasionally stall after a long period of time while the search itself
  continued correctly. Two root causes were found and fixed: thread-unsafe use of the
  shared output stream and iteration state (fixed in a December 2025 audit) and a
  lost-wakeup race in the threaded candidate evaluator (fixed in 4.0; see RELEASE_NOTES.md).
  Should output stalling nevertheless recur, the workaround remains to switch textual
  output off using set_output_detail(FST::SILENT); the output stream is then not used
  in the course of search, only at the end to report final results.

## Contributing and security

Bug reports and feature requests are welcome as issues. Pull requests are
welcome too; [CONTRIBUTING.md](CONTRIBUTING.md) describes what they must
bring along (the Contributor License Agreement, the golden gate, the coding
rules). Security problems go through [SECURITY.md](SECURITY.md), not through
public issues.

## Version history

FST4 4.2 is the first published FST4 version; 4.0 and 4.1 were internal
development milestones that never shipped separately, so a user of FST3
receives all three steps at once. [RELEASE_NOTES.md](RELEASE_NOTES.md) keeps
one detailed section per version.

4.2.0 (the first public FST4 release)
- wrapper criterion metrics beyond accuracy: balanced accuracy, F1, F1-macro,
  AUC (two-class exact, multiclass one-vs-rest macro), per-class cost weights;
  R2/RMSE/MAE for the regression wrapper (--wrapper-metric, --class-weights,
  --f1-positive)
- five new deterministic, dependency-free wrapper models: Random Forest,
  linear SVM, RBF kernel SVM (with native C/gamma optimization), ridge
  regression, linear discriminant analysis
- external-classifier bridge (wrapper-extern) and external criterion
  (extern-value), up to pure set-function optimization without any data file
  (--features N)
- LibSVM demoted to an optional extension: the native SVM wrappers cover both
  practical kernels; LibSVM is no longer distributed (licensing policy) and
  is fetched by the user when the LibSVM-backed tools are wanted
- research-grade reporting: --report detailed (criterion-vs-size curve with
  holdout re-validation, ROC/PR curves, confusion matrices, tuned
  hyperparameters), --dump-scores, --curves-svg, dataset from standard input
  (--data -)
- CSV input format with dialect auto-detection and missing-value handling;
  ARFF loader robustness (any line length, unsupported attribute types
  skipped, quoted names and values)
- leakage-free normalization option: --scaler-fit train fits the scaler on
  each training part only
- multinomial mutual-information ranking criterion (multinom-mi) with the
  mutual-information formula corrected
- candidate-level threading for all four Branch & Bound variants and for the
  hybrid and ensemble step evaluators - every search and evaluator is now
  threadable, bit-identical to sequential at any thread count; persistent
  worker pool
- criterion monotonicity is a declared property; Branch & Bound warns when
  the supplied criterion does not declare it
- kNN decision rule unified between classification and scoring (moved kNN
  results)
- reproducible-by-default builds: a fixed, CPU-independent release flag
  set (-DFST4_NATIVE_OPT=ON opts into machine-specific optimization)
- two full correctness reviews with every finding fixed, among them:
  Bhattacharyya over/underflow on badly scaled data, non-finite criterion
  values now rejected, threaded repeat-over-splits trials no longer repeating
  trial 1, --classify-samples respecting per-feature scaling, --search2 bif
  seeding every sequential search, --validate admitting every native
  classifier, class indices printed 0-based throughout
- bundled sample data sets with per-file attribution (data/NOTICE); the
  historical speech data set withdrawn (affected demos run on wdbc_30), the
  sonar class labels corrected to the official UCI split
- golden gate at 112 scenarios per platform, baselines for Windows x64,
  macOS arm64 and Linux arm64

<details>
<summary>Earlier versions: FST4 4.1 and 4.0 (internal milestones) and the FST3 releases</summary>

4.1.0 (internal milestone)
- budgeted stochastic (sampled) sequential step evaluator (--evaluator
  sampled): per-step evaluation cost independent of dimensionality, for
  very-high-dimensional wrapper selection
- threaded evaluation bit-identical to sequential at matched seed, for the
  sampled steps and the streaming searches (Monte Carlo, exhaustive) alike
- multinomial logistic regression classifier (wrapper-logreg), deterministic
  and dependency-free
- SFFS frontier bound (--sffs-delta)
- DAF ranking consolidated to the published 2011 method (probe-generating
  search + feature-statistics tracker), plus opt-in pairwise group statistics
- FST-owned PRNG making seeded runs reproducible across platforms
- ready-to-run parity config files for every published FST3 demo (demos/)

4.0.0 (internal milestone)
- C++17 required; the Boost dependency is fully removed (replaced by the C++
  standard library)
- the library is now header-only - no library translation unit needs compiling
- new fst4 command-line tool exposes the complete toolbox without C++ coding
  (all components and search methods, config files, text/JSON/CSV output,
  search progress streaming, reproducible seeding, cross-scenario similarity
  comparison)
- CMake build system introduced; the historical Makefile retired
- full-codebase audit completed with every finding either fixed or explicitly
  adjudicated (see RELEASE_NOTES.md) - including silent-wrong-result,
  crash and thread-safety defects; an incorrect lock that could freeze very long
  multi-threaded runs was found and fixed in a December 2025 pre-release audit
- golden-output regression harness (tests/) enables byte-exact build verification
- compiled demo collection reduced to three exemplar sources; all other demo
  scenarios are preserved as fst4 CLI scenarios producing identical results
  (demos/*.cfg, tests/manifest.tsv)
- CLI11 2.4.2 bundled; LibSVM 3.37 supported as an optional drop-in; optional
  Eigen-backed matrix numerics (verification builds)
- legacy cursor APIs and the threaded compatibility class shims are deprecated,
  removal planned for the next major release
- see RELEASE_NOTES.md for the complete report

3.1.0.beta
- optimal Branch & Bound methods
  + BBB, Basic Branch & Bound
  + IBB, Improved Branch & Bound
  + BBPP, Branch & Bound with Partial Prediction (averaging predictor)
  + FBB, Fast Branch & Bound (averaging predictor)
- DAF, Dependency-Aware Feature Ranking, is a new highly efficient method for very-high-dimensional
  FS; unlike BIF it does not ignore contextual information and, consequently, is capable of
  yielding considerably better results (enables wrapper-based feature selection with dimensionality
  in the order of 10^5-10^6; works with arbitrary wrapper)
  + DAF0 (standard)
  + DAF1 (normalized)
- SFRS/SBRS - Sequential Retreating Search algorithm is related to Floating Search but
  more thorough. Suitable also for use with secondary criterion (result regularization)
- 'generalized' variants of all sequential methods enabling more thorough search by testing
  feature g-tuples instead of just single features per step (see 1982 Devijver Kittler book)
  + (G)SFS, (G)SBS,
  + (G)SFFS, (G)SBFS,
  + (G)OS,
  + (G)DOS
  + (G)SFRS, (G)SBRS
- all sequential methods now allow start from arbitrary subset (useful for tuning
  of results using several different methods)
- individual ranking (BIF) threaded implementation (handy in very high dimensional tasks)
- Monte Carlo and threaded Monte Carlo method selects the best from a random sequence of feature subsets
- SFS/SBS, SFFS/SBFS, and SFRS/SBRS now enable post-search retreival of best results of each subset size
  as observed in the course of search
- modified SFFS implementation to fit more closely the original definition (now runs faster)
- re-implmented threading in sequential methods now more efficient due to reduced number of thread creations/destructions
- search method output is now redirectable to arbitrary output stream
- search method output can be switched off (introduced output levels SILENT, NORMAL, DETAILED)
- improved result trackers (cloning, joining, etc.)
- arbitrary data part access substitution (TEST for TRAIN, etc.) to enable bias estimation
- bias estimating wrapper
- cleaner stopwatch implementation
- now permits missing values in data - such values are substituted per feature by the mean value over valid values
- classifiers now implement method classify(), enabling classification of an arbitrary sample
- refactored directory structure
- lots of new demos showing broader variety of usage scenarios
- demos grouped according to purpose (for easier orientation especially of novice users)
- various minor improvements and additions (e.g., alternative random initialization of subsets, etc.)
- corrected several bugs and minor issues


3.0.2.beta
- added Exhaustive Search procedure in both sequential and threaded
  implementations to enable optimal feature selection
- corrected minor issues to support LibSVM 3.0
- result trackers now support cloning and memory usage limits
- added logfile with captured output of all demos
  for verification purposes (rundemos.log)
- corrected several minor issues

3.0.1.beta
- added support for reading ARFF (Waikato Weka) data files
- corrected minor issues to enable compilation in Visual C++

3.0.0.beta
- initial public release
- templated C++ code, using Boost library
- feature selection criteria
  ~ classification accuracy estimation based (wrappers), see data access options below
    + normal Bayes classifier
    + k-Nearest Neighbor classifier (based on various L-distances)
    + Support Vectior Machine (optional, depends on external LibSVM library)
  ~ normal model based (filter)
    + Bhattacharyya distance
    + Divergence
    + Generalized Mahalanobis distance
  ~ multinomial model based (filter) - Bhattacharyya, Mutual Information
  ~ criteria ensembles
  ~ hybrids
- feature selection methods
  ~ ranking (BIF, best individual features)
  ~ sequential search (hill-climbing)
    + sequential selection (SFS/SBS, restricted/unrestricted)
    + floating search (SFFS/SBFS, restricted/unrestricted)
    + oscillating search (OS, deterministic, randomized, restricted/unrestricted)
    + dynamic oscillating search (DOS, deterministic, randomized, restricted/unrestricted)
    + in any of the above: threaded, sequential, hybrid or ensemble based feature preference evaluation
  ~ supporting techniques (freely combinable with methods above)
    + subset size optimization vs. subset size as user parameter
    + result regularization (preference of solutions with slightly lower criterion value to counter over-fitting)
    + feature acquisition cost minimization
    + feature selection process stability evaluation
    + two-process similarity evaluation (to determine impact of parameter change etc.)
- flexible data processing
  ~ nested multi-level sampling (splitting to training, valitation, test and possibly other data parts)
  ~ sampling through extendable objects (includes re-substitution, cross-valiation, hold-out, leave-one-out, random sampling, etc.)
  ~ normalization through extendable objects (interval shrinking, whitening)
  ~ support for textual flat data format TRN (see FST1)

</details>
