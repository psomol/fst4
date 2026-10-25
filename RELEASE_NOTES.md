# Feature Selection Toolbox 4 — 4.2 Release Notes (the first public FST4 release)

> **FST4 ships as v4.2**, the first published FST4 version; 4.0 and 4.1 were
> internal development milestones that never shipped separately. Each version
> keeps its own detailed section below so that the 4.0 → 4.1 → 4.2 lineage
> stays visible, with the internal-milestone status noted per section.

## Version history at a glance

Publishing as v4.2 implies a 4.0 and a 4.1: those are internal development
milestones — **v4.2 is the first published FST4 version** (the last public
release of the toolbox was FST3). What distinguishes them:

- **4.0 — the modernization.** FST3's codebase brought to C++17 (Boost
  dependency removed), header-only, CMake-only; the `fst4` command-line
  tool making the whole toolbox usable without writing C++; one merged
  implementation per search algorithm (sequential vs. threaded execution
  chosen by an injected evaluation engine, not by class choice); a
  full-codebase audit fully adjudicated; the byte-exact golden-output
  regression harness as the release gate; optional Eigen cross-validation
  of the linear-algebra core.
- **4.1 — very-high-dimensional selection.** The budgeted
  stochastic (sampled) sequential step evaluator with per-step evaluation
  cost independent of dimensionality; threaded evaluation made
  bit-identical to sequential at matched seed (sampled steps and streaming
  searches alike); the multinomial logistic-regression classifier; the
  SFFS frontier bound (`--sffs-delta`); DAF ranking consolidated to the
  published 2011 method plus opt-in pairwise group statistics; the
  cross-platform PRNG shim making seeded runs reproducible across
  Windows/macOS; ready-to-run parity configs for every published FST3
  demo.
- **4.2 — the practitioner wrapper expansion.** Wrapper metrics beyond
  accuracy (balanced accuracy, F1, AUC, per-class cost weights, R²/RMSE/
  MAE); five new deterministic dependency-free wrapper models (Random
  Forest, linear SVM, RBF kernel SVM, ridge regression, LDA) plus the
  external-classifier bridge AND the external-criterion/set-function
  layers; LibSVM thereby demoted to an optional extension (and no longer
  distributed, per licensing policy); candidate-level threading for all
  B&B variants and the hybrid/ensemble step evaluators (every search and
  evaluator now threadable, bit-identical to sequential); the kNN
  decision-rule unification; the reliability hardening pass behind all of
  it.

## The 4.2 wrapper expansion

- **Wrapper criterion metrics beyond accuracy** — every wrapper criterion
  (`wrapper-knn/bayes/logreg/multinom/svm/regressor`) now optionally optimizes
  and reports a metric other than plain accuracy (`--wrapper-metric`): `balanced-accuracy` (macro-averaged recall; test-absent
  classes skipped), `f1` (positive class = class index 1, two-class data),
  `f1-macro` (0/0 contributes 0), `auc` (two-class; exact Mann-Whitney rank
  statistic with midrank ties, computed from deterministic per-sample scores —
  logistic-regression linear scores, Bayes/multinomial posteriors, kNN vote
  fractions, LibSVM decision values; LibSVM's probability mode is deliberately
  not used — it retrains with libc-rand-driven internal CV), and for the
  regression wrapper `r2`/`neg-rmse`/`neg-mae`. `--class-weights w0,w1,...`
  adds per-class cost weighting of the confusion matrix (combines with
  accuracy/f1/f1-macro; deliberately not with balanced-accuracy or auc).
  Metrics are computed per criterion-protocol split and averaged, like
  accuracy always was; `--validate` lines report holdout accuracy and the
  selected metric side by side (text/json/csv writers extended). The seams:
  `Classifier::test_confusion()/test_scores()/test_metric()` with default
  implementations on the abstract base (per-pattern `classify()`/
  `classify_scores()` do the work), `Regressor::test_metric()`, and
  `Criterion_Wrapper::set_metric()`. Everything is RNG-free and copied by
  clones, so the bit-identical threading guarantees are untouched; the default
  (`accuracy`, no weights) bypasses the new machinery entirely — every
  pre-existing golden scenario passes byte-identical, five new `fst4_metric_*`
  scenarios pin the surface. New
  demos: `demo14.cfg` (balanced vs. plain accuracy compared by tracker
  similarity; the number is reused from the never-released FST3 draft),
  `demo15.cfg` (AUC/logreg), `demo16.cfg` (cost weights), `demo17.cfg` (R²
  regression).

- **Random Forest wrapper** (`--criterion wrapper-rf`, `classifier_rf.hpp`) —
  a dependency-free CART/Gini Random Forest (bagging, sqrt-mtry per-node
  feature subsampling; `--rf-trees/-mtry/-max-depth/-min-leaf`), deterministic
  by design: training randomness comes from a local xorshift64* generator
  seeded by an FNV-1a hash of the candidate subset content, salted with the
  process `--seed` and the split index — never from the thread_local
  `fst_rand` stream. Every subset evaluation is therefore a pure function of
  (subset, data, seed): bit-identical at any `--threads` count and under any
  evaluator, verified by the paired `fst4_rf`/`fst4_rf_threads` golden
  scenarios (identical result payloads). All tie-breaks are pinned (total
  sort order (value, index); equal-gain splits keep the first candidate;
  vote ties take the lowest class index; FP-midpoint thresholds guarded).
  The full 4.2 metric palette applies (`classify_scores` = tree-vote
  fractions, so AUC works). New demo: `demo18.cfg` (spambase, 100 trees,
  SFS to 10 features, threads 4 — under a minute).

- **Ridge regression wrapper** (`--criterion wrapper-ridge`,
  `regressor_ridge.hpp`, `--ridge-lambda` default 0.01) — closed-form
  L2-regularized least squares solved by Cholesky on internally standardized
  features (bias unpenalized), sharing the wrapper-regressor pipeline
  (class-label dependent variable, `--validate same`). The criterion value
  defaults to R² through the 4.2 metric palette (`neg-rmse`/`neg-mae`
  selectable); deterministic and RNG-free — byte-identical at any `--seed`
  and `--threads`. New demo: `demo19.cfg` (the linear counterpart of
  demo17; on WDBC it beats the weighted-neighbours regressor's holdout
  R²). DOC FIX ridealong: the help/reference texts claimed the regression
  pipeline's dependent variable is "the data's last column" — it has always
  been the class label ("regression simulating classification", the
  demo11reg protocol); texts corrected, behaviour untouched (golden-pinned).

- **Linear SVM wrapper** (`--criterion wrapper-linsvm`,
  `classifier_linear_svm.hpp`, `--linsvm-c/-loss/-maxiter/-tol`) — a
  dependency-free L2-regularized linear SVM trained by dual coordinate
  descent (the Hsieh et al. 2008 method, implemented from scratch; no
  LIBLINEAR or LibSVM code). The coordinate order is deliberately FIXED
  CYCLIC instead of the usual random permutation: training is RNG-free and
  fully deterministic — byte-identical at any `--seed` and `--threads`.
  Squared-hinge (default) and hinge losses; internal per-split
  standardization; bias via an appended constant coordinate; binary data
  trains one model (class index 0 = the +1 side), multiclass is
  one-vs-rest. `classify_scores` = decision values, so the full 4.2 metric
  palette applies (AUC included). This retires the long-documented LibSVM
  LINEAR-kernel freeze for linear-SVM selection (Known Issues updated).
  Goldens `fst4_linsvm` (wdbc) + `fst4_linsvm_multiclass` (waveform,
  one-vs-rest); new demo `demo38.cfg` (the demo18 spambase protocol with a
  linear margin model — seconds).

- **RBF kernel SVM wrapper** (`--criterion wrapper-rbfsvm`,
  `classifier_rbf_svm.hpp`, `--rbfsvm-c/-gamma/-loss/-maxiter/-tol`) — a
  dependency-free C-SVC with the RBF kernel,
  trained by kernel dual coordinate descent: the linear solver's structure
  kernelized via the PENALIZED-BIAS kernel offset (K+1, the kernel analogue
  of the appended constant coordinate), which removes the equality
  constraint that otherwise forces two-variable SMO. Fixed cyclic order =
  RNG-free and fully deterministic — byte-identical at any `--seed` and
  `--threads`. Margin sums are maintained incrementally (O(1) gradient
  checks); the N×N kernel matrix is precomputed within a 512 MB budget and
  shared by all one-vs-rest models (above budget: rows recomputed on
  demand). Hinge (default, the C-SVC correspondence) and squared-hinge
  losses; internal per-split standardization; gamma default 1/d on the
  standardized features; models compacted to their support vectors;
  `classify_scores` = decision values (full metric palette incl. AUC).
  Deliberately C-SVC+RBF only — no nu-variants/SVR/one-class/probability/
  shrinking/poly/sigmoid — and deliberately NOT bit-comparable to LibSVM
  (cross-checked at matched C/gamma on whitened wdbc: holdout 0.954 vs.
  LibSVM's 0.958). Together with wrapper-linsvm this covers both SVM
  kernels of practical value natively: LibSVM is now an Eigen-like optional
  extension (FST3 parity configs and cross-checks only). Goldens
  `fst4_rbfsvm` (wdbc) + `fst4_rbfsvm_multiclass` (waveform, one-vs-rest,
  threaded engine); new demo `demo28.cfg` (the demo23 sonar protocol with
  the native kernel machine — sub-second). `--rbfsvm-optimize
  pre|alternating` is the native C/gamma twin of `--svm-optimize`: the same
  space-filling random search (fst_rand on the master thread — deterministic
  at matched `--seed`) and the same pre/seed/alternating run-protocol
  choreography, with no LibSVM involved. Golden `fst4_rbfsvm_opt`; new demo
  `demo29.cfg` (the FULL demo23 protocol natively, grid included).

- **Research-grade result reporting: `--report detailed`** — one flag, orthogonal to `--output` and
  `--verbosity`; `standard` (default) keeps the historical output
  byte-identical, `detailed` adds, in all three formats: `curve_over_d`
  (the best subset at every size the search visited — criterion value
  recorded live from the solution announcements, plus a holdout
  re-validation of each subset: the criterion-axis vs. validated-axis
  overfitting curve, with the two axes deliberately kept apart), tabular
  ROC and PR curves of the validated classifier on the depth-0 holdout
  (one point per distinct threshold, deterministic tie handling; per-class
  one-vs-rest on multiclass), confusion matrices in every validation
  entry, and the tuned hyperparameters after an optimize protocol
  (previously prose-only — a reproduction gap). No plotting dependency:
  the tables feed gnuplot/matplotlib directly (recipes in
  doc/FST4_CLI.md). The same doc gains a "Unix pipe usage" section for
  the by-design pipe-clean stdout contract of the machine formats.
  Goldens `fst4_report_detailed` (wdbc, the curve shows holdout peaking
  at d=3 while the criterion keeps rising to d=4 — the classic lesson),
  `fst4_report_multiclass` (waveform, per-class curves + macro-AUC
  column), `fst4_report_tuned` (CSV sections + tuned C/gamma).

- **The reporting story completed: `--dump-scores`, `--curves-svg`, `--data -`**
  (the three items deferred from the reporting work's first cut).
  `--dump-scores FILE` writes the selected subset's per-sample holdout
  predictions as CSV — true class, predicted class, and one score column
  per class where the classifier exposes scores — reusing the first
  `--validate` classifier with no extra training, in the same enumeration
  the ROC/PR curves are built from (dump row *i* = curve point *i*); it is
  independent of `--report`, so it works in standard mode. `--curves-svg
  FILE` renders the detailed-mode curves as a self-contained SVG with no
  plotting dependency and no external resources: ROC and PR panels (one
  polyline per class on multiclass data) plus the over-d panel, that one
  with SEPARATE left/right y-axes so the selection-time criterion and the
  independent holdout are never read off one scale. `--data -` reads the
  dataset from standard input (`--format` then required), which makes fst4
  a filter in both directions — `zcat data.trn.gz | fst4 --data - --format
  trn ... --output json | jq ...`. Goldens `fst4_dump_scores`,
  `fst4_curves_svg` (both catting the written file, so the artifact content
  itself is pinned) and `fst4_stdin_trn`; existing output is untouched —
  standard-mode stdout is byte-identical with the new flags set.

- **One-vs-rest macro AUC** —
  `--wrapper-metric auc` now covers any class count: two-class data keeps
  the exact binary AUC (unchanged); K>2
  computes the unweighted mean of per-class binary AUCs, each from the
  class's own `classify_scores` column against the rest with the same
  midrank tie rule, skipping classes absent from the test part (the
  balanced-accuracy convention). New seam: `Classifier::test_scores_matrix`
  (samples × classes score matrix, deterministic enumeration). The LibSVM
  wrapper gained the missing multiclass score mapping: the one-vs-one
  decision values aggregated per class (score[i] += dec(i,j), score[j] -=
  dec(i,j)) — continuous and deterministic, still never the probability
  mode. Goldens `fst4_metric_aucmacro` (logreg) + `fst4_metric_aucmacro_svm`
  (LibSVM mapping); new demo `demo27.cfg` (waveform macro-AUC selection).

- **LDA wrapper** (`--criterion wrapper-lda`, `classifier_lda.hpp`,
  parameterless) — the shared-covariance (Fisher/Gaussian) linear
  discriminant: per-class Model_Normal moments pooled by the class priors,
  ONE matrix inverse per training through the same Indexed_Matrix LU
  machinery as the normal-model filters and normal Bayes — the optional
  Eigen-backed path (`FST4_USE_EIGEN`) stays in parity. Closed-form,
  deterministic, K classes, roughly K× cheaper per training than the Bayes
  classifier; `classify_scores` = discriminant values (AUC-ready). Like all
  normal-model components it fails on singular pooled covariance (constant
  features, subsets larger than the sample count). Golden `fst4_lda`
  (waveform, 3 classes). On waveform it is the strongest 4.2 linear model
  (holdout 0.783 at d=5 vs. linsvm 0.750).

- **External-classifier bridge** (`--criterion wrapper-extern
  --extern-cmd "<command>"`, `classifier_extern.hpp`) — wrapper selection
  around any classifier speaking a three-message line protocol
  (TRAIN header + labeled rows → OK; TEST header + rows → predicted labels;
  `%.17g` values) over ONE persistent child process per criterion clone
  (spawn cost paid once, not per evaluation; clones never share pipes, so
  threaded evaluation stays isolation-safe). The process plumbing is the
  only platform-sensitive code in FST4, isolated in the one header. Both
  paths are golden-verified: the POSIX side (fork/exec) and the Windows
  side (CreateProcess) each pass the `fst4_extern` scenarios on their own
  machine. Documented loudly and deliberately: the command is
  user-supplied code execution through the shell, and FST4's
  reproducibility guarantees END at the bridge. AUC is unavailable
  (labels-only protocol); confusion-based 4.2 metrics work through a
  batched `test_confusion` override. Reference external classifier:
  `tests/extern_centroid.awk` (nearest centroid); new demo `demo64.cfg`.

- **External VALUE criterion — bring your own criterion, or none at all**
  (`--criterion extern-value --extern-cmd "<command>"`,
  `criterion_extern_value.hpp`; shares the `extern_bridge.hpp` process
  plumbing and every wrapper-extern caveat). Where wrapper-extern plugs an
  external CLASSIFIER into FST4's accuracy estimation, extern-value hands the
  whole criterion to the external tool: the full-width train part streams
  ONCE per split (a DATA block, re-sent exactly when a filter would
  re-estimate its model), and each subset evaluation is then an index-only
  `VALUE <d> <f1>..<fd>` request answered with one double — the
  sufficient-statistics tool pattern: O(d) per request with feature
  identities available (per-feature caching, cost-aware measures).
  `--extern-monotonic` asserts cardinality-monotonicity, which admits the
  optimal `bb-*` searches — optimality then rests entirely on the assertion
  (without it the CLI refuses B&B and the library warns). `--extern-no-data`
  suppresses DATA entirely: pure SET-FUNCTION optimization over an abstract
  feature universe. Introduction-verified: the reference tool
  (`tests/extern_value_meandiff.awk`, sum of per-feature |class-mean
  differences|) matches an independent python computation at ~5e-13, and
  bb-improved/bb-fast return the brute-force-confirmed exhaustive optimum;
  non-finite responses are rejected loudly. Goldens `fst4_extern_value`,
  `fst4_extern_value_bb`, `fst4_extern_value_nodata` (the set-function
  reference `tests/extern_value_setfunc.awk`). The three extern layers —
  external classifier / external data-dependent criterion / external set
  function — are documented side by side in doc/FST4_CLI.md.

- **Branch & Bound candidate-level threading + the persistent worker pool** —
  all four B&B variants (`bb-basic`, `bb-improved`, `bb-fast`, `bb-bbpp`) now
  accept `--threads N` with results **bit-identical to sequential for every
  variant at any thread count**: candidate evaluations batch through the injected
  candidate-evaluation engine (IBB's level pre-ordering; FBB/BBPP via a two-pass
  predict-vs-compute split decided upfront on the master — exact because
  predictor state is per-feature, now a documented contract in
  `branch_and_bound_predictor.hpp`, with predictor learns replayed in
  add-order; chosen-node post-processing and leaf evaluation batch through
  base-class hooks). The lazy compute-before-cut stays on the master
  (critical path); tree-level parallelism (concurrent subtrees over the
  shared bound) is a deliberate non-goal — it would trade away bit-parity.
  Supporting it, the threaded engine's batch workers now spawn ONCE and PARK
  between batches (completion-counter handshake, join only at destruction),
  which collapsed the per-batch overhead that penalized many-small-batch
  users: bb-basic t4 went from +78% over sequential to +16%, bbpp from +43%
  to +7%, IBB scales to 2.5× at t8, and the sampled evaluator's step
  batches got cheaper too — with unchanged scheduling semantics, so all
  determinism guarantees hold. Note the honest caveat: with the cheap
  built-in filters at small d, the small-batch variants still do not PAY
  from threading (their computed fraction is small by design — that is what
  makes FBB fast); `--threads` on B&B is an expensive-criterion tool.
  Goldens: `fst4_bbpp` (first-ever BBPP coverage) + threaded twins
  `fst4_bbpp/bbfast/bbbasic_threads`.

- **Hybrid and ensemble step evaluators take the threaded engine** — the two
  remaining sequential-step evaluators accept `--threads N`: hybrid
  thread-pools the main-criterion phase (the filter pre-sort stays on the
  master), ensemble runs one engine batch per member criterion (vote
  combination on the master). Results identical to sequential (per-result
  callbacks fire in candidate order, ties break by earliest candidate), so
  every `--evaluator` is now threadable. The `fst4_ensemble_demo50` golden
  runs threaded in-scenario.

- **Criterion monotonicity is now a declared property** —
  `Criterion::is_monotonic()` (default false); the normal-model and
  multinomial abstract bases override to true (all five built-in filters are
  cardinality-monotonic — Divergence in theory only, see Known Problems).
  `Search_Branch_And_Bound::search()` prints a WARNING at normal verbosity
  when the supplied criterion does not declare the property, so the B&B
  optimality caveat reaches library users with custom criteria
  automatically; built-in filters stay silent (golden output unchanged).
  The adapter criteria (negative/subset-size/sum-of-weights) deliberately
  keep the false default — `Criterion_Negative` flips monotonicity by
  construction.

- **Multinomial Mutual Information exposed** (`--criterion multinom-mi`) —
  `Criterion_Multinomial_MI` existed in the codebase since FST3 but was
  never instantiated anywhere (found as dead code; exposed rather than
  deleted). It is an individual
  (per-feature) MI ranking criterion: a subset evaluates to the SUM of
  per-feature MI values — monotonic but NOT a joint measure, intended for
  `--search bif` term ranking on count/text data. Golden `fst4_multinom_mi`;
  new demo `demo44.cfg` (reuters term ranking). NOTE: the per-term formula
  inherited from FST3 was wrong and is corrected in this release (see
  "Reliability: the second review pass" below); the cross-check quoted in
  earlier drafts of these notes had verified the code against its own
  formula.

## Data pipeline: leakage-free normalization

- **Per-split data scaling, `--scaler-fit train|all`** — the
  load-time scalers (`to01`, `white`, and `--missing-value` imputation)
  historically learn their parameters over the ENTIRE dataset before any
  train/test split, so test data contributes to the normalization — an
  FST3-compatible, documented methodological leakage. `--scaler-fit train`
  closes it: the transform is learned on the current depth-0 split's TRAIN
  part only, applied to all data, and re-fitted automatically on every
  depth-0 split change — the same per-split discipline the 4.1/4.2 dense
  trainers (logreg/ridge/linsvm/rbfsvm) already apply internally. Threaded
  runs stay bit-identical to sequential (every accessor clone derives the
  identical split transform on a private buffer). The default `all`
  preserves the historical behavior byte-identically; `--classify-samples`
  requires it (user vectors need the load-time transform). Library API:
  `Data_Accessor_Splitting_Mem::set_scaler_fit_train()` before
  `initialize()`; user-derived scalers support clone-based threading by
  overriding the new `Data_Scaler::duplicate()`.

## Data input: CSV format + ARFF robustness

- **CSV input format** —
  the simplest Excel→FST4 transfer path, removing the TRN/ARFF-only barrier:
  `--data file.csv` (or `.tsv`; `--format csv` for other names). Dialect
  handling is auto-detecting by default: delimiter from line 1
  (comma/semicolon/tab, `--csv-delimiter` overrides) with a deliberate
  comma-LAST tie-break so European-locale Excel exports (semicolon-delimited
  with decimal COMMAS — accepted whenever the delimiter is not the comma)
  load with zero flags; header row detection (`--csv-header` overrides);
  class column = the LAST column by default (`--csv-class` takes a 0-based
  index or a header column name); Excel's UTF-8 BOM and `""`-escaped
  double-quoting handled; class labels are arbitrary tokens. Rows with
  missing values (`?`, `NA`, `N/A`, `NaN`, `NULL`, empty) are skipped
  LOUDLY with a summary (the ARFF skip discipline); malformed input stays
  fatal with the row number. Golden scenarios `fst4_csv_basic`/`_semicolon`
  pin the same 20 records in both dialects and must report identical
  result lines — the dialect-equivalence proof.
- **`--csv-missing-to <code>`** — closes the gap plain row-skipping
  leaves: row-skipping is the safe default but costs a user with scattered
  blanks much of the data set, while FST4's `--missing-value` imputation
  (the demo62 protocol) could not be reached from CSV at all. The flag maps
  missing FEATURE cells to a numeric code and KEEPS the row; a missing
  CLASS label still skips, since a row without a label belongs to no class.
  The mapping is deliberately PURE — it does not imply `--missing-value`.
  Pair the two flags (same code) to substitute per-feature training
  averages, or leave the code in place as an explicit missingness indicator
  a tree-based wrapper can split on. Three guards keep the pairing honest:
  `--scaler void` only (to01/whitening would fold the code into the scaling
  statistics), disagreeing codes across the two flags are a fatal error
  (the mapped code would never be substituted), and the load summary names
  the pairing flag whenever the mapping fires. Golden
  `fst4_csv_missing_to`: the semicolon fixture's missing row is kept (21
  records, not 20), separating all three modes — its criterion value
  differs from skip mode, its validation from map-only mode.
- **Reproducible-by-default builds: the two-preset flag policy.** A default
  CMake build now uses
  fixed, CPU-independent release codegen (`-O2 -funroll-loops
  -ffp-contract=off`), identical on every architecture — binaries are portable
  across machines of the same platform+arch, seeded results reproduce
  machine-to-machine, and the golden regression baselines are captured under
  exactly these flags, so a default build is always gate-comparable.
  `-DFST4_NATIVE_OPT=ON` opts into the historical machine-specific native
  optimization (`-march=native`/`-mcpu=native`) for speed experiments; the
  golden harness detects such builds, refuses to capture baselines from them
  and warns when checking against them. On Apple Silicon the default equals
  the flags in force since the macOS port, so nothing changes there; x86
  builds drop `-march=native` and the historical x87 (`-mfpmath=sse,387`)
  mixing by default.
- **`--features N` — the data-file-free entry point.** No `--data` at all: the flag
  declares an abstract universe of N features, and an external tool scores
  index sets over it (`--criterion extern-value --extern-no-data
  --extern-cmd CMD` — required explicitly). This completes what
  `--extern-no-data` began: that mode already suppressed the DATA block,
  but a real data file still had to exist purely to declare the universe
  size. FST4 thereby works as a general black-box set-function optimizer —
  cost tables, simulations, external experiments — with its full search
  machinery available: every search including the four optimal B&B
  variants (under `--extern-monotonic`), all trackers with DAF statistics,
  `--search2 bif` seeding, the sampled evaluator, subset-size/cost
  regularization, and `--threads N` bit-identical to sequential.
  Everything whose meaning depends on data existing is refused with an
  actionable message (`--validate`, real splitters, scalers, wrapper
  criteria, `--report detailed`, hybrid/ensemble evaluators). The JSON
  invocation echo carries `"features": N` in place of the data/format
  keys; the data block reports the honest degenerate shape (0 classes, 0
  samples). Golden `fst4_features_setfunc` pins the entry against
  `fst4_extern_value_nodata`: same protocol, same `[1,2,20,21,22]`/91
  optimum — the entry points differ, the optimization is identical.
- **ARFF loader robustness** — data lines of ANY length
  (the historical 8 KB cap silently split wide rows) and graceful per-column
  SKIP of unsupported attribute types (STRING/DATE/unknown) with loud
  reporting, instead of aborting the whole load; malformed headers stay
  fatal. Also fixed in passing: a `FILE*` leak in `prepareFile()`.

## Behavior changes in 4.2 (golden-recaptured)

Two deliberate behavior changes landed with the 4.2.0 version bump in one
full-recapture wave (every golden re-verified green after recapture):

- **kNN decision-rule unification** — kNN `classify()`
  now IS the `classify_scores()` decision rule: vote over exactly the k
  nearest neighbours, argmax, first maximum (lowest class index) wins — so
  accuracy/confusion and AUC/ROC describe the SAME classifier for every
  classifier in the toolbox. The historical rule (early-majority vote over
  up to (k−1)·classes+1 nearest-first neighbours, nearest-order tie
  resolution) is retired together with its machinery and the "wrong class"
  convention; kNN also joins the shared base-class accuracy loop. Impact is
  provably nil on two-class data at the shipped scenarios' k (the two rules
  coincide there — only the sole K>2 kNN golden moved in content); on
  multiclass data kNN accuracy values can shift, which is the point: they
  now measure the same rule the ranking metrics score.
- **Oscillating Search initial-subset shrink un-inverted** — `Search_OS::adjust_initial_subset` shrinks an oversized initial
  subset by deselecting the WORST-BIF-ranked features; the historical code
  walked the ranking best-first in both directions, i.e. removed the BEST
  features when shrinking. The path is unreachable through the fst4 CLI
  (`--search2-d` must equal `--target-d` for os), so no CLI scenario is
  affected; library-API users with oversized seeds get the intended
  behavior. Verified by a library-level probe: the adjusted subset keeps
  exactly the best-ranked members of the seed.

## Small fixes and hygiene

- **`--f1-positive <idx>`** — `--wrapper-metric f1` scores class index 1 as the
  positive class. On data whose positive class is index 0 that optimizes the
  wrong objective, and `--class-weights` is not a substitute for it. The new
  flag names the positive class explicitly (two-class data; `f1-macro`
  averages over all classes and `auc` keeps its own class-1 convention, both
  stated in `--help`). The default is unchanged — an explicit
  `--f1-positive 1` is byte-identical to omitting the flag.
- **The golden harness knows LibSVM is optional** — a binary built without it
  (`-DFST4_WITH_SVM=OFF`, or `libsvm/` never populated) now SKIPS the
  `wrapper-svm` scenarios instead of failing them, so such a build gates
  cleanly on everything else. The skip count stays in the summary line: an
  SVM-less green run is not full coverage and does not report itself as such.
- **A latent 32-bit wrap in threaded streaming searches** — the ordered-replay
  ring counters were `unsigned long`, which is 32 bits on Windows (LLP64) and
  64 on macOS/Linux. The ring is indexed by `head % W`, so a wrap would have
  desynchronized the slot mapping whenever the worker count is not a power of
  two — silently corrupting result ORDER, not just the reported ordinal.
  Reaching it takes more than 2^32 candidate evaluations in one search, which
  is out of practical reach; the point is that the failure would have been
  silent. Widened to 64 bits across the streaming path and the trial counters
  that move in lockstep with it.
- **Debug builds are warning-clean** — the `-Wall` Debug configuration built
  with 84 warnings over 42 sites (unused variables, misleading indentation,
  member-initializer order); it now builds clean. Every change is a statement
  split, a dead-declaration removal, or an initializer reordering, so the
  golden set is byte-identical — verified across the full suite, slow tier
  included, not only the fast tier.

## Reliability: the 4.2 review

The 4.2 feature program was followed by a dedicated review of the whole
codebase: a security review of the process-spawning surface, a bug hunt over
criteria/searches/dataio/CLI with every candidate finding verified before
acceptance, a direction-symmetry sweep of the forward/backward search pairs,
performance scouting, and a final simplification pass. The user-relevant
fixes:

- **Backward searches no longer corrupt state across invocations** —
  initialization order wrote mode-dependent state before the search
  direction was set, so a Subset left in backward mode by one trial came up
  EMPTY at the next: every sbs/sbfs/sbrs run under `--repeat-over-splits`
  (the stability protocols) died with "search not finished". Found exactly
  where golden coverage was zero; fixed in all three searches, and the gate
  grew direction-symmetry scenarios (sbfs, sbrs, threaded sbs, bb-basic, a
  backward repeat-over-splits protocol) so the class stays covered.
- **Threaded runs fail as cleanly as sequential ones** — a criterion
  failure (e.g. singular covariance) inside a worker thread previously
  escaped as std::terminate (SIGABRT, no message); workers now capture the
  exception, the batch drains, and the master rethrows — threaded failure
  behavior matches sequential exactly.
- **`--classify-samples` respects the scaler** — user-supplied vectors
  were classified RAW against models trained on scaler-transformed data
  (silently wrong predictions under to01/whitening, wrong missing-value
  handling); they now pass through the active scaler and `--missing-value`
  substitution exactly as the training data did.
- **External-bridge robustness** — SIGPIPE from a dying
  child no longer kills fst4 silently; TEST responses are parsed strictly
  (garbage no longer becomes class 0) with a clean respawn after protocol
  failure; clone children no longer inherit each other's pipe ends
  (FD_CLOEXEC + a spawn lock — the source of a shutdown deadlock under
  `--threads`); untrusted-config runs print a one-line spawn notice.
- **LibSVM wrapper hardening** — a single-class training fold made the
  binary scores path read past the model's label array (heap OOB); the
  branch now follows the MODEL's class count.
- **Portability fixes** — the linearized sample-matrix
  offset arithmetic is size_t-widened across all models and classifiers
  (32-bit index products overflowed at samples×features ≥ 2^32); the ARFF
  loader's variable-length arrays are gone (they broke the documented
  Visual C++ path — MSVC hard error); NOMINMAX guards the Windows extern
  header; assorted LLP64 width hygiene.
- **Smaller fixes** — a TRN parser operator typo (`%` for `&`) restoring
  the intended error code; RFC-valid JSON escaping of control
  characters in `--output json`; an unsigned-underflow guard on
  degenerate splits; parity-config corrections incl. threading the
  demo50 golden and de-capping demo60's tracker to match its retired
  source (result-neutral — verified); doc/comment sweeps incl. disclosing that the FST3 parity target is the git-history
  demo sources, which diverge from the live FST3 web page for at least six
  demos.
- **Simplification** — the review's final pass removed duplicated code
  (shared dense-training helpers, a base-class accuracy loop, per-pattern
  allocation hoists, CLI dispatch dedup), every step byte-identical under the
  full regression gate.

## Reliability: the second review pass

A second full correctness review — the criterion and classifier mathematics,
the data layer, the search layer, the CLI pipeline and every user-facing text
— found five defects that produced wrong results, hung, or failed on
realistic input, plus a tail of edge-case and documentation items. All are
fixed in this release; where a fix moves a golden, the entry says so.

- **Bhattacharyya no longer over/underflows** — the covariance term
  `0.5*ln(|S|/sqrt(|S0||S1|))` was formed from three raw determinants
  (products of d LU pivots), which leave the double range long before the
  matrices are ill-conditioned: 120 Gaussian features with variance 1e8 gave
  `value=nan` at d=80 and a search that "succeeded" with features 0..79 in
  index order (NaN never compares greater, so the first candidate always
  won), while the same data at scale 1e-3 aborted with "division by zero"
  although every matrix was perfectly conditioned. The term is now formed in
  the log domain via `Indexed_Matrix::log_determinant()` — scale-independent,
  exact wherever the old product was still finite — and
  `Criterion_Normal::evaluate()` refuses to hand a non-finite value to any
  search (a diagnostic error instead of silent garbage). All 21 goldens that
  evaluate the normal-model filters are byte-identical.
- **`multinom-mi` computes mutual information** — the per-term score
  inherited from FST3 was `sum_c P(c) ln(theta(f|c)/P(f))` with a
  Laplace-smoothed numerator over an UNSMOOTHED corpus marginal: the joint
  factor theta(f|c) was missing, a term seen once in a small class scored
  several nats, 2769 of the 10105 reuters terms scored NEGATIVE and every
  one of the top-20 terms occurred in exactly one document. The score is now
  `MI(f) = sum_c P(c) theta(f|c) ln(theta(f|c)/P(f))` with
  `P(f) = sum_c P(c) theta(f|c)` — each term's share of I(term; class) under
  the multinomial model, non-negative by the log-sum inequality, so the
  subset sum is genuinely monotonic and the `bb-*` admission holds. Golden
  `fst4_multinom_mi` recaptured: the multinomial NB validation accuracy of
  the reuters top-500 rises from 0.440 to 0.907 (multinomial Bhattacharyya
  top-500: 0.901), and the two rankings now share 472 of their 500 terms
  (before: 0). `demo44.cfg`'s header is rewritten accordingly.
- **`--repeat-over-splits` with an un-nested wrapper is refused instead of
  hanging** — without `--splitter2` a wrapper criterion's own split loop runs
  on the depth-0 splits, the very cursor the trial loop walks: every
  evaluation exhausted and reset the trial enumeration, so `randrand`/
  `randfix` trials never ended (7 "trials" per minute, forever) and `cv`/
  `leave1out` produced one silent trial with no stability block — and each
  trial value was a mean over ALL outer splits anyway. The combination (also
  with the ensemble evaluator) now fails at startup with the fix in the
  message: nest the estimation with `--splitter2`, as demo54/55 do. The
  help, `--list`, the `stabileval` error and the CLI reference, which
  recommended the un-nested form, are corrected. Filters (which re-estimate
  their model per trial) keep working un-nested; goldens unchanged.
- **Threaded trial protocols no longer repeat trial 1** — the thread-pooled
  evaluation engine clones the criterion once per worker and re-uses the
  clones while the same criterion object is supplied; a wrapper's clone
  carries a deep copy of the data accessor's split state, so under
  `--repeat-over-splits --threads N` the workers stayed on the FIRST trial's
  depth-0 split while the master advanced: every trial reproduced trial 1
  verbatim (CWrel = ATI = 1, zero value stddev), and the un-nested filter
  form kept the first trial's model. The engine interface gains
  `invalidate_clones()`; the CLI registers every engine it builds and drops
  the clones at the start of each trial, of each `--regr-multipliers` sweep
  step and of each alternating `--svm-optimize`/`--rbfsvm-optimize`
  re-search (the same staleness with a re-tuned classifier or a re-set
  multiplier). Threaded runs of these protocols are now byte-identical to
  `--threads 1`; new golden `fst4_repeat_threads` pins the demo54 protocol at
  4 threads against the sequential golden. The straight sequential step is
  constructed as the merged `Sequential_Step_Straight` with the injected
  engine (the deprecated `*_Threaded` shim is no longer used by the CLI);
  output unchanged.
- **`--classify-samples` transforms every coordinate with ITS feature's
  scaler** — the load-time scaler is one one-dimensional object re-learned
  feature after feature, so after loading it held only the LAST feature's
  min/max (to01), mean/stddev (white) or substitution average
  (`--missing-value`), and the 4.2 routing of user vectors through it
  applied that one transform to all coordinates: ten rows copied verbatim
  from the training part came back as `1 1 1 1 1 1 1 1 1 1` from a 1-NN
  classifier under to01 or white (void: the correct `0 0 0 0 0 1 1 1 1 1`).
  The accessor now keeps a duplicate of the scaler per feature and exposes
  `scale_sample_inplace()`, which the CLI applies to each user vector. The
  same mechanism carries the current split's transform under `--scaler-fit
  train`, so that combination is no longer refused (the classifier and the
  vectors use the split the classifier is trained on). New goldens
  `fst4_classify_scaled`/`fst4_classify_fit_train` pin the verbatim-row
  probe; the void-scaler golden is unchanged.
- **ARFF quoting** — dense rows rejected quoted values (`'alpha'` failed the
  nominal lookup, `"4.90"` the numeric parse: "Nominal value not found", load
  aborted) although the reader's documentation promised optional quotes and
  the sparse parser already stripped them; and a quoted attribute name
  containing a blank (`@ATTRIBUTE 'my attr' NUMERIC`, Weka's own way of
  writing such names) split at the blank into a name `'my` and an
  "unsupported type" `attr' NUMERIC`, which the column-skipping path then
  SKIPPED: the file loaded with one feature fewer and shifted indices. Both
  parse now (quotes are stripped from values, a quoted name ends at its
  closing quote). A failed ARFF load also raises an error naming the file
  instead of the bare "File not loaded" that hid the reader's row diagnostic.
  New golden `fst4_arff_quoted` (fixture `tests/quoted.arff`).
- **Random draws reach every index** — the library PRNG returns 15 bits
  (0..32767), and two call sites took a plain modulus of it: random feature
  subsets (`make_random_subset`: Monte Carlo probes, DAF, `--search os`
  initial subsets, `--restarts-nonimproving`, warm-up probes) never contained
  a feature with index >= 32768 except through the linear probe's
  wrap-around, and `randrand`/`randfix` drew training samples of a class
  with more than 32768 samples from its first 32768 only. Both now go
  through `fst_rand_below(n)`, which is the identical single draw up to
  n = 32768 (every seeded sequence stays what it was — all goldens
  byte-identical) and composes two draws into 30 bits above it.
- **Nested splitting at depth 2 and deeper translates through the right
  list** — a new level's intervals were reduced through the enclosing
  level's `_train` list (relative to depth-2's train part) instead of its
  absolute `_reduced_train`; correct at depth 1 only because depth 0 aliases
  the two. Library-only (the CLI nests two levels); identical at depth 1.
- **Search-layer edge cases** — two Debug-only off-by-one assertions
  (`get_parent_node()` at the leaf level aborted every Debug build of
  `bb-fast`/`bb-bbpp` as soon as a leaf was evaluated; `--search exhaustive
  --target-d 0` aborted on reaching cardinality n; release results were
  correct); SFFS/SFRS reset their per-cardinality solution table before the
  "unreachable target" early return, so `get_result()` can no longer hand out
  a previous run's subsets; a seeded SFFS (`enable_initial_subset()`) that
  backtracks below its seed's size now records the first subset it meets at
  a never-visited cardinality and stops there instead of continuing to
  backtrack without moving its pivot; OS/DOS on the library path without an
  initial subset switch the Subset to forward mode before clearing it (the
  fix the other sequential searches received in 4.2). Comments corrected: the
  `Subset::stateless_copy()` warning claimed a backward-mode subset receives
  an inverted selection (membership is the marker's sign in both modes, as
  every backward search relies on), the hybrid step's "keep_perc==100 behaves
  like the straight step" (same candidate set and best value, but ties in the
  main criterion may pick a different subset — the shortlist arrives in
  filter order), the B&B brief's "optimizes cardinality if target_d==0" (B&B
  searches one fixed cardinality).
- **`--sffs-delta` documented as it works** — the bound applies in the
  search's OWN direction: SFFS stops adding features at target-d+delta, SBFS
  stops removing them at target-d-delta; the conditional backtracking steps
  are bounded by their improvement rule alone (SFFS may still backtrack below
  target-d-delta), and SFRS's retreat always runs to the hard bound. The
  `set_delta()` documentation, the in-code comments and the `--help` text
  promised a symmetric target-d±delta window, which neither FST4 nor the
  paper's standalone implementation (arXiv:2608.01502) ever had.
- **Data-layer minor items** — CSV lines of pure delimiter residue (`,,,`,
  which Excel emits for deleted rows) are skipped as blank, as the loader's
  comment always claimed (a full-width one was reported as a missing-value
  row, a short one was fatal); the `randrand` class documentation states the
  actual test-part size when the percentages sum to 100 (the whole complement
  of the train part); its Debug precondition loop covers the closed interval
  it checks and the intervaller's reduce() asserts the strict invariant; the
  Model_Normal/Model_Multinomial documentation says where the class prior
  comes from (the whole data set's class sizes, not the train part — identical
  under stratified splitting); `demo40t.cfg` and `demo52t.cfg` carry the
  runtime note the demo section promises.
- **`--search2 bif` seeds every sequential search** — the BIF pre-ranking
  was computed and printed, but only the oscillating searches (os/dos) start
  from a supplied subset by construction; sfs/sffs/sfrs silently cleared the
  seed and ran unseeded (the n pre-search evaluations and any
  `--svm-optimize`/`--rbfsvm-optimize` re-tune on the seed were wasted, and
  the result was byte-identical to the unseeded run). The CLI now enables the
  initial subset on all sequential searches when `--search2` is given, with
  validity rules for the seed size: sfs needs `--search2-d` below
  `--target-d` (it extends the seed), sbs above (it shrinks it), the floating
  and retreating searches need it on their side of the target or equal to it.
  New goldens `fst4_search2_sfs` (the seeded SFS finds a better subset than
  the unseeded run on WDBC) and `fst4_search2_sffs`.
- **Every native classifier validates** — `--validate` accepted knn, bayes,
  logreg, multinom and svm only, although lda, rf, linsvm, rbfsvm and extern
  are built by the same factory (the introduction's `--validate
  knn,bayes,lda` failed with "unknown classifier"). All ten are admitted
  (extern needs `--extern-cmd`); each takes its own parameter flags. New
  golden `fst4_validate_native`. Help texts corrected on the way: `--knn-k`
  also sets k for the knn validator, `--hybrid-prefilter` lists
  `multinom-mi`, `--missing-value`/`--csv-missing-to` substitute the
  per-feature average of the whole data set by default and of the depth-0
  TRAIN part under `--scaler-fit train` (the help said "training average"
  unconditionally).
- **Standardization and LDA are translation-invariant in practice** — the
  shared per-feature standardization of logreg/linsvm/rbfsvm/ridge computed
  the variance as `E[x²]-mean²` in one pass, and LDA formed its discriminant
  from the raw means: both lose digits catastrophically once |mean| ≫ sd.
  Identical Gaussian data with and without a +1e8 offset gave different
  criterion values from every one of these wrappers (LDA fell from 0.73 to
  0.60 accuracy). The variance is now a second, centered pass over the
  in-memory design matrix and the LDA discriminant is formed relative to the
  prior-weighted mean of the class means (same decisions, same per-sample
  score differences). On the offset probe all values are now identical with
  and without the offset (ridge's R² to 9 digits — the floating-point limit
  of representing 1e8+O(1) values). Goldens: only the three `--report
  detailed` scenarios move, and only in their ROC/PR threshold columns'
  10th–12th significant digits (`fst4_report_detailed`,
  `fst4_report_multiclass`, `fst4_report_tuned`, recaptured); every subset,
  criterion value and accuracy is unchanged.
- **The load summary prints classes 0-based** — `Class 0 size: ...`, like
  every other index FST4 prints or accepts: feature indices in every subset
  print, the confusion-matrix rows, `--f1-positive` and `--class-weights` all
  count from 0, while the summary counted from 1 (a user reading "Class 1
  size" and setting `--f1-positive 1` targeted the second class). The one
  index that stays 1-based is the `classified,<row>` line of
  `--classify-samples`, which numbers the input file's rows — the CLI
  reference and help say so explicitly. Moves the 8 goldens that run at
  normal verbosity (`demo10`, `demo52t`, `demo55`, the two smoke tests,
  `fst4_verbose_demo10`, `fst4_progress_json`, `fst4_progress_quiet`;
  recaptured), one line per class each.
- **Documentation corrected** — `FST4_INTRO.html`: the hero command is
  runnable as pasted (its comments no longer sit after the line-continuation
  backslashes), the scaling sentence says that training-part-only
  normalization is `--scaler-fit train` (the default fits on the whole data
  set), the overfitting-curve claim is scoped to single-search runs with
  `--validate`, and the golden-suite count is current; `doc/FST4_CLI.md`: a
  section for the 17 flags the reference never mentioned (`--sffs-delta`,
  the LibSVM parameters, `--top-k`, `--tracker-capacity`, `--bb-gamma/-delta`,
  `--search2-d`, `--mc-seconds`, `--mc-card-prob`, `--hybrid-keep-*`,
  `--daf-count`, `--daf-pairs-top`, `--feature-costs`, `--restarts-max`),
  the CSV header rule as implemented (the class column is exempt from the
  numeric test), `multinom-mi` in the criteria row, `demo44.cfg` in the demo
  list; `README.md` no longer points at the unshipped `doc/html`; the model
  class documentation names the data the class prior comes from.

## Reliability: the live-code cleanup pass

A follow-up to the comment cleanup: while the shipped headers were being read
comment by comment, a list of live-code oddities accumulated — missing guards,
dead or vacuous statements, output strings that did not say what they meant,
and include/const/naming hygiene. None of them changed a result, which is why
they were deferred; all are fixed now, under the golden gate, in four commits
(guards, dead code, output strings, hygiene). The fast tier is byte-identical
after every one of them and the full suite, slow tier included, at the end.

- **Missing guards** — the plain-accuracy wrapper path and the bias estimator
  divided by the split count without checking it (the metric path did); a
  Monte Carlo search asked for `--target-d 0` without a cardinality range
  collapsed its size range to [n,n], i.e. sampled the full feature set over
  and over (and tripped an assert in Debug builds) — it now draws probe sizes
  from [1,n], as its documentation promised; the sequential candidate
  evaluator handed out a best result after a failed evaluation while the
  threaded one refused (both refuse now); three members were read before
  ever being written (the void scaler's statistics, `Subset`'s
  temporary-feature cursor, the CV splitter's fold counter); the CV
  splitter's copy constructor copied in-progress split state that no other
  splitter copies; the ARFF row reader indexed a class-size table that does
  not exist for files without a CLASS attribute; `Subset`'s range
  `select`/`deselect` asserted against the raw size where the single-index
  overloads use the frozen-adjusted one.
- **Dead and vacuous code** — some ninety `>= 0` comparisons on unsigned
  operands in asserts across 27 headers (each paired upper bound stays); an `||` that had
  to be `&&` in the multinomial model's precondition; a negative test on an
  unsigned index in the neighbours regressor; the TRN header parser's
  never-set error flag with its three dead guards and its ignored `fgets`
  result (a file ending inside the header is now a distinct load error, code
  12); an unreachable return in the intervaller's validity check; a dead
  rescale branch in the stability tracker; a disabled `if(true)` switch and a
  member used as a loop cursor in the LibSVM wrapper; an `if(false)` debug
  print in the memory accessor; an assigned-never-read class count in the
  ridge regressor; an unused enumerator and two unsynchronized, uncalled
  counters in the threaded evaluator; a never-used trailing slot in four DAF
  work vectors; the hybrid step's keep-max asserts now agree with each other
  (`> 0`, as documented); the duplicate assert in the sum-of-weights
  criterion.
- **Output strings** — the ranking search's finish line names the concrete
  search (it said `Search_Ranking()` for BIF too); the DAF warning's grammar;
  the logistic-regression size-guard exception no longer names a CLI flag
  from library code; the RBF-SVM optimize log's gamma values are documented
  as directly comparable across its lines.
- **Hygiene** — three missing includes (`<vector>` on the Windows path of
  the external bridge, `<cctype>`/`<cstdlib>` in the ARFF reader, the scaler
  header in the memory accessor) and ten unused ones (`<cstdlib>`/`<ctime>`
  in the five Branch & Bound headers); five out-of-class definitions that
  dropped a declaration's top-level `const`; `Indexed_Vector` gained
  read-only accessors (`operator[]`, `at_raw`, `at`, `get_raw` in const
  form), so const callers such as LDA stay const; kNN collects exactly the k
  neighbours it votes over instead of `(k-1)·classes+1` (identical votes: the
  earlier-seen neighbour wins a distance tie either way); `Distance::reset()`
  leaves all features active, so the `Distance_*(n)` constructors yield a
  usable object without a `narrow_to()`; `std::pow` qualified in the Lp
  distance; the `M_PI` fallback comment unified; the `Data_Scaler` forward
  declarations name their parameter `DATATYPE`; an empty `protected:`
  section in the sequential step base; the TRN parser's Czech identifiers
  renamed in English; one indentation style in the neighbours regressor.
  Deliberately NOT changed: the ARFF/CSV readers' load diagnostics still go
  to `std::cout` unconditionally — the readers carry no verbosity setting,
  and the `fst4` tool already routes them with the rest of the prose (silent
  runs and the machine output formats never see them).

## Distribution and repository shape (4.2 housecleaning)

- **License: AGPL-3.0 with a commercial option.** FST4 is released under
  the GNU Affero General Public License, version 3 — `LICENSE` carries the
  full text, and every source file's header states the terms by reference —
  which retires the non-commercial-use terms of FST3. Closed-source use
  without the AGPL's obligations needs a commercial license from ÚTIA (in
  case of interest, write to utia@utia.cas.cz), and contributions require a
  contributor license agreement so that the dual-licensing model survives
  outside contributions. `LICENSING.md` explains the model and gives the
  contacts (licensing: utia@utia.cas.cz; technical questions and pull
  requests: somol@utia.cas.cz with "FST" in the subject line).
- The main readme is `README.md` (markdown; was `readme.txt`), with the
  third-party section stating explicitly that LibSVM and Eigen are OPTIONAL
  — a default build needs neither — alongside the CLI11 notice duty.
- All 51 runnable demo configs and the three exemplar sources live in
  `demos/` (`./fst4 --config demos/demoNN.cfg` from the distribution root);
  the demo-numbering audit confirmed the FST3 numbering survives intact for
  every parity demo, with the FST4-native additions slotted into the
  topical decades where space allowed (documented exceptions: demo38,
  demo44; the external bridge moved to demo64 into its proper group).
- **Bundled sample data sets, with attribution.** The 15 data files the
  demos and golden scenarios use ship WITH the toolbox rather than as a
  separate download. `data/NOTICE` records, per file, the creators, the
  citation, the DOI where one exists, the licence, and the modifications
  FST4 applied — the statement CC BY 4.0 §3(a)(1) requires — and each data
  file repeats its origin and terms in its own header, so a file separated
  from the package still carries them (exception: the `.csv` files — the
  format has no comment syntax, so NOTICE is their attribution record).
  `README.md` gained a "Bundled sample data sets" section covering the same
  ground for a reader who never opens `data/`. Licence basis: the nine UCI
  sets are CC BY 4.0; `reuters_apte.arff` is Reuters Ltd. "research purposes
  only" (fully anonymized; those terms are the data set's own, stated in
  `data/NOTICE` and README, and do not follow from FST4's software license).
  `data/` now holds exactly the distribution set — every file in it is
  referenced by a demo config or a golden scenario; the 13 working files that
  were not (compression duplicates, unreferenced sets, research scratch)
  moved out of the distribution.
- **The historical speech data set is NOT distributed.** FST3 and the FST4
  development line long used a British Telecom speech data set
  (`speech_15.trn`; 1418 samples, 15 features, 2 classes) as the default
  demo data. It leaves the distribution with 4.2; every demo config, demo
  exemplar and golden scenario that used it now runs on `wdbc_30.trn`
  (Breast Cancer Wisconsin Diagnostic, CC BY 4.0) instead. Demo protocols
  are otherwise unchanged, with ONE forced exception: demo40/demo40t's
  exhaustive search — all subset sizes was feasible at 15 features but is
  not at 30, so those demos (and their golden scenario) run exhaustive
  search bounded to `--target-d 5` (C(30,5) = 142,506 subsets). Results
  quoted for the speech protocols in FST3-era materials are therefore not
  reproducible from this distribution's demos.
- **Data naming and format hygiene.** `waveform-5000.arff` is renamed
  `waveform_40.arff` — it is the same 5000-sample, 40-feature data as
  `waveform_40.trn` (verified per class), so the two format variants now
  share the FST naming convention (basename + dimensionality). The sonar
  TRN files carried a WRONG class split from their historical conversion
  (103 rocks / 105 mines, mislabeling six samples): both `sonar_60.trn`
  and `sonar_60_missing_data.trn` are corrected to the official UCI split
  (97 rocks / 111 mines) — the ARFF always had the correct labels and is
  renamed `sonar_60.arff` (was `sonar.arff`), the two sonar format variants
  now being verified identical. FST3-published numbers for the sonar demos
  (22/23/56/62) are therefore not reproducible from this distribution
  either — correctness over parity. New
  `wdbc_30.csv` and `spambase_57.csv` are CSV conversions of the
  corresponding `.trn` files (same values, same sample order); `demo15.cfg`
  and `demo38.cfg` read them, demonstrating the 4.2 CSV loader on real data
  with results identical to the TRN versions.
- Build trees are per-platform by convention (`build-win`, `build-mac`,
  `build-linux`) and the golden baselines are per-platform directories
  auto-selected by `tests/golden.sh` from `uname`. **Three baselines ship —
  `tests/golden-win-x64`, `tests/golden-macos-arm64` and
  `tests/golden-linux-arm64` — each complete at the full 112 scenarios and
  each verified green on its own platform's binaries, slow tier included.**
  `FST_GOLDEN_DIR` overrides the choice; a platform with no baseline of its
  own captures one and points there.
  Measured agreement: Linux arm64 and macOS arm64 are byte-for-byte identical
  on 111 of the 112 scenarios, and both differ from Windows x64 on two
  (`fst4_report_multiclass`, `fst4_report_tuned`) — in every case only in
  low-order digits of a `--report detailed` ROC/PR line, where an iterative
  solve printed at ~12 significant digits reaches a marginally different
  fixed point per architecture. Selected subsets, criterion values, sizes,
  rankings and every other output line agree everywhere. The per-platform split is kept as
  insurance for platforms not yet measured, not as a response to observed
  divergence.
- The golden harness takes `FST_LASTRUN=<dir>` to relocate its scratch
  directory, so two gate runs can share one checkout (a container and its
  host, say) without clobbering each other's intermediates. Output is
  normalized back to the canonical path, so goldens stay byte-identical and
  machine-independent wherever the scratch directory lives.
- The development records (audit ledgers, design records, review findings,
  milestone plans) are kept out of the distribution, so the shipped tree
  carries only what a user needs: `doc/` for the reference documentation,
  `demos/` for runnable configs, `tests/` for the verification harness.

# Feature Selection Toolbox 4 — 4.1 Release Notes (internal milestone)

> 4.1 was an internal development milestone and never shipped as a separate
> public release; its changes reach users as part of v4.2. The section is
> kept per-version to keep the development lineage clear.

- **Threaded sampled evaluator** — `--evaluator sampled` now accepts `--threads N`:
  each step's proposal batch (and the `--warmup-probes` loop) is evaluated in a
  thread pool through the injected `Candidate_Evaluator` engine. Determinism
  semantics: the proposal
  sampling stays on the master thread's RNG stream (every `fst_rand()` draw of a
  step precedes its first evaluation, and criterion `evaluate()` never draws), and
  results are replayed to the per-feature statistics engine, tracker and
  best-candidate selection in proposal order at batch completion. Threaded runs are
  therefore **bit-identical to `--threads 1` at matched `--seed`, for any thread
  count** — verified on madelon and speech data across samplers
  (softmax/uniform/topk), forward and backward budgeted steps, warm-up, and
  wrapper/filter criteria (the speech verification data set left the
  distribution in 4.2 — see the data notes; the corresponding golden
  scenarios now pin the same protocols on WDBC). The sequential path is
  unchanged (golden-gated).
- **Sampled-step proposal sampler: out-of-bounds access on complete softmax
  underflow — FIXED** (pre-existing since the sampled step's introduction; found by
  the threading validation). With the automatic temperature at its 1e-9 floor
  (score IQR == 0 — cold or fully tied statistics, e.g. backward steps with no
  warm-up), `exp(Δ/tau)` underflows to +0.0 for every candidate off the exact-best
  ties; once those were drawn, neither the weighted-draw loop nor the
  numeric-drift guard (both require w>0) could select, and the NDEBUG'd assert let
  `pick==w.size()` through to `cand[pick]`/`w[pick]` — an out-of-bounds read AND
  write. The heap garbage produced duplicated, environment-layout-dependent
  proposals: same binary/seed/flags gave different search trajectories depending
  on process environment size, and threaded runs (different heap layout) diverged
  from sequential on this corner. Fixed with a deterministic degenerate branch:
  when all remaining weights are zero, pick the tau→0 softmax limit
  (argmax(score) forward / argmin(score) backward, lowest index wins ties;
  drawn entries carry a negative sentinel). Healthy trajectories are bit-identical
  to before the fix; affected runs are now
  environment-invariant and bit-identical sequential-vs-threaded — the threaded
  determinism guarantee above now holds without exceptions.
- **Multinomial logistic regression classifier** — `classifier_logreg.hpp`
  (`Classifier_Logistic_Regression`), a fifth wrapper classifier:
  `--criterion wrapper-logreg` and `--validate logreg`, dependency-free (the
  Newton system is solved by an internal Cholesky factorization). One softmax
  model over all classes (bias per class); training minimizes mean cross-entropy
  + `--logreg-lambda`/2·‖weights‖² (default 0.01, biases unpenalized) by damped
  (step-halving) Newton iterations from zero-initialized weights, stopping at
  gradient infinity-norm < `--logreg-tol` (default 1e-6) or `--logreg-maxiter`
  (default 100). Features are standardized internally per `train()` on the
  current split's TRAIN part, so unscaled wrapper data conditions well and
  constant features are inert (their weight stays exactly 0) rather than fatal
  as in the normal-model filters. Training consumes no RNG and traverses data
  in fixed order: runs are byte-identical at any `--seed` and `--threads`
  (clone-per-worker; verified under the straight and sampled threaded
  evaluators). Cost note: each iteration assembles a
  K(d+1)×K(d+1) Hessian — many classes combined with large subsets is the
  expensive corner (guarded with an actionable error above K(d+1)=8192).
  Golden-pinned by the `fst4_logreg` (two-class, WDBC SFFS) and
  `fst4_logreg_multiclass` (3-class waveform SFS + `--validate logreg`)
  scenarios (captured on both platforms).
- **FST3 demo parity configs** — every demo published in the public FST3 at
  fst.utia.cz exists as a ready-to-run config file, named after the demo it
  reproduces (`demo10.cfg` … `demo63.cfg`, 36 files; since the 4.2
  housecleaning they live in `demos/` together with the exemplar sources:
  `./fst4 --config demos/demoNN.cfg` from the repo root).
  Parameters follow the golden-verified `fst4_*`
  scenarios; each header states the protocol and runtime warnings. See the
  "FST3 demo parity configs" section in `doc/FST4_CLI.md`.
- **DAF consolidation: ONE implementation, the published 2011 method** — the
  experimental "new-generation" DAF stack (`Search_DAF_Ranking`, the
  `DAF_State_Impl0/Impl1/Impl_Pair0` state classes, result history and stopping
  machinery) and its demo36/demo37 exercisers were retired entirely. These
  were internal-only experiments never published in the public FST3: demo36's
  DAF1 implementation silently degraded to a dummy identity ranking on
  insufficient probe coverage, and demo37's pair0 ranking was an unfinished,
  unpublished greedy heuristic — not the 2011 DAF. Dependency-Aware Feature
  ranking in FST4 is exactly the published protocol: any probe-generating search
  (typically `--search montecarlo`) + `--tracker feature-stats` + `--daf-rank 0|1`
  (demo34/demo35t). CLI surface removed: `--search daf`, `--daf-state`,
  `--daf-trials` (with the `fst4_daf_new`/`fst4_daf_demo37` golden scenarios).
- **Pairwise (group) DAF statistics** — `--daf-pairs K` (+ `--daf-pairs-top M`)
  computes 2×2 presence-cell contrasts over all pairs of the top-K DAF0-ranked
  features from the feature-stats tracker's stored probes: pair-DAF0
  (mean(both present) − mean(both absent)), pair-DAF1 (the frequency-weighted
  stdev-normalized analogue) and synergy (the 2×2 interaction contrast). This
  adopts the retired pair0 testbed's pure-conditioning idea but exposes the
  statistics only — no ranking heuristic is baked in. Implementation is a pure
  post-processing pass (search logic untouched): only the both-present cell is
  accumulated per probe; the other cells derive by inclusion–exclusion, so cost
  is O(probes·K + probes·p²) for p top-K features present per probe, and the
  top-K restriction keeps memory at K(K−1)/2 entries. Per-feature DAF outputs
  are byte-identical with the option off (golden-gated); pinned by the new
  `fst4_daf_pairs` scenario.
- **demo14 retired; new FST4-native demo configs** — the unfinished
  "(to be Stochastic) Floating Search" draft (demo14.cpp with its
  `seq_step_weighted.hpp`/`result_weighter_deterministic.hpp` draft headers,
  never in the public FST3) is fully replaced by the sampled evaluator
  (`--step-stats rawmean` preserves the draft's statistic as an ablation). Three new demo configs reuse freed numbers and are
  explicitly NOT FST3 parity configs: `demo13t.cfg` (threaded logistic-regression
  wrapper on WDBC), `demo36.cfg` (the stochastic-floating-search paper's gisette
  protocol: sampled SFFS, 1-NN wrapper, d=300) and `demo37t.cfg` (the paper's
  reuters extreme-d protocol: sampled SFFS at d=500 over 10,105 features,
  8 threads, bit-identical to sequential). The compiled exemplars are now
  demo10, demo52t, demo55. Supporting these, `--sffs-delta N` exposes
  `Search_SFFS::set_delta` (present in the library since FST3, never used by a
  demo): with a nonzero `--target-d` the floating frontier stops at
  target-d+delta and the best tracked subset of the target size is reported —
  without it a floating search always runs the full frontier, which is exactly
  what makes the classic protocol infeasible at 5000-10105 features (the paper
  campaign used external frontier-kill watchdogs for the same purpose). Default
  0 preserves the classic behavior; pinned by the `fst4_sffs_delta` golden.
- **Deterministic threaded streaming (Monte Carlo, exhaustive)** — the threaded
  engine's `evaluate_stream()` was reworked to the sampled-evaluator recipe
  : the master thread generates all candidates
  into a bounded ring (randomness rides the master `fst_rand` stream), workers
  evaluate criterion clones concurrently, and the master replays completed slots
  strictly in generation order (tracker adds, best-updates, progress callbacks).
  `--search montecarlo` and `--search exhaustive` with `--threads N` are now
  **bit-identical to sequential at matched `--seed` for any thread count** —
  including the full verbose progress prose (`trial=` indices included); only
  the engine-name echo lines and the `"threads"` invocation-echo field differ.
  This makes the DAF probe pipeline (montecarlo + feature-stats) deterministic
  under threading, where the former scheme (workers pulling the generator under
  a lock) drew from worker-thread_local RNG streams and was not run-reproducible
  for randomized generators. Pinned by the `fst4_daf_pairs_threads` golden (the
  `fst4_daf_pairs` twin). Library-level compatibility note: the
  `Candidate_Evaluator::BestCallback` signature gained a generation-ordinal
  parameter `(int worker, unsigned long ordinal, value, candidate)`; both engines
  now fire it on the master in generation order with worker index −1.
- **Cross-platform seed-for-seed reproducibility: the FST-owned PRNG shim and
  the arch-guarded build flags** (`fst_rand.hpp`) — all randomness in FST4
  code goes through `FST::fst_rand()`/`fst_srand()`, a UCRT-compatible
  generator with `thread_local` state (an unseeded run equals `--seed 1`,
  which is why the CLI scenarios reproduce the historical demos' unseeded
  `rand()` results exactly). Seeded runs are reproducible seed-for-seed
  ACROSS PLATFORMS (Windows/UCRT and macOS/arm64 produce the same draw
  sequences; vendored LibSVM is the deliberate untouched exception). On the
  build side, the CMake arch guard kept the x86 release flags byte-parity
  critical and untouched (`-O2 -march=native -funroll-loops -mfpmath=sse,387`
  — golden byte-parity depends on identical FP codegen) while arm64 builds
  got fixed portable flags with contraction off; the macOS golden baseline
  (`tests/golden-macos-arm64`) is captured under exactly those flags.
  (**Superseded in 4.2** by the two-preset policy: a default build now uses
  the same fixed portable flags on every architecture, and the native flags
  moved behind `-DFST4_NATIVE_OPT=ON`. See "Reproducible-by-default builds"
  in the 4.2 notes above.) New
  code must draw through the shim — never `rand()` or `<random>` directly.

# Feature Selection Toolbox 4 — 4.0.0 Release Notes (internal milestone)

> 4.0 was an internal development milestone and never shipped as a separate
> public release; its changes reach users as part of v4.2 (the last public
> release before v4.2 was FST3).

Feature Selection Toolbox 4 (FST4) is the first stable release since FST3
3.1.1.beta, and the largest change to the toolbox since FST3 itself. The major
version reflects compatibility-relevant changes, not internals: the toolchain
requirement, the dependency set, the distribution model, and the primary
interfaces all changed. The algorithmic core — fifteen years of numerically
validated feature-selection methods — is preserved and verified byte-exact
against the 3.1.1.beta outputs throughout.

## Headlines

- **`fst4` command-line tool** — the whole toolbox is usable without writing C++:
  every released demo scenario is expressible as flags or a config file
  (`doc/FST4_CLI.md`; component registry via `fst4 --list`). Includes machine-readable
  output (`--output json|csv`), search-progress streaming (`--progress json`),
  reproducible runs (`--seed`, `--stable-output`), and cross-scenario similarity
  (`--compare-config`).
- **Header-only library** — include the headers, done. No compiled translation unit,
  no Boost, no external dependencies (LibSVM and Eigen optional).
- **C++17** — the C++98/Boost era ends; smart pointers and threading come from the
  standard library. GCC, MinGW-W64 and MSVC (C++17 mode) are supported.
- **CMake build** (`cmake -S . -B build && cmake --build build`) with exact
  per-translation-unit dependency tracking. The historical Makefile is retired.
- **Numerics cross-validated** — an optional Eigen-backed `Indexed_Matrix` build
  (`-DFST4_USE_EIGEN=ON`) reproduces the hand-rolled LU/inverse/determinant results
  to print precision on all normal-model criteria, divergence included.
- **A full-codebase audit, fully adjudicated** — some 46 recorded findings:
  silent-wrong-results defects, crashes and undefined behavior,
  clone-contract violations, data-I/O parser defects, numerics edge cases,
  thread-safety hazards, DEBUG-build breakage. Every finding is either fixed
  and regression-gated, or explicitly adjudicated with rationale. Nothing
  remains silently open.
- **Golden-output regression harness** (`tests/`) — the release gate: byte-identical
  output verification across the full demo and CLI scenario set, on every change.

## The `fst4` command-line tool

New in 4.0, the `fst4` binary exposes the complete component registry at runtime:
all data formats (TRN, ARFF), splitters (CV, holdout, leave-one-out, random,
including nested `--splitter2` stacks with an independent `--validate` holdout),
scalers, filter criteria (normal-model Bhattacharyya/Divergence/Generalized
Mahalanobis, multinomial), wrapper criteria (kNN with pluggable distances,
Normal Bayes, multinomial naive Bayes, SVM via bundled LibSVM, kNN-based
regression), all search algorithms (SFS/SBS, floating, retreating, oscillating,
dynamic oscillating, exhaustive, Monte Carlo, four Branch & Bound variants, BIF
and both generations of dependency-aware DAF ranking), hybrid and ensemble step
evaluators, and all result trackers.

The demo run protocols are first-class flags: stability trials over repeated
splits (`--repeat-over-splits`), non-improving-restart randomized search,
regularization and margin sweeps, BIF-seeded search (`--search2 bif`), SVM
parameter optimization, bias estimation, sample classification, and config
files (`--config`). `--compare-config` runs a second
scenario in the same process and reports cross-scenario feature-preference
similarity (IATI/ICW/IANHI) — the demo55 protocol.

Determinism is a contract: `--threads` defaults to 1, `--seed 1` reproduces the
historical unseeded demos exactly, and `--stable-output` strips timing so runs
are byte-comparable. The JSON/CSV output echoes the fully resolved invocation
(every component and parameter with defaults filled in, plus the seed), so any
published result is reproducible from its own output file.

## Toolchain, dependencies, distribution

- **C++17 required** (was C++98 + Boost). All Boost usage — smart pointers,
  threading — was replaced by the standard library. Boost is no longer needed
  in any configuration.
- **Header-only**: the former `global.cpp` translation unit is an empty
  compatibility stub; new consumers just include headers. Visual Studio
  projects need no external include or library paths.
- **CMake** is the build system (3.16+). Release flags replicate the historical
  optimization flags exactly, preserving floating-point codegen and therefore
  result reproducibility. Header dependencies are tracked automatically per
  translation unit, retiring a historical Makefile defect where header edits
  could silently re-link stale objects.
- **LibSVM 3.37** is supported (up from 3.11; drop-in compatible — FST4
  touches only the stable public API); since 4.2 it is an optional drop-in
  that is not distributed with FST4. `-DFST4_WITH_SVM=OFF` builds without it.
- **CLI11 2.4.2** (BSD-3) is vendored for the CLI, license file intact.
  Third-party licensing is documented in the distribution.
- **Optional Eigen backend**: `-DFST4_USE_EIGEN=ON -DFST4_EIGEN_DIR=<eigen-root>`
  swaps `Indexed_Matrix` LU/inverse/determinant to `Eigen::PartialPivLU`
  (`EIGEN_MPL2_ONLY` enforced). Intended for verification builds; the
  zero-dependency hand-rolled path remains the default.

## Architectural changes

The 4.0 program restructured the library without
changing its results — every step was gated on byte-identical golden outputs.

- **Duplication eliminated.** The threaded search algorithms were previously
  near-copies of their sequential twins, a proven source of drift bugs. Each
  search now has one body; sequential vs. thread-pooled execution is chosen by
  an injected `Candidate_Evaluator` engine. The two-class normal-model criteria
  share a template-method skeleton (`Criterion_Normal`); the data splitters
  share the split-iteration scaffolding (`Data_Splitter`).
- **The clone contract is now structural.** Thread-safe cloning was previously
  hand-written per class and was the library's largest audited defect class
  (five broken `clone()` implementations shipped in 3.1.1.beta). All concrete
  clonable classes now derive from a CRTP mixin (`Clonable_Impl`) that
  implements the clone protocol uniformly — this defect class can no longer be
  written.
- **Stateful cursor protocols got safe replacements.** Feature iteration
  (`Subset::selected_features()`/`temporary_features()`) and data-block
  iteration (`blocks(DataPart)`) are now reentrant RAII ranges; the legacy
  one-shared-cursor-per-object APIs remain available but deprecated (below).
- **Search progress is observable.** A `Search_Observer` seam delivers progress
  and improved-solution events from all search algorithms; the CLI's
  `--progress text|json|quiet` is built on it. With no observer attached, the
  historical text output is preserved byte-exactly.
- **Runtime-sized thread pool** — thread counts are constructor arguments, not
  template parameters.
- **Numerical trust.** The hand-rolled Crout LU/inverse/determinant code was
  cross-validated against Eigen: on the reference data, all three normal-model
  criteria produce identical subsets and criterion values to 12 significant
  digits under either backend.

## Reliability: the audit

A full review of the library sources (~24k lines) plus representative demos was
conducted at the start of the program and tracked in a defect ledger; release-hardening
policy required every finding fixed or explicitly adjudicated before tagging.
Highlights of what 3.x users were exposed to and 4.0 fixes:

**Silent wrong results.** The Normal Bayes classifier never subtracted the
class mean from the pattern (wrong results whenever class means differ — since
the initial import); it also lacked log-domain math, so on high-dimensional or
distant patterns probabilities underflowed to zero and classification was
systematically biased toward class 0 (now a guarded log-domain fallback,
byte-identical when the plain path is healthy). ARFF nominal attributes were
read as bit-pattern garbage; ARFF STRING attributes silently produced garbage
(now rejected loudly); indented `@DATA`/`@ATTRIBUTE` lines caused files to
"load" with zero records; TRN files without `#data` shifted all values by one
row. The whitening scaler corrupted missing values. The regression wrapper
routed dependent-variable indexes > 0 into a classification branch. The
threaded exhaustive search could return an uninitialized best value when
threads outnumbered candidates.

**Crashes, hangs, undefined behavior.** Every Subset feature-iterator loop read
one element past the end (executed in essentially every search); a lost-wakeup
race in the threaded candidate evaluator could hang any threaded search;
ThreadPool had formal data races; sparse ARFF files with class value 0
crashed the loader; several `new[]`/`delete` mismatches and buffer-reuse
hazards (model/matrix reuse on a second, smaller dataset; SVM index arrays
never regrown) were fixed. `get_combinations()` overflowed 32-bit factorials,
which had made threaded generalized sequential steps unusable beyond toy
dimensions.

**Clone-contract violations.** Five `clone()` implementations produced clones
with null or uninitialized members, crashing threaded searches and tracker
regularization on first use. Fixed directly, then the whole defect class was
eliminated structurally (see Architectural changes).

**DEBUG builds and diagnostics.** Several `-D DEBUG` translation units did not
compile or aborted on spurious asserts; all fixed. All demos previously exited
with code 0 even on fatal errors (one demo had "passed" for years while never
loading its data file); they now exit nonzero, and the missing dataset was
restored.

**Adjudicated as intended, with rationale recorded:** the default
(non-randomized) splitter path deliberately keeps the historical unseeded
`rand()` sequence — it is pinned by the golden outputs and the CLI `--seed`
contract; the randomized path now uses a correctly per-instance-seeded
`std::mt19937`. Data scaling is learned over the entire dataset before
splitting (documented methodological leakage, unchanged for compatibility).
The known LibSVM LINEAR/SIGMOID-kernel slowness and the Divergence criterion's
documented numerical sensitivity remain documented cautions (the latter did
not reproduce in the Eigen cross-validation).

### A fix inherited from a pre-release audit

One important fix predates the 4.0 program and is credited separately: an
incorrect lock in the threaded search infrastructure could freeze
multi-threaded runs after very long computations — a symptom observed in
practice years earlier. It was found and fixed in a December 2025 audit of the
published FST3 codebase performed with Claude Opus 4.5 (unsynchronized console
output replaced with synchronized printing; mutex protection added to shared
iteration state in the threaded ranking, feature-statistics and DAF search
paths). The 4.0 program started from a codebase that already contained this
fix. Independently of it, the present release fixed a lost-wakeup race in the
threaded candidate evaluator that could produce the same
stall symptom.

## Compatibility notes (what a 3.x user must know)

- **C++17 required** (was: C++98 + Boost). Boost is no longer used at all.
- **Behavioral fixes change results for previously-affected inputs**: the normal
  Bayes classifier no longer collapses to class 0 on numerically hard patterns
  (log-domain fallback); indented/whitespace-bearing ARFF files now parse
  correctly; TRN files without `#data` load correctly; ARFF STRING attributes are
  rejected loudly instead of silently producing garbage. Results on well-formed
  inputs are unchanged (verified byte-exact against the 3.1.1.beta golden outputs).
- **`global.cpp` no longer needs to be compiled** — it remains as an empty stub, so
  existing project files keep working.
- **Demos exit nonzero on fatal errors** (previously always 0) — scripts that
  tested demo exit codes will now see failures reported correctly.
- **Compiled demos are reduced to three exemplars** (see Demo reorganization).

## Deprecated in 4.0 — removal planned for the next major release

- `Subset::getFirstFeature` / `getNextFeature` (and the temporary-feature cursor) —
  use the reentrant `selected_features()` / `temporary_features()` ranges.
- `Data_Accessor::getFirstBlock` / `getNextBlock` with manual loopdepth constants —
  use the RAII `blocks(DataPart)` range, which owns its loop slot.
- The threaded compatibility shims `Sequential_Step_Straight_Threaded`,
  `Search_Exhaustive_Threaded`, `Search_Monte_Carlo_Threaded`,
  `Search_Ranking_BIF_Threaded`, `Search_Branch_And_Bound_Improved_Threaded` —
  construct the merged class with a `Candidate_Evaluator_Threaded` engine instead.

All deprecated APIs are fully functional in 4.0 and carry `\deprecated` markers in
the headers.

## Demo reorganization

The 39 compiled demos of FST3 are reduced to three library-API exemplars:
demo10 (simplest filter workflow), demo52t (threaded wrapper criterion with
tracker regularization) and demo55 (two-scenario stability/similarity
choreography). Every
retired demo's scenario ships instead as an `fst4` config-file scenario
producing the same results, verified byte-exact where the demo's output was
deterministic — including the long-running high-dimensional and optimal-search
scenarios on their original datasets. See `tests/manifest.tsv` for the scenario
registry and the ready-to-run configs in `demos/` for the per-demo command lines.

The per-demo configs are exposed explicitly as `demos/demoNN.cfg`, one per
demo published in the public FST3 (36 files) — see the 4.1 notes above.

## Quality gate

Every change in this release passed the golden-output regression harness
(`tests/`): byte-identical outputs across the full demo and CLI scenario set,
fast and slow tier. At the 4.2 release the gate stands at 112 scenarios per
platform, green on all three shipped baselines (Windows x64, macOS arm64,
Linux arm64).
