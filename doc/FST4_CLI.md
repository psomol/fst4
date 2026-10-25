# fst4 — the FST4 command-line tool

Exposes the Feature Selection Toolbox 4 library as a single executable: data file,
splitters, scaler, criterion, search algorithm, result tracking and the run protocol
are all selected via command-line flags, so the whole toolbox is usable without
writing any C++. Build it with CMake — `cmake --build <build-dir> --target fst4`;
see the build quickstart in `README.md`. When LibSVM is present the target compiles
with `-D FST_SVM` and links the drop-in (LibSVM is not distributed with FST4; fetch
it per `libsvm/readme.txt`). It is optional: since 4.2 the native `wrapper-linsvm`
and `wrapper-rbfsvm` criteria cover linear and RBF SVM selection without it.

## Synopsis

```
fst4 --data FILE --criterion NAME --search NAME [options]
fst4 --data - --format trn|arff|csv ...   # dataset from standard input
fst4 --features N --criterion extern-value --extern-no-data --extern-cmd CMD ...
                            # no data at all: black-box set-function optimization
fst4 --config FILE          # options from an INI/TOML-style file (CLI wins on conflict)
fst4 --list                 # print all available components with validity notes
fst4 --help                 # full flag reference
```

Results go to stdout as text (default), JSON or CSV (`--output`); with json/csv all
library progress prose is redirected to stderr so stdout stays machine-readable.
`--progress` controls search-progress delivery independently of the final
results: `text` (default, the classic log), `json` (one JSON line per improved
solution — `{"event":"solution","value":…,"d":…,"features":[…]}` — on the progress
stream; free-text progress is dropped; requires `--verbosity normal` or higher,
since observer events fire at the same points as the text log), or `quiet`
(suppress search progress entirely while keeping setup/result output). It is
implemented via the `Search_Observer` seam (`Search::set_observer()`), which
library users can implement directly for custom progress handling.
`--stable-output` omits timings so runs are byte-comparable (used by the golden
tests). Exit code 0 = success, 1 = any error (validity violations produce an
actionable message on stderr; library errors are relayed, some with added hints —
e.g. singular covariance matrices name the constant features responsible).

## The run shape (demo choreography)

Every run follows the protocol shared by the FST4 demos:

1. Data is loaded and split by the **outer splitter** (`--splitter`, splitting
   depth 0). Its train part feeds feature selection; its test part is reserved for
   final validation.
2. With `--splitter2` (splitting depth 1, e.g. `cv`), the criterion is estimated on
   inner splits of the outer train part — the demos' nested-splitter protocol.
   Without it the criterion runs directly on the depth-0 split.
3. The **search** selects a subset maximizing the **criterion** (`--target-d N`
   fixes the subset size; 0 lets wrapper criteria optimize it; monotonic filters
   and Branch & Bound require a nonzero value).
4. `--validate clf[,clf2]` finally trains each named classifier (knn, bayes,
   lda, logreg, rf, linsvm, rbfsvm, multinom, svm, extern, or `same` = the
   wrapper criterion's own; each takes its own `--knn-*`/`--logreg-*`/`--rf-*`/
   `--linsvm-*`/`--rbfsvm-*`/`--svm-*`/`--extern-cmd` settings) on the depth-0
   train part and tests it on the depth-0 test part.

Alternative run protocols replace step 3 (mutually exclusive):
`--repeat-over-splits` (one search per outer split — stability trials; combine
with `--splitter randrand --rr-splits N` and `--tracker stabileval`; a wrapper
criterion additionally needs `--splitter2`, because the outer splits ARE the
trials and the classifier's own train/test estimation must run on inner splits
of each trial's training part — the demo54/55 protocol; filters re-estimate
their model per trial and need no inner splitter),
`--restarts-nonimproving N` (random-restart search), `--regr-multipliers a,b,c`
(regression kernel sweep, best-by-train wins). Steps around the search:
`--search2 bif` (BIF pre-ranking seeds the search), `--svm-optimize
pre|alternating` (LibSVM parameter tuning), `--rbfsvm-optimize
pre|alternating` (the native RBF SVM C/gamma twin), `--bias-estimate`,
`--classify-samples FILE` (rows are full-length feature vectors, transformed
per feature exactly as the loaded data was — scaler and `--missing-value`
substitution — before the classifier trained on the depth-0 train part
predicts them), and the `--regularize-*` post-search sweep.

## Components

Run `fst4 --list` for the full registry with one-line validity notes. Summary:

| Group | Names |
|---|---|
| formats | trn, arff, csv (inferred from extension; `.tsv` also means csv). CSV is the simplest Excel-export path: delimiter auto-detected from line 1 (comma/semicolon/tab, override `--csv-delimiter`; decimal COMMAS accepted whenever the delimiter is not the comma, so European-locale Excel exports load with zero flags), header row auto-detected (override `--csv-header yes\|no`; the test is "any non-numeric field in line 1 OUTSIDE the class column" — the class column is exempt because its labels are arbitrary tokens, which is what lets header-less files with textual labels load; an all-numeric header row is therefore taken for data — force it with `--csv-header yes`), class column defaults to the LAST column (`--csv-class` takes a 0-based column index or a header column name; labels are arbitrary tokens matched as exact strings, so `1` and `1.0` count as two different classes; class ids in first-appearance order), Excel UTF-8 BOM and `""`-escaped double-quoting handled. Rows with missing values (`""`, `?`, `NA`, `N/A`, `NaN`, `NULL`) are skipped loudly — or KEPT with `--csv-missing-to <code>`, which maps missing FEATURE cells to a numeric code instead (a missing CLASS label still skips the row: it belongs to no class). The mapping is deliberately pure — pair it with `--missing-value <same code>` to substitute per-feature training averages (the demo62 protocol), or leave the code in place as an explicit missingness indicator a tree-based wrapper can split on. Guarded three ways: `--scaler void` only (to01/white would fold the code into the scaling statistics), the two codes must agree when both flags are given (a differing `--missing-value` would never match), and the load summary names the pairing flag when it is absent. Malformed rows (wrong column count, non-numeric feature value that is not a missing marker) abort the load with the row number. Digit-grouping separators are NOT supported and are best switched off in the exporting program: under a comma delimiter `1,234.56` splits the row (wrong column count, fatal), and under semicolon/tab it is a non-numeric token (also fatal) — but `1,234`, having no decimal point, is silently normalized by the decimal-comma rule and read as 1.234 |
| splitters | 5050, cv, holdout, resub, leave1out, randrand, randfix (outer `--splitter`, inner `--splitter2`; params `--cv-folds`, `--holdout-train`, `--rr-splits/-train/-test` and `--splitter2-*` twins) |
| scalers | void, to01, white. `--scaler-fit all\|train` chooses what the transform is learned from: `all` (default) = the entire dataset at load, before any splitting — the historical FST3 behavior, in which test data contributes to min/max or mean/stddev (a documented methodological leakage); `train` = the current depth-0 split's TRAIN part only, re-fitted automatically on every depth-0 split change and applied to both parts (leakage-free normalization — the same per-split discipline the logreg/ridge/linsvm/rbfsvm trainers already apply internally). `train` also covers `--missing-value` imputation (the per-feature substitution average is then computed from the TRAIN part only). Costs one full re-scale pass per depth-0 split change (noticeable with wrapper criteria over a CV outer splitter, where every subset evaluation iterates the splits); threaded runs stay bit-identical to sequential — every worker derives the identical split transform independently. `--classify-samples` vectors receive the transform fitted on the depth-0 split the classifier is trained on (per feature, like the loaded data) |
| criteria | bhattacharyya, divergence, gmahalanobis (normal-model filters, monotonic, two-class); multinom-bhattacharyya (multinomial filter, count/text data), multinom-mi (multinomial per-term mutual information: a subset scores the SUM of its terms' MI shares — monotonic, a ranking criterion for `--search bif` on count/text data, see demo44.cfg); wrapper-knn, wrapper-bayes, wrapper-logreg (multinomial logistic regression, `--logreg-lambda/-maxiter/-tol`; deterministic damped-Newton training on internally standardized data, RNG-free — byte-identical at any `--seed`/`--threads`; constant features are safe; many classes × large subsets is the expensive corner), wrapper-rf (4.2: dependency-free Random Forest — CART/Gini trees, bagging, sqrt-mtry node subsampling; `--rf-trees/-mtry/-max-depth/-min-leaf`; training seeds derive from (subset content, `--seed`, split index), never the fst_rand stream, so evaluations are pure functions — bit-identical at any `--threads`), wrapper-linsvm (4.2: dependency-free linear SVM via dual coordinate descent, `--linsvm-c/-loss/-maxiter/-tol`; fixed cyclic order = RNG-free/deterministic; one-vs-rest multiclass; no LibSVM and no LINEAR-kernel freeze), wrapper-rbfsvm (4.2: dependency-free RBF kernel SVM — C-SVC-only kernel dual coordinate descent with the penalized-bias kernel offset, `--rbfsvm-c/-gamma/-loss/-maxiter/-tol`; fixed cyclic order = RNG-free/deterministic — bit-identical at any `--threads`; gamma default 1/subset-size on internally standardized features; one-vs-rest multiclass; `--rbfsvm-optimize pre\|alternating` = the native C/gamma twin of `--svm-optimize`, same space-filling fst_rand search and run protocols; no LibSVM — deliberately NOT bit-comparable to wrapper-svm, the correspondence bar is near-identical accuracy at matched C/gamma), wrapper-lda (4.2: shared-covariance linear discriminant, parameterless; pooled-covariance inverse via the Indexed_Matrix LU machinery — Eigen-path parity; closed-form, K classes; fails on singular covariance like the normal-model filters), wrapper-extern (4.2: external-classifier bridge, `--extern-cmd "<command>"` — one persistent child process per criterion clone driven over a TRAIN/TEST line protocol, reference implementation `tests/extern_centroid.awk`; **user-supplied code execution** and **reproducibility guarantees end at the bridge**; AUC unavailable — labels-only protocol), wrapper-multinom, wrapper-svm, wrapper-regressor (weighted-neighbours regression whose dependent variable is the CLASS LABEL — "regression simulating classification", the demo11reg protocol; its default criterion value is the fraction of test samples predicted within ±0.5 of the true label, i.e. a hit rate rather than a prediction error, so selecting `r2`/`neg-rmse`/`neg-mae` changes both the scale and the meaning of the reported value), wrapper-ridge (4.2: closed-form L2-regularized linear regression via Cholesky on internally standardized data, `--ridge-lambda`; criterion value defaults to R² — deterministic, RNG-free like logreg); extern-value (external VALUE-mode criterion — the tool returns the criterion value itself over the VALUE line protocol, `--extern-cmd` + optional `--extern-monotonic` monotonicity assertion admitting `bb-*`; reference `tests/extern_value_meandiff.awk`; wrapper-extern's caveats apply). All wrapper-* criteria accept `--wrapper-metric` (4.2): balanced-accuracy \| f1 \| f1-macro \| auc for classification wrappers (f1 is two-class with positive class = class index 1 by default, selectable with `--f1-positive <idx>` — FST class order is the data-file class order; use f1-macro on multiclass; auc covers any class count — two-class = the exact binary AUC with midrank ties, K>2 = one-vs-rest macro AUC, the mean of per-class binary AUCs with absent classes skipped; all from deterministic per-sample scores — logreg linear scores, Bayes posteriors, kNN vote fractions, SVM decision values with K>2 aggregated per class from the one-vs-one pairs; note kNN's scores take only k+1 distinct values, so its AUC is a coarse estimate and its ROC/PR curves are visibly stepped), r2 \| neg-rmse \| neg-mae for wrapper-regressor; plus `--class-weights w0,w1,...` (per-class cost weights on the confusion matrix). Default accuracy = the plain accuracy. `--validate` reports accuracy and the selected metric side by side |
| searches | bif, sfs/sbs, sffs/sbfs, sfrs/sbrs, os, dos, exhaustive, montecarlo, bb-basic/bb-improved/bb-fast/bb-bbpp (all four B&B variants accept `--threads N` — candidate-level engine batches, bit-identical to sequential) |
| evaluators | straight (default), hybrid (`--hybrid-prefilter`; the prefilter pre-sort stays on the master, the main-criterion phase thread-pools), ensemble (`--ensemble-knn-ks`; each member's candidate evaluations thread-pool, one batch per member per step, vote combination on the master), sampled (budgeted stochastic step, `--step-*`/`--warmup-*` knobs; the method of Somol and Grim, arXiv:2608.01502) — sequential searches only; all four accept `--threads N` with results identical to `--threads 1` (bit-identical at matched `--seed`) |
| trackers | dupless (default, feeds `top_k`), regularizer, stabileval, feature-stats (`--daf-rank` extracts the DAF ranking — the published 2011 method; `--daf-pairs K` adds pairwise group statistics over the top-K DAF0 features) |

Key validity rules (enforced with actionable errors): B&B and `os` need
`--target-d`; B&B needs a monotonic filter criterion; monotonic filters always
need `--target-d`; `--tracker-attach evaluator`, `--generalization-level` and the
hybrid/ensemble evaluators need a sequential-family search; `stabileval` needs
`--repeat-over-splits`; `--repeat-over-splits` with a wrapper criterion (or the
ensemble evaluator) needs `--splitter2`; the regression wrappers (`wrapper-regressor`,
`wrapper-ridge`) validate only with `same` (their target is the class label —
"regression simulating classification", the demo11reg protocol); normal-model filters
fail on constant features
(the error names them); `--wrapper-metric`/`--class-weights` need a wrapper-*
criterion, regression metrics need `wrapper-regressor` (and vice versa), f1
needs two-class data (auc covers any class count),
`--f1-positive` needs `--wrapper-metric f1` and a valid class index of the
two-class data, `--class-weights` needs one positive weight per class and
combines with neither balanced-accuracy (double reweighting) nor auc.

### Flags not covered above

The component table and the protocol list name the flags a typical run needs;
`fst4 --help` documents every flag. The remaining ones, by group:

- Search extent and seeding: `--sffs-delta D` bounds the floating searches
  in their OWN direction (sffs stops adding features at target-d+D, sbfs
  stops removing them at target-d-D; the conditional backtracking steps are
  bounded only by their improvement rule — required by the very-high-
  dimensional protocols of `demo36.cfg`/`demo37t.cfg`, default 0 =
  unrestricted); `--search2-d N` is the size of the `--search2 bif` seed
  (default 0 = `--target-d`; sfs needs it smaller than the target, sbs
  larger, the floating/retreating searches on their side of the target or
  equal to it); `--restarts-max N` caps the total number of
  `--restarts-nonimproving` runs (default 0 = unlimited).
- Monte Carlo: `--mc-seconds S` is a wall-clock stopping budget (alternative
  or complement to `--mc-trials`; results become load-dependent);
  `--mc-card-prob p` draws probe subsets by per-feature inclusion
  probability instead of the `--mc-card-min/-max` cardinality range.
- Fast Branch & Bound: `--bb-gamma G` is the prediction optimism factor and
  `--bb-delta N` the minimum prediction evidence count (with `--search
  bb-fast`; defaults 1.0 and 1).
- Hybrid evaluator: `--hybrid-keep-perc P` (percentage of prefiltered
  candidates that reach the main criterion, default 50) and
  `--hybrid-keep-max N` (hard cap on that shortlist, default 1000).
- Trackers and reporting: `--top-k K` (number of best tracked subsets
  reported, default 5), `--tracker-capacity N` (maximum tracked results;
  default `--top-k` for dupless, unlimited for the others),
  `--feature-costs c0,c1,...` (per-feature acquisition costs for
  `--regularize-by weights`), `--daf-count N` (number of top-DAF features to
  report, default 0 = all), `--daf-pairs-top N` (number of top pairs by
  pair-DAF0 to report, default 50; 0 = all).
- LibSVM (`wrapper-svm`, `--validate svm`): `--svm-kernel
  linear|rbf|poly|sigmoid` (default rbf, the most stable), `--svm-c`,
  `--svm-gamma`, `--svm-coef0` (defaults: the library defaults, or the
  `--svm-optimize` result).

## Output schema (JSON)

`version`, `invocation` (fully resolved flag echo incl. seed — the reproducibility
anchor), `data` (counts), `result` (features/size/value), then per enabled feature:
`per_split`, `validation`, `bias`, `classified`, `trials`, `stability`, `top_k`,
`regularization`, `daf`, `daf_pairs`, `daf_validation`, `multiplier_sweep`+`best_multiplier`,
`compare` (`--compare-config`: `scenario_b` with its trials/stability +
`similarity` with IATI/ICW/IANHI), and `stats` (timings/evaluations; omitted under
`--stable-output`). CSV mirrors the same blocks as `kind,rank,value,size,features`
rows (scenario B rows carry `b_*` kinds, plus `similarity` rows). NOTE the one
1-based index in FST4: `classified` entries (CSV kind `classified`, JSON array
order) number the input rows of the `--classify-samples` file from 1, in file
order, because they identify lines of that file — every feature and class
index FST4 prints or accepts is 0-based.

### Report levels (`--report standard|detailed`)

`standard` (default) is the complete result exactly as above, unchanged by
the flag. `detailed` adds the research sections, in all three formats:

- `curve_over_d` — one row per subset size the search visited: the best-at-d
  criterion value (recorded live from the search's solution announcements; no
  re-evaluation) and, when `--validate` is given, a fresh holdout validation
  of that subset. This is the criterion-axis vs. validated-axis curve — the
  standard overfitting plot; the two axes are deliberately labeled apart
  (selection-time estimate vs. independent holdout). Single-search protocols
  only (trials/sweeps have no single trajectory).
- `curves` — tabular ROC and PR curves of the first `--validate` classifier
  (that supports per-sample scores) on the depth-0 holdout: one point per
  distinct threshold, `roc` rows = [threshold, FPR, TPR], `pr` rows =
  [threshold, recall, precision]; binary = positive class 1, K>2 = one
  one-vs-rest curve per class. Deterministic (the AUC tie conventions).
  Plot directly: `jq -r '.curves.per_class[0].roc[] | @tsv'` → gnuplot.
- `confusion` inside each validation entry (row = true class).
- `tuned_parameters` — the final C/gamma(/coef0) after `--svm-optimize` /
  `--rbfsvm-optimize` (otherwise reproducing a tuned run would require
  re-running the tuning).

### Side files: `--dump-scores` and `--curves-svg`

Two flags write to files instead of stdout, so the result stream stays the
size it was. Both are independent of `--output`; neither changes stdout
beyond echoing its own name in the `invocation` block.

- **`--dump-scores FILE`** (any report level) — the final subset's per-sample
  predictions on the depth-0 holdout, as CSV with `#` comment headers naming
  the classifier and the subset: `sample,true_class,predicted_class` plus one
  `score_K` column per class where the classifier exposes scores. It uses the
  FIRST classifier named by `--validate`, reusing the model already trained
  there (no extra training), and walks the holdout in the SAME class-major
  order the ROC/PR curves are built from — so row *i* of the dump is point *i*
  of the curve input. Classifiers with no score support (the `wrapper-extern`
  label protocol) get the two label columns and a comment saying so. Feeds
  external ROC variants, calibration analysis and per-sample error inspection:

  ```sh
  # error rows only  (data rows start with the sample index, so /^[0-9]/ skips
  # both the # comments and the column header, however many comments there are)
  awk -F, '/^[0-9]/ && $2!=$3' scores.csv
  # count them — this must equal the confusion matrix's off-diagonal sum
  awk -F, '/^[0-9]/ && $2!=$3' scores.csv | wc -l
  # feed an external ROC tool: score for class 1, then the true label (two-class
  # data, where the columns are sample,true,predicted,score_0,score_1)
  awk -F, '/^[0-9]/ {print $5, $2}' scores.csv
  ```

- **`--curves-svg FILE`** (requires `--report detailed`, which computes the
  curves) — the same curves as a self-contained SVG: no plotting dependency,
  no external resources, opens in any browser and drops into LaTeX. One panel
  each for ROC and PR (one polyline per class on multiclass data) and one for
  the over-d curve. That last panel carries **two y-axes** — left = criterion
  value, right = validated accuracy — because the selection-time estimate and
  the independent holdout must never share one scale. Deterministic: fixed
  coordinate precision, byte-identical across runs.

### Unix pipe usage

fst4 is pipe-clean in BOTH directions. Downstream: with `--output json|csv`
ALL library/search prose goes to **stderr** and stdout carries nothing but
the machine-readable result; exit code 0 = success, 1 = any error (messages
+ actionable hints on stderr). `--verbosity silent` suppresses prose
entirely. Upstream: **`--data -`** reads the dataset from standard input.
`--format trn|arff|csv` is then required — there is no file name to infer it
from — and the invocation echo reports `"data": "-"`, not a temporary path.

The bytes are buffered before the run starts, because a pipe is read once
while the readers need to re-read (the TRN reader seeks back past the header,
and `--compare-config` builds a second accessor over the same data). Results
are identical to naming the file directly.

Recipes:

```sh
# criterion value only
./fst4 --data data/wdbc_30.trn --criterion gmahalanobis --search sfs \
    --target-d 7 --output json 2>/dev/null | jq -r '.result.value'
# selected features as a comma list
... --output json 2>/dev/null | jq -r '.result.features | join(",")'
# the overfitting curve as TSV (d, criterion, holdout accuracy)
... --validate same --report detailed --output json 2>/dev/null \
    | jq -r '.curve_over_d[] | [.d, .value, .validation[0].accuracy] | @tsv'
# ROC points for gnuplot (FPR TPR)
... --report detailed --output csv 2>/dev/null | awk -F, '$1=="roc"{print $4, $5}'
# live improvements while the search runs (JSON lines on stderr)
... --progress json 2> >(grep --line-buffered '"event"') > result.json
# dataset in on stdin, result out on stdout — fst4 as a filter
zcat data.trn.gz | ./fst4 --data - --format trn --criterion gmahalanobis \
    --search sfs --target-d 7 --output json 2>/dev/null | jq -r '.result.features'
# select on a column subset prepared upstream, no temporary file
cut -d, -f1-20,41 big.csv | ./fst4 --data - --format csv --criterion wrapper-knn \
    --search sffs --target-d 5 --output csv 2>/dev/null
```

## Examples

```sh
# demo10: filter selection + independent validation
fst4 --data data/wdbc_30.trn --criterion gmahalanobis --search sfs --target-d 7 --validate knn

# demo26: nested splits, Bayes wrapper, dual validation
fst4 --data data/wdbc_30.trn --criterion wrapper-bayes --splitter 5050 --splitter2 cv \
     --search sffs --validate same,knn

# demo41: optimal Branch & Bound on a monotonic filter, kNN validation
fst4 --data data/wdbc_30.trn --criterion bhattacharyya --search bb-improved --target-d 10 \
     --knn-distance Lp --validate knn

# demo54: 10-trial stability evaluation
fst4 --data data/wdbc_30.trn --criterion wrapper-knn --splitter randrand --rr-splits 10 \
     --rr-train 80 --rr-test 20 --splitter2 cv --search sfs --repeat-over-splits --tracker stabileval

# demo51: subset-size regularization sweep
fst4 --data data/waveform_40.trn --criterion wrapper-knn --splitter 5050 --splitter2 cv \
     --search dos --dos-delta 3 --tracker regularizer --regularize-by subsetsize \
     --regularize-margins 0.001,0.002,0.003 --validate same

# DAF ranking (demo34 protocol, scaled down) + pairwise group statistics over the
# top-6 DAF0 features; --threads N is bit-identical to sequential at matched --seed
fst4 --data data/wdbc_30.trn --criterion wrapper-knn --splitter 5050 --splitter2 cv \
     --search montecarlo --mc-trials 1000 --mc-card-min 5 --mc-card-max 50 \
     --tracker feature-stats --daf-rank 0 --daf-pairs 6 --validate knn --daf-validate-every 5

# demo55: cross-scenario similarity — scenario A on the command line, scenario B in a
# config file (same keys as --config; must use the same --data and the stabileval +
# repeat-over-splits protocol); reports IATI/ICW/IANHI between the two trackers
fst4 --data data/wdbc_30.trn --criterion wrapper-knn --knn-k 3 --knn-distance L1 \
     --splitter randrand --rr-splits 10 --rr-train 95 --rr-test 5 --splitter2 cv \
     --search dos --dos-delta 10 --repeat-over-splits --tracker stabileval \
     --compare-config tests/fst4_compare_demo55_b.cfg
```

## FST3 demo parity configs

Every released FST3/FST4-3.x demo lives on as a ready-to-run config file in the
`demos/` subdirectory, named after the demo it reproduces: `demos/demo10.cfg` …
`demos/demo63.cfg` (36 files, one per demo published at fst.utia.cz). The
numbers 14, 36 and 37 never existed in the public FST3 and are not parity
material; FST4 reuses them for new demos (below). The published 2011 DAF
method is the montecarlo + `--tracker feature-stats` + `--daf-rank` protocol,
see `demo34.cfg`/`demo35t.cfg`. Run any of them from the repo root:

```sh
fst4 --config demos/demo22.cfg    # Example 22: randomized Oscillating Search
```

Each file's header comment states the demo title, the protocol, and a runtime
warning where the original demo was compute-bound or wall-clock-budgeted
(demo24/24t/34/35t are load-dependent by design; demo11t/25t/32t/33/33t/40/50/60
run for tens of minutes to hours). The protocol parameters follow the
golden-verified `fst4_*` scenarios in `tests/manifest.tsv`. The three compiled C++
exemplars of the same scenarios (demo10, demo52t, demo55) are reproduced, with
their sources cross-referenced to the classes they use, in the Examples section
of the generated API documentation.

Since FST4 4.2 parity is in PROTOCOL, not in numbers, for the 16 demos that
FST3 ran on its speech data: that data set is no longer distributed, and the
affected configs run the same flags on `data/wdbc_30.trn` (Breast Cancer
Wisconsin Diagnostic) instead, so results differ from FST3-published speech
numbers. One forced protocol deviation: demo40/demo40t bound their exhaustive
search to `target-d = 5` — enumerating all subset sizes was feasible at
speech's 15 features but is not at WDBC's 30.

Also since 4.2, the sonar demos (22/23/56/62) differ from FST3-published
numbers for a different reason: the historical sonar TRN conversion declared
a wrong class split (103 rocks / 105 mines), mislabeling six samples. Both
`sonar_60.trn` and `sonar_60_missing_data.trn` now carry the official UCI
split (97 rocks / 111 mines) — correctness over parity. The ARFF variant
always had the correct labels and is renamed `sonar_60.arff`.

Additional configs are **new FST4-native demos** (4.1/4.2), reusing freed or
never-released numbers — their headers state explicitly that they are not FST3
parity configs:

- `demo13t.cfg` — threaded wrapper selection with the 4.1 logistic-regression
  classifier on the near-linearly-separable WDBC data (fast).
- `demo14.cfg` — 4.2 wrapper metrics: balanced-accuracy vs. plain-accuracy
  3-NN selection compared by cross-scenario tracker similarity (the demo55
  protocol; scenario B in `tests/fst4_compare_demo14_plain.cfg`). The number
  is reused — FST3's internal demo14 draft was never released (fast).
- `demo15.cfg` — 4.2 wrapper metrics: logistic-regression wrapper optimizing
  AUC on WDBC (fast). Reads `data/wdbc_30.csv` — the CSV conversion of
  wdbc_30.trn — demonstrating the CSV loader on real data (results identical
  to the TRN version).
- `demo16.cfg` — 4.2 wrapper metrics: cost-sensitive normal-Bayes selection
  with `--class-weights 1,5` on WDBC (fast).
- `demo17.cfg` — 4.2 wrapper metrics: regression selection scored by R² with
  the weighted-neighbours regressor on WDBC data (fast).
- `demo18.cfg` — 4.2 Random Forest wrapper on spambase: 100-tree CART/Gini
  forest as the criterion, SFS to 10 features, threads 4 (under a minute;
  bit-identical at any thread count).
- `demo19.cfg` — 4.2 ridge regression wrapper scored by R² on WDBC — the
  linear counterpart of demo17 (fast).
- `demo27.cfg` — 4.2 wrapper metrics: multiclass selection by one-vs-rest
  macro AUC (logistic regression on waveform; the same `wrapper-metric =
  "auc"` flag as demo15, on K>2 classes; fast).
- `demo28.cfg` — 4.2 RBF kernel SVM wrapper on sonar (the demo23 protocol
  minus its LibSVM parameter grid; native kernel dual coordinate descent, no
  LibSVM involved — results deliberately not bit-comparable to demo23's;
  sub-second).
- `demo29.cfg` — 4.2 RBF kernel SVM wrapper with `rbfsvm-optimize =
  "alternating"`: the FULL demo23 protocol natively — the same space-filling
  C/gamma search and alternating optimize/re-search choreography as
  `--svm-optimize`, no LibSVM involved (seconds; note the small-sample
  overfitting lesson in the config header).
- `demo44.cfg` — 4.2 multinomial mutual-information term ranking on the
  reuters text data: the demo31 protocol with `multinom-mi` in place of the
  multinomial Bhattacharyya, so the two term-goodness rankings compare one
  flag for one flag (seconds).
- `demo38.cfg` — 4.2 linear SVM wrapper on spambase (the demo18 protocol with
  a linear margin model; no LibSVM involved; seconds). Reads
  `data/spambase_57.csv` — the CSV conversion of spambase_57.trn —
  demonstrating the CSV loader on real data (results identical to the TRN
  version).
- `demo64.cfg` — 4.2 external-classifier bridge: selection wrapped around the
  reference awk nearest-centroid model over the persistent-process line
  protocol (needs awk on PATH; seconds).

### The wrapper-extern line protocol (4.2)

One persistent child process per criterion clone, launched through the system
shell (`--extern-cmd`), driven on stdin/stdout; values travel as `%.17g`
decimals (bit round-trip for doubles), class labels are 0-based indices:

```
-> TRAIN <N> <d> <K>\n     then N lines:  <label> <v1> ... <vd>
<- OK\n                    (anything else fails the evaluation, loudly)
-> TEST <M> <d>\n          then M lines:  <v1> ... <vd>
<- M lines:                <predicted label>
```

TRAIN/TEST pairs repeat for every candidate subset; EOF on stdin asks the
child to exit. Reference implementation: `tests/extern_centroid.awk`.
**Caveats by design**: the command is user-supplied code execution (treat
configs carrying `extern-cmd` like scripts); FST4's determinism guarantees
end at the bridge (a stochastic external trainer makes results run- and
thread-dependent); no timeout machinery (a hung command hangs the search).

### The three extern layers

FST4 offers three distinct ways to hand computation to an external tool,
differing in WHAT the tool is and WHAT travels over the pipe:

| layer | the tool is | data over the pipe | per-evaluation request | typical use |
|---|---|---|---|---|
| `wrapper-extern` | an external CLASSIFIER | train+test blocks, subset-narrowed, EVERY evaluation | TRAIN + TEST blocks, labels back | plug in a classifier FST4 lacks; FST4 computes accuracy/metrics from its labels |
| `extern-value` | an external DATA-DEPENDENT CRITERION | full-width TRAIN part, ONCE per split | `VALUE` + subset indices, one value back | a custom filter measure (MI estimators, HSIC, relief-F, any Python/R score) driven by FST4's search machinery |
| `extern-value --extern-no-data` | an external SET FUNCTION | NONE | `VALUE` + subset indices, one value back | black-box subset optimization: cost tables, simulations, external experiments |

Cadence, stated explicitly to prevent the natural misreading: the
extern-value DATA block travels **once per split** — never per evaluation
and never sample-by-sample — as one batch carrying the whole current
training part at full feature width. Only the tiny index-only VALUE
requests repeat per candidate subset. The DATA block is also the
split-synchronization mechanism: only FST4 knows which trial or fold is
current, so an external data-dependent criterion must receive its training
data through the protocol — a self-loading tool could not follow the
choreography and would leak held-out data into the criterion.

The set-function layer has its own data-file-free entry point:
**`--features N`** replaces `--data` entirely and declares an
abstract universe of N features — nothing else about it exists. Requires
`--criterion extern-value --extern-no-data` explicitly, and refuses
everything whose meaning depends on data existing (`--validate`, real
splitters, scalers, wrapper criteria, `--report detailed`, the
hybrid/ensemble evaluators). What remains is the full search machinery:
every search including the four B&B variants (`--extern-monotonic`
admitting them), all trackers with DAF statistics, `--search2 bif`
seeding, the sampled evaluator, subset-size/cost regularization
(`--regularize-by weights` — feature costs are abstract too), and
`--threads N` with bit-identical results. The invocation echo then carries
`"features": N` in place of the data/format keys, and the data block
reports the honest degenerate shape (0 classes, 0 samples). Golden
`fst4_features_setfunc` pins the entry against `fst4_extern_value_nodata`:
same protocol, same result line — the entry points differ, the
optimization is identical.

### The extern-value VALUE protocol

`--criterion extern-value` makes the external tool compute the CRITERION
VALUE itself — unlike wrapper-extern, where the tool is a classifier and
FST4 computes accuracy from its labels. Same persistent-child machinery
(one process per criterion clone, shared plumbing in `extern_bridge.hpp`);
the data travels ONCE per split, evaluations are index-only requests:

```
-> DATA <N> <D> <K>\n      then N lines:  <label> <v1> ... <vD>   (full width)
<- OK\n
-> VALUE <d> <f1> ... <fd>\n   (0-based raw feature indices, ascending)
<- one line:               <criterion value>   (strict full-line double)
```

The DATA block carries the full-width TRAIN part of the current split (the
filter convention — the criterion sees training data only) and is re-sent
whenever the split changes (the same refresh contract as filter model
re-estimation) or the child is respawned — see "The three extern layers"
above for why the data must arrive through the protocol and for the
per-layer cadence. `--extern-no-data` suppresses the DATA block entirely
for the set-function layer (with `--data` the file then only defines the
universe size; `--features N` drops the file entirely — see the layers
section); data-free reference: `tests/extern_value_setfunc.awk` (positive
additive weights — monotonic by construction). Each subset evaluation then
costs d integers instead of re-streaming N×d values — the cost model that
matters under B&B's thousands of evaluations — and the raw indices give
the tool feature identities (per-feature caching, acquisition-cost-aware
measures). The intended tool shape is the SUFFICIENT-STATISTICS pattern:
precompute per-feature statistics at DATA time, answer each VALUE in O(d).
Reference implementation: `tests/extern_value_meandiff.awk` (sum of
per-feature |class-mean differences| — monotonic by construction).

Value semantics: FST4 **maximizes** — larger must mean better (negate
error-style measures inside the tool). Beyond direction there is no range
or scale requirement: values are only compared within the same search,
never across criteria (built-in filters already return unbounded distances
while wrappers return [0,1] accuracies). Values must be finite — `nan`/
`inf` responses fail loudly — and deterministic for identical requests;
margin-style flags (`--tracker-margin`, `--regularize-margins`) are
expressed in the external measure's own units, as with every criterion.

Normalization follows the standard FST4 convention: the tool receives the
data AFTER the global `--scaler` (by default learned per-feature over the
whole dataset at load, before splitting; under `--scaler-fit train` the
DATA block carries the current split's TRAIN-part-fitted transform, since
it is re-sent on the same initialize()/respawn contract that follows split
changes) — exactly what every built-in criterion sees, bit-exact via
`%.17g`. The bridge applies NO internal per-training
standardization (unlike logreg/ridge/linsvm/rbfsvm's internal detail); if
the measure needs further normalization the tool owns it. Caution when
combined with `--extern-monotonic`: per-feature-independent normalization
inside the tool preserves a per-feature-sum's monotonicity, but any
normalization coupling the streamed columns (subset-level variance or
totals) makes contributions depend on which other features are present and
can silently break the asserted monotonicity — voiding the B&B optimality
guarantee.

`--extern-monotonic` asserts that the external measure is monotonic w.r.t.
subset cardinality (removing a feature never increases the value). The
assertion maps to `Criterion::is_monotonic()` and admits the optimal
`bb-*` searches — **the optimality guarantee then rests entirely on the
assertion**: a false assertion yields silently suboptimal "optimal"
results. Without the flag, `bb-*` refuses extern-value (and the library
prints its monotonicity warning). Verified at introduction: with the
reference tool, bb-improved and bb-fast both return the exhaustive-search
optimum, confirmed against a pure-python brute force (see the
fst4_extern_value* manifest comments). All wrapper-extern caveats apply
unchanged; under B&B a stochastic external measure breaks not just
reproducibility but the optimality logic itself.
- `demo36.cfg` — the stochastic-floating-search paper's gisette protocol
  (Somol and Grim, arXiv:2608.01502, 2026):
  budgeted stochastic (sampled) SFFS, 1-NN wrapper, target d=300 (hours).
- `demo37t.cfg` — the paper's reuters extreme-d protocol: sampled SFFS at
  d=500 over 10,105 features, multinomial Bhattacharyya filter, 8 threads
  (bit-identical to sequential; minutes).

## Reproducibility

`--seed` (default 1) seeds FST4's own random generator (`FST::fst_srand()`);
all library randomness (randrand splitters, Monte Carlo probes, random
subsets) draws from it. The generator is FST-owned, so seeded runs reproduce
seed-for-seed across platforms (Windows, macOS, Linux), and under
`--stable-output` the same flags give byte-identical output. The demos never
seed explicitly, which equals `--seed 1`; that is why the `fst4_*` golden
scenarios reproduce demo results exactly (see `tests/manifest.tsv`).

The sampled evaluator keeps this determinism even threaded: with `--evaluator
sampled --threads N` the proposal sampling stays on the master thread's RNG
stream and only criterion evaluations fan out, with results replayed in proposal
order — output is bit-identical to `--threads 1` at matched `--seed`, for any N.

The same recipe covers the streaming searches: `--search montecarlo` and `--search exhaustive` with `--threads N`
generate candidates on the master thread and replay results in generation
order, so they too are bit-identical to sequential at matched `--seed` for any
thread count (only the engine-name echo and the `"threads"` invocation-echo
field differ). In particular the DAF probe pipeline (montecarlo +
feature-stats) is fully reproducible threaded.

The Branch & Bound family follows the same recipe: all four variants
batch their independent criterion evaluations through the engine — IBB's
level pre-ordering, FBB/BBPP's non-predictable candidates (the
predict-vs-compute decisions are made upfront on the master, which is exact
because predictor state is per-feature), BBPP/BBB's chosen-node computations,
and leaf processing in every variant — with results and prediction learning
replayed in candidate order. `--threads N` is bit-identical to sequential for
any N; the lazy compute-before-cut in FBB/BBB stays on the master (a single
evaluation on the critical path). Tree-level parallelism (concurrent
subtrees) is deliberately out of scope — the shared bound would make the
work nondeterministic. Performance note (one machine, wdbc d=10,
bhattacharyya): bb-improved gains real speedup (2.2× at `--threads 4`,
2.5× at 8); bb-fast/bb-bbpp/bb-basic remain
near-sequential-speed threaded (+7–16% overhead) because their
genuinely-computed fraction is small by design — with the cheap built-in
filters keep `--threads 1` there, with expensive criteria the batches
dominate and threading pays across the family.

## Known limitations

- `--compare-config` requires both scenarios to run on the same `--data`
  file with the stabileval tracker and the repeat-over-splits protocol; the two
  scenarios run sequentially on one continuous random stream (the demo55
  in-process sequence), so scenario B's random splits depend on scenario A
  having run first.
