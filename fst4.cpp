/*!======================================================================
   Feature Selection Toolbox 4 - fst4 command-line tool
   ----------------------------------------------------

   \file    fst4.cpp
   \brief   Exposes FST4 functionality as a command-line tool: data file,
            splitter, scaler, criterion, search algorithm and their settings
            are selected via command-line arguments.
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work

   Design: every component is instantiated ONCE against the abstract component
   bases (type erasure) and concrete implementations are chosen at runtime
   through string-keyed factories - total template instantiations stay linear
   in the number of components.

   Component surface: `fst4 --list` prints the complete registry (data formats,
   splitters, scalers, criteria, searches, step evaluators, trackers) with
   validity notes, `fst4 --help` every flag; doc/FST4_CLI.md is the reference.

   Results are sourced from the result tracker and the final subset, and
   printed as text (default), JSON, or CSV. `--stable-output` omits the
   timing/evaluation-count block so that output is reproducible byte for byte.

   Compile-time menu notes: Classifier_kNN constructs its DISTANCE internally,
   so each distance is a separate kNN instantiation; Distance_Lp's exponent is
   a template parameter, so `--knn-distance` Lp means p=3/2 (the value used by
   the demos).
*/ /*
=========================================================================
Copyright and license:
  * FST4 software (with the exception of any bundled or externally linked
    third-party components) is Copyright (C) 2000-present Institute of
    Information Theory and Automation of the Czech Academy of Sciences (UTIA).
  * FST4 is free software: you can redistribute it and/or modify it under
    the terms of the GNU Affero General Public License as published by the
    Free Software Foundation, version 3. The full text of the license is in
    the file LICENSE in the distribution root and at
    https://www.gnu.org/licenses/agpl-3.0.html.
  * FST4 is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
    License for more details.
  * Closed-source use without the AGPL's copyleft obligations requires a
    commercial license from UTIA; in case of interest write to
    utia@utia.cas.cz. LICENSING.md in the distribution root explains both
    options, the contributor license agreement that every contribution
    requires, and the contacts (technical questions and pull requests:
    somol@utia.cas.cz, with "FST" in the subject line).
========================================================================== */
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CLI11.hpp"

#include "error.hpp"
#include "global.hpp"
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
#include "subset.hpp"
#include "data_intervaller.hpp"
#include "data_splitter.hpp"
#include "data_splitter_5050.hpp"
#include "data_splitter_cv.hpp"
#include "data_splitter_holdout.hpp"
#include "data_splitter_resub.hpp"
#include "data_splitter_randrand.hpp"
#include "data_splitter_randfix.hpp"
#include "data_splitter_leave1out.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
#include "data_scaler_to01.hpp"
#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"
#include "data_accessor_splitting_memCSV.hpp"
#include "data_accessor_universe.hpp"
#include "criterion_normal_bhattacharyya.hpp"
#include "criterion_normal_divergence.hpp"
#include "criterion_normal_gmahalanobis.hpp"
#include "criterion_multinom_bhattacharyya.hpp"
#include "classifier_multinom_naivebayes.hpp"
#ifdef FST_SVM
#include "classifier_svm.hpp" // requires LibSVM (not distributed; fetch per libsvm/readme.txt)
#endif
#include "criterion_wrapper.hpp"
#include "criterion_wrapper_bias_estimate.hpp"
#include "regressor.hpp"
#include "regressor_weighted_neighbours.hpp"
#include "regressor_ridge.hpp"
#include "distance_euclid.hpp"
#include "distance_L1.hpp"
#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
#include "classifier_normal_bayes.hpp"
#include "classifier_logreg.hpp"
#include "classifier_rf.hpp"
#include "classifier_linear_svm.hpp"
#include "classifier_rbf_svm.hpp"
#include "classifier_lda.hpp"
#include "classifier_extern.hpp"
#include "criterion_extern_value.hpp"
#include "seq_step_straight.hpp"
#include "seq_step_hybrid.hpp"
#include "seq_step_ensemble.hpp"
#include "result_weighter_sampled.hpp"
#include "seq_step_sampled.hpp"
#include "search_seq_sfs.hpp"
#include "search_seq_sffs.hpp"
#include "search_seq_sfrs.hpp"
#include "search_seq_os.hpp"
#include "search_seq_dos.hpp"
#include "search_ranking_bif.hpp"
#include "search_exhaustive.hpp"
#include "search_monte_carlo.hpp"
#include "search_branch_and_bound_basic.hpp"
#include "search_branch_and_bound_improved.hpp"
#include "search_branch_and_bound_fast.hpp"
#include "search_branch_and_bound_partial_prediction.hpp"
#include "branch_and_bound_predictor_averaging.hpp"
#include "candidate_evaluator.hpp"
#include "candidate_evaluator_sequential.hpp"
#include "candidate_evaluator_threaded.hpp"
#include "criterion_negative.hpp"
#include "criterion_subsetsize.hpp"
#include "criterion_sumofweights.hpp"
#include "result_tracker_dupless.hpp"
#include "result_tracker_regularizer.hpp"
#include "result_tracker_stabileval.hpp"
#include "result_tracker_feature_stats.hpp"

#define FST4_VERSION "4.2.0" //!< fst4 CLI version string; echoed in the text/JSON/CSV invocation block

/*! \namespace fst4_cli
    \brief Internals of the fst4 command-line tool: its type aliases, the option record, the run
    pipeline with its result types, the CLI-specific observers and adapters, and the output emitters.

    Nothing in here is library API -- the library is namespace FST, which the tool uses through
    explicit qualification. The namespace keeps the tool's own types visibly apart from the
    library's, in the reference documentation as much as for the compiler; main() alone stays
    at global scope and imports this namespace. */
namespace fst4_cli {

typedef double RETURNTYPE; //!< criterion/classifier/regressor return value type used throughout the CLI
typedef double DATATYPE; //!< raw sample value type read from the data file
typedef double REALTYPE; //!< intermediate real-valued computation type (always real, never integral)
typedef unsigned int IDXTYPE; //!< sample (row) index type
typedef unsigned int DIMTYPE; //!< feature (dimension) index type
typedef short BINTYPE; //!< per-feature membership flag type backing SUBSET
typedef FST::Subset<BINTYPE,DIMTYPE> SUBSET; //!< the concrete feature-subset type instantiated across the whole CLI
typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER; //!< concrete split-interval representation shared by every splitter
typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER; //!< shared pointer to the abstract Data_Splitter base, holding whichever concrete splitter `--splitter`/`--splitter2` selects at runtime

// ---- the type-erased seams: ABSTRACT bases as template arguments ----
typedef FST::Data_Accessor<DATATYPE,IDXTYPE> DATAACCESSOR; //!< abstract data-access seam: the base every format accessor (and the `--features` data-free universe accessor) implements
typedef FST::Criterion<RETURNTYPE,SUBSET> CRITERION; //!< abstract criterion seam: the base every `--criterion` implementation and internal adapter implements
typedef FST::Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CLASSIFIER; //!< abstract classifier seam: the base every `--validate` classifier and wrapper-* criterion's inner classifier implements
typedef FST::Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATOR; //!< abstract sequential-step seam: the base every `--evaluator` implementation implements
typedef FST::Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> SEARCH; //!< abstract search seam: the base every `--search` algorithm implements

// ---- concrete components, each instantiated ONCE against the abstract seams ----
typedef FST::Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_MEM; //!< shared concrete-side base: initialize(), setSplittingDepth()
typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_TRN; //!< TRN-format data accessor (`--format` trn)
typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_ARFF; //!< ARFF-format (Weka) data accessor (`--format` arff)
typedef FST::Data_Accessor_Splitting_MemCSV<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_CSV; //!< CSV/TSV-format data accessor (`--format` csv)
typedef FST::Data_Accessor_Universe<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_UNIVERSE; //!< `--features` N: data-free set-function entry
typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050; //!< 1st half train, 2nd half test, deterministic (`--splitter`/`--splitter2` 5050, the default)
typedef FST::Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV; //!< k-fold cross-validation splitter (`--splitter`/`--splitter2` cv; `--cv-folds`/`--splitter2-cv-folds`)
typedef FST::Data_Splitter_Holdout<INTERVALLER,IDXTYPE> SPLITTERHOLDOUT; //!< percentage holdout splitter: first x% train, rest test (`--splitter`/`--splitter2` holdout; `--holdout-train`/`--splitter2-holdout-train`)
typedef FST::Data_Splitter_Resub<INTERVALLER,IDXTYPE> SPLITTERRESUB; //!< resubstitution splitter: train = test = all data (`--splitter`/`--splitter2` resub)
typedef FST::Data_Splitter_RandomRandom<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRR; //!< randrand splitter: independent random train/test percentages each split, disjoint, multi-split capable (`--splitter`/`--splitter2` randrand)
typedef FST::Data_Splitter_TrainRandom_TestFixed<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRANDFIX; //!< randfix splitter: like randrand but with a fixed test part reused across splits (`--splitter`/`--splitter2` randfix)
typedef FST::Data_Splitter_Leave1Out<INTERVALLER,IDXTYPE> SPLITTERL1O; //!< leave-one-out splitter, intended as the inner `--splitter2` (`--splitter`/`--splitter2` leave1out)
typedef FST::Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_BHAT; //!< bhattacharyya: normal-model filter, monotonic (B&B-compatible), two-class data only
typedef FST::Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_DIV; //!< divergence: normal-model filter, monotonic in theory but numerically unstable -- prefer bhattacharyya; two-class only
typedef FST::Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_GMAHA; //!< gmahalanobis: normal-model filter (generalized Mahalanobis distance), monotonic, two-class only
typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DIST_L2; //!< Euclidean (L2) distance, the kNN default (`--knn-distance` L2)
typedef FST::Distance_L1<DATATYPE,DIMTYPE,SUBSET> DIST_L1; //!< L1 (Manhattan) distance (`--knn-distance` L1)
typedef FST::Distance_Lp<DATATYPE,REALTYPE,DIMTYPE,SUBSET,3,2> DIST_LP; //!< p=3/2, as used by demo40-43 (compile-time menu, see file header) (`--knn-distance` Lp)
typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DIST_L2> KNN_L2; //!< kNN classifier/wrapper-knn criterion using the L2 distance (`--knn-distance` L2, default)
typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DIST_L1> KNN_L1; //!< kNN classifier/wrapper-knn criterion using the L1 distance (`--knn-distance` L1)
typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DIST_LP> KNN_LP; //!< kNN classifier/wrapper-knn criterion using the Lp distance (`--knn-distance` Lp)
typedef FST::Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> BAYES; //!< normal Bayes classifier (wrapper-bayes)
typedef FST::Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> MULTINOM_NB; //!< multinomial naive-Bayes classifier, count/text data (wrapper-multinom)
typedef FST::Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> LOGREG; //!< multinomial logistic regression classifier: deterministic damped-Newton training on internally standardized data (wrapper-logreg)
typedef FST::Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> RF; //!< dependency-free Random Forest classifier: CART/Gini trees, bagging, sqrt-mtry node subsampling (wrapper-rf)
typedef FST::Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> LINSVM; //!< dependency-free linear SVM classifier: L2-regularized dual coordinate descent, no LibSVM needed (wrapper-linsvm)
typedef FST::Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> RBFSVM; //!< dependency-free RBF-kernel SVM classifier: kernel dual coordinate descent, no LibSVM needed (wrapper-rbfsvm)
typedef FST::Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> LDA; //!< shared-covariance linear discriminant classifier, parameterless and deterministic (wrapper-lda)
typedef FST::Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> EXTERN; //!< external-classifier bridge over a persistent-process line protocol (wrapper-extern; `--extern-cmd`)
typedef FST::Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_EXTVAL; //!< the external VALUE-mode criterion: the external tool returns the criterion value itself (extern-value; `--extern-cmd`)
typedef FST::Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_MBHAT; //!< multinom-bhattacharyya: multinomial-model filter, monotonic, count/text data
typedef FST::Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_MMI; //!< multinom-mi: multinomial individual mutual-information ranking criterion; subset value = sum of per-feature MI (monotonic, NOT a joint measure; intended for `--search` bif)
#ifdef FST_SVM
typedef FST::Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> SVM; //!< LibSVM-backed classifier (wrapper-svm; requires LibSVM, fetched separately per libsvm/readme.txt -- only built when FST_SVM is defined)
#endif
typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> CRIT_WRAPPER; //!< the generic wrapper criterion: estimates a CLASSIFIER's accuracy/metric as the criterion value (instantiates every classification wrapper-* criterion)
// ---- regression stack (demo11reg): the Regressor abstract base is a separate seam ----
typedef FST::Regressor<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> REGRESSOR; //!< abstract regressor seam, separate from CLASSIFIER: the base wrapper-regressor/wrapper-ridge implement
typedef FST::Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DIST_L2> RWN; //!< weighted-neighbours kNN regressor; dependent variable = the class label, "regression simulating classification" (wrapper-regressor, demo11reg protocol)
typedef FST::Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> RIDGE; //!< closed-form ridge regressor: Cholesky-solved L2-regularized least squares (wrapper-ridge)
typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,REGRESSOR,DATAACCESSOR> CRIT_WRAPPER_REG; //!< the generic wrapper criterion instantiated over REGRESSOR: estimates a regressor's metric as the criterion value (wrapper-regressor/wrapper-ridge)
typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_SEQ; //!< the default (straight) sequential step: plain in-line candidate evaluation (`--evaluator` straight)
typedef FST::Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,CRITERION> EVAL_HYBRID;   //!< the hybrid sequential step: `--hybrid-prefilter` pre-orders candidates, only the top `--hybrid-keep-perc`% reach the main criterion (`--evaluator` hybrid, demo53); prefilter bound to the abstract base too
typedef FST::Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_ENSEMBLE; //!< the ensemble sequential step: kNN wrappers at the `--ensemble-knn-ks` values vote on candidate ranking (`--evaluator` ensemble, demo50)
typedef FST::Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> WEIGHTER_SAMPLED;  //!< online per-feature statistics engine feeding the sampled (budgeted stochastic) sequential step
typedef FST::Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER_SAMPLED> EVAL_SAMPLED; //!< the sampled sequential step: budgeted stochastic candidate proposals driven by WEIGHTER_SAMPLED's online per-feature statistics (`--evaluator` sampled, EXPERIMENTAL)
typedef FST::Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> CRIT_BIAS; //!< the bias-estimate wrapper criterion decorator: estimates the wrapped classifier's train-vs-test optimism on the selected subset (`--bias-estimate`, demo56)
typedef FST::Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> ENGINE;   //!< the candidate-evaluation engine seam (in-line sequential or thread-pooled), injected into searches and step evaluators
typedef std::shared_ptr<ENGINE> PENGINE; //!< shared pointer to ENGINE
typedef FST::Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE> BBPREDICTOR; //!< averaging predictor used by the FBB/BBPP Branch & Bound variants (bb-fast/bb-bbpp) to decide predict-vs-compute per candidate
typedef FST::Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER; //!< the default (dupless) result tracker: best-k distinct subsets, feeds the top_k output block (`--tracker` dupless)
typedef FST::Result_Tracker_Regularizer<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,CRITERION> TRACKER_REG; //!< the regularizer result tracker: records candidates for the `--regularize-by`/`--regularize-margins` post-search sweep (`--tracker` regularizer)
typedef FST::Result_Tracker_Stability_Evaluator<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER_STABIL; //!< the stability-evaluator result tracker: per-trial stability statistics, requires `--repeat-over-splits` (`--tracker` stabileval)
typedef FST::Result_Tracker_Feature_Stats<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER_STATS; //!< the feature-stats result tracker: probe statistics behind `--daf-rank`/`--daf-pairs` (`--tracker` feature-stats)
typedef FST::Criterion_Subset_Size<RETURNTYPE,SUBSET> CRIT_SUBSIZE; //!< adapter criterion returning the evaluated subset's size; wrapped in CRIT_NEG to build the `--regularize-by` subsetsize secondary criterion (maximizing -size prefers smaller subsets)
typedef FST::Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET> CRIT_SUMW; //!< adapter criterion returning the sum of the selected features' `--feature-costs`; wrapped in CRIT_NEG to build the `--regularize-by` weights secondary criterion (maximizing -cost prefers cheaper subsets)
typedef FST::Criterion_Negative<CRITERION,RETURNTYPE,SUBSET> CRIT_NEG; //!< decorator criterion returning the negative of an inner criterion's value; used to turn CRIT_SUBSIZE/CRIT_SUMW's minimization goals into the maximization the regularizer sweep expects

//! Criterion decorator counting evaluate() calls; the counter is shared by
//! clone()s so evaluations performed by threaded-evaluator workers are included
class Criterion_Counting : public FST::Clonable_Impl<Criterion_Counting,CRITERION> {
public:
	typedef FST::Clonable_Impl<Criterion_Counting,CRITERION> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	//! Constructs the counting decorator around \p inner, sharing evaluation counts through \p count.
	Criterion_Counting(const std::shared_ptr<CRITERION> &inner, const std::shared_ptr<std::atomic<long long> > &count)
		: _inner(inner), _count(count) {}
	//! counts the evaluation, then delegates to the wrapped criterion; \return the wrapped criterion's verdict
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override {++(*_count); return _inner->evaluate(result,sub);}
	//! delegates to the wrapped criterion's print() -- the decorator is invisible in listings
	virtual std::ostream& print(std::ostream& os) const override {return _inner->print(os);}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! deep-clones the wrapped criterion but SHARES the atomic evaluation counter across clones
	Criterion_Counting(const Criterion_Counting& cc) : _inner(FST::clone_unique(*cc._inner)), _count(cc._count) {}
	std::shared_ptr<CRITERION> _inner; //!< the wrapped criterion whose evaluate() calls are counted
	std::shared_ptr<std::atomic<long long> > _count; //!< evaluation counter, shared across clone()s (threaded-evaluator workers included)
};

//! swallows all output (used with `--verbosity` silent)
class NullBuf : public std::streambuf {
protected:
	virtual int overflow(int c) override {return c;} //!< discards the character; streams into this buffer always report success
};

/*! `--data` -: the dataset arrives on stdin (the upstream half of the Unix-pipe story; the
    downstream half is the prose-free stdout of the machine output formats). A pipe is not seekable and is consumed once, while the
    readers need both: the TRN reader seeks back to the data start after the header scan,
    and the accessor is constructed again for a `--compare-config` second scenario. The
    bytes are therefore buffered ONCE into a temporary file that backs every reader and
    is removed when the run ends. std::filesystem (C++17) keeps this portable, so no
    second platform branch is introduced - extern_bridge.hpp stays the only
    platform-sensitive code in FST4. */
class Stdin_Data_File {
public:
	Stdin_Data_File()
	{
		namespace fs=std::filesystem;
		std::error_code ec;
		const fs::path dir=fs::temp_directory_path(ec);
		if(ec) throw FST::fst_error("--data -: no temporary directory available to buffer stdin ("+ec.message()+")");
		// Test for empty input BEFORE claiming a buffer file: operator<<(streambuf*) also
		// sets failbit when it inserts nothing, so testing after the write cannot tell an
		// empty stdin from a genuine write failure — and a throw here must not leave a
		// claimed file behind (the destructor never runs when the constructor throws)
		if(std::cin.peek()==std::char_traits<char>::eof())
			throw FST::fst_error("--data -: no data on stdin (expected a dataset in the --format given; "
			                     "pipe one in, e.g. 'cat data.trn | fst4 --data - --format trn ...')");
		// clock tick + retry counter names the file; the C11 exclusive-create flag ("x",
		// implemented by UCRT, glibc and macOS alike — one portable code path) makes the
		// CLAIM atomic: a separate exists()-probe before creating would let two fst4
		// processes starting within the same clock tick adopt the same buffer file
		const unsigned long long tick=(unsigned long long)std::chrono::system_clock::now().time_since_epoch().count();
		for(unsigned int attempt=0;;attempt++) {
			char name[64];
			std::snprintf(name,sizeof(name),"fst4_stdin_%llu_%u.tmp",tick,attempt);
			_path=(dir/name).string();
			FILE *claim=std::fopen(_path.c_str(),"wbx");
			if(claim) {std::fclose(claim); break;} // the name is exclusively ours; reopening it below is race-free
			if(errno!=EEXIST) throw FST::fst_error("--data -: cannot create the stdin buffer '"+_path+"' ("+std::strerror(errno)+")");
			if(attempt>=1000) throw FST::fst_error("--data -: cannot find an unused temporary file name in '"+dir.string()+"'");
		}
		{
			std::ofstream out(_path.c_str(),std::ios::binary); // truncates the just-claimed empty file
			if(!out) {remove_file(); throw FST::fst_error("--data -: cannot open '"+_path+"' to buffer stdin");}
			out << std::cin.rdbuf();
			out.close();
			if(!out) {remove_file(); throw FST::fst_error("--data -: writing the stdin buffer to '"+_path+"' failed (disk full?)");}
		}
	}
	~Stdin_Data_File() {remove_file();}
	const std::string& path() const {return _path;} //!< the buffer file's path; what the data-format readers actually open when `--data` -
private:
	Stdin_Data_File(const Stdin_Data_File&);            //!< the temporary file has one owner
	Stdin_Data_File& operator=(const Stdin_Data_File&); //!< the temporary file has one owner
	/*! must never throw: it also runs from the destructor. std::filesystem::remove's
	    error_code overload is noexcept, but constructing the path argument from _path
	    allocates, so the catch-all is what actually makes this destructor-safe */
	void remove_file() {try {std::error_code ec; std::filesystem::remove(_path,ec);} catch(...) {}}
	std::string _path; //!< path of the temporary file backing the buffered stdin
};

//! `--progress` json: emits one JSON line per improved solution to the progress stream; free-text progress is dropped
class Observer_JSON : public FST::Search_Observer<RETURNTYPE,SUBSET> {
public:
	explicit Observer_JSON(std::ostream &os) : _os(os) {} //!< emits JSON progress lines to \p os (the `--progress` stream)
	//! emits `{"event":"solution","value":…,"d":…,"features":[…]}` for the improved solution \p sub; \p msg (the text form) is dropped
	virtual void on_solution(const RETURNTYPE value, const SUBSET &sub, const std::string &msg) override {
		std::ostringstream sos;
		sos << "{\"event\":\"solution\",\"value\":" << value << ",\"d\":" << sub.get_d() << ",\"features\":[";
		bool first=true;
		for(DIMTYPE f : sub.selected_features()) {sos << (first?"":",") << f; first=false;}
		sos << "]}" << std::endl;
		FST::syncout::print(_os,sos);
	}
private:
	std::ostream &_os; //!< the progress stream (stderr for text/JSON/CSV results, per `--output`)
};

/*! `--report` detailed: records the best (value, subset) per subset size announced by the
    search (the curve_over_d source) while passing progress through
    unchanged — inner attached: everything is delegated; inner null: messages are printed
    exactly as they are without any observer (search.hpp report helpers), so
    attaching the recorder never changes the progress bytes. on_solution fires on the
    master thread only, so no synchronization is needed. */
class Observer_Curve : public FST::Search_Observer<RETURNTYPE,SUBSET> {
public:
	typedef std::shared_ptr<FST::Search_Observer<RETURNTYPE,SUBSET> > PInner; //!< the (possibly null) delegate observer wrapped by this recorder
	Observer_Curve(PInner inner, std::ostream &os) : _inner(inner), _os(os) {} //!< wraps \p inner (null = no delegate) and falls back to writing \p os directly when it is
	//! passes \p msg to the delegate, or prints it exactly as the no-observer path would
	virtual void on_progress(const std::string &msg) override {
		if(_inner) _inner->on_progress(msg);
		else {std::ostringstream sos; sos<<msg; FST::syncout::print(_os,sos);}
	}
	//! records \p sub as the best solution of its size when \p value beats the recorded one, then passes the event on like on_progress()
	virtual void on_solution(const RETURNTYPE value, const SUBSET &sub, const std::string &msg) override {
		const DIMTYPE d=sub.get_d_raw();
		auto it=_best.find(d);
		if(it==_best.end() || value>it->second.first) {
			std::shared_ptr<SUBSET> copy(new SUBSET(sub.get_n_raw()));
			copy->deselect_all();
			for(DIMTYPE f : sub.selected_features()) copy->select(f);
			_best[d]=std::make_pair(value,copy);
		}
		if(_inner) _inner->on_solution(value,sub,msg);
		else {std::ostringstream sos; sos<<msg; FST::syncout::print(_os,sos);}
	}
	const std::map<DIMTYPE,std::pair<RETURNTYPE,std::shared_ptr<SUBSET> > >& best_per_size() const {return _best;} //!< the curve_over_d source: best (value, subset) recorded per subset size so far
private:
	PInner _inner; //!< the delegate observer, or null when there is none
	std::ostream &_os; //!< fallback progress stream, used only when _inner is null
	std::map<DIMTYPE,std::pair<RETURNTYPE,std::shared_ptr<SUBSET> > > _best; //!< best (value, subset) per subset size seen so far; ordered: writers emit ascending d
};


// ---------------------------------------------------------------------------

/*! \brief All fst4 CLI settings in one place: one field per command-line flag (plus a
    few derived/internal fields such as data_resolved), populated by CLI11 in
    add_options() and consumed by the factory functions and run_pipeline() below.
    Unless noted, string fields default to "" (flag omitted / feature off); concrete
    defaults live in the Config() constructor below. */
struct Config {
	std::string data;              //!< `--data`: path to the data file (TRN/ARFF/CSV), or "-" to read the dataset from standard input (then `--format` is required); "" when `--features` is used instead (data-free entry)
	std::string data_resolved;     //!< `--data` -: the temporary file backing the buffered stdin; "" = `--data` is itself a path. The writers keep echoing `--data`, so the reported invocation stays "-" (and byte-stable)
	std::string format;            //!< `--format`: data file format (trn|arff|csv); "" = infer from extension
	std::string csv_class;         //!< CSV class column: "" = last column; integer = 0-based file column index; otherwise = header column name (implies a header row)
	std::string csv_delimiter;     //!< auto | comma | semicolon | tab (auto = detected from the first line; decimal commas accepted when delimiter != comma)
	std::string csv_header;        //!< auto | yes | no (auto = any non-numeric non-class field in line 1 means header)
	double csv_missing_to;         //!< NaN = off; else: the CSV loader maps missing FEATURE tokens to this code and keeps the row (pair with `--missing-value` \<same code\> for imputation; void scaler only; missing class labels still skip)
	std::string splitter;          //!< outer splitter (splitting depth 0)
	std::string splitter2;         //!< inner splitter (splitting depth 1); "" = single-level
	std::string scaler;            //!< `--scaler`: data normalization -- void (default, none) | to01 (per-feature to [0,1]) | white (whitening: zero mean, unit variance)
	std::string scaler_fit;        //!< all (default: fit on the whole dataset before splitting) | train (per-split: fit on each depth-0 TRAIN part only, so no test-part statistics leak into the transform)
	std::string criterion;         //!< `--criterion`: feature-selection criterion name (a normal-/multinomial-model filter or a wrapper-* classifier/regressor criterion; see `--list` for the full registry with validity notes)
	std::string knn_distance;      //!< `--knn-distance`: kNN distance metric -- L1 | L2 (default) | Lp (p=3/2, compile-time menu)
	std::string search;            //!< `--search`: search algorithm name (bif, sfs/sbs, sffs/sbfs, sfrs/sbrs, os, dos, exhaustive, montecarlo, bb-basic/bb-improved/bb-fast/bb-bbpp)
	std::string search2;           //!< seeding pre-search ("bif"): its result subset seeds the main search (demo25t/33)
	std::string svm_kernel;        //!< linear | rbf | poly | sigmoid (default rbf; LINEAR/SIGMOID may freeze LibSVM on some data)
	std::string svm_optimize;      //!< none | pre | alternating: LibSVM C/gamma/coef0 grid optimization (demo12t / demo23)
	std::string regr_multipliers;  //!< comma list: kernel-multiplier sweep, best-by-train wins (demo11reg)
	std::string evaluator;         //!< sequential step object: straight | hybrid (filter prefilter, demo53) | ensemble (voting, demo50) | sampled (budgeted stochastic step)
	std::string hybrid_prefilter;  //!< filter criterion pre-filtering candidates (with `--evaluator` hybrid)
	std::string ensemble_knn_ks;   //!< comma list of kNN k values forming the voting ensemble (with `--evaluator` ensemble)
	std::string step_sampler;      //!< sampled evaluator proposal policy: softmax | uniform | topk
	std::string step_stats;        //!< sampled evaluator per-feature statistic: zcontrast | rawmean
	std::string classify_samples;  //!< file of whitespace-separated samples to classify on the selected subspace (demo63)
	std::string validate;          //!< comma list of classifiers for final depth-0 validation; "" = none; "same" = the wrapper criterion's classifier
	std::string wrapper_metric;    //!< wrapper criterion metric beyond accuracy: balanced-accuracy | f1 | f1-macro | auc (classification) | r2 | neg-rmse | neg-mae (regression); "" = accuracy (default)
	std::string class_weights;     //!< comma list of per-class cost weights, class-index order (classification wrappers); "" = unweighted
	unsigned int f1_positive;      //!< class index scored as positive by `--wrapper-metric` f1 (two-class data; default 1)
	std::string tracker;           //!< dupless (default) | regularizer | stabileval | feature-stats
	std::string tracker_attach;    //!< "" = auto (search object; regularizer attaches to the evaluator per demo choreography)
	std::string regularize_by;     //!< secondary criterion for the regularization sweep: subsetsize | weights
	std::string regularize_margins;//!< comma list of margins for optimize_within_margin
	std::string feature_costs;     //!< comma list of per-feature costs (with `--regularize-by` weights)
	std::string output;            //!< `--output`: result format on stdout -- text (default) | json | csv; with json/csv all search progress goes to stderr
	std::string verbosity;         //!< `--verbosity`: search progress verbosity -- silent | normal (default) | detailed
	std::string progress_mode;     //!< search progress delivery: text (default) | json (one line per improved solution) | quiet (observer swallows progress)
	std::string compare_config;    //!< config file of a second scenario: run it too and report cross-scenario tracker similarity (the demo55 protocol)
	double missing_value;          //!< NaN = off; else: substitute this exact value by the per-feature average of the available values -- dataset-wide by default, the depth-0 TRAIN part's under `--scaler-fit` train (demo62; void scaler only)
	unsigned int cv_folds;          //!< `--cv-folds`: number of outer cross-validation folds (with `--splitter` cv; default 3)
	unsigned int holdout_train;    //!< `--holdout-train`: percentage of data used for outer training (with `--splitter` holdout; default 70)
	unsigned int rr_splits;   //!< `--rr-splits`: outer randrand/randfix run count (number of random splits)
	unsigned int rr_train;    //!< `--rr-train`: % of samples drawn for outer training
	unsigned int rr_test;     //!< `--rr-test`: % of samples drawn for outer testing (disjoint from training)
	unsigned int cv2_folds;        //!< `--splitter2-cv-folds`: number of inner cross-validation folds (with `--splitter2` cv; default 3)
	unsigned int holdout2_train;   //!< `--splitter2-holdout-train`: inner training percentage (with `--splitter2` holdout; default 70)
	unsigned int rr2_splits;  //!< `--splitter2-rr-splits`: inner randrand/randfix run count -- the `--splitter2` twin of rr_splits
	unsigned int rr2_train;   //!< `--splitter2-rr-train`: % of samples drawn for inner training -- the `--splitter2` twin of rr_train
	unsigned int rr2_test;    //!< `--splitter2-rr-test`: % of samples drawn for inner testing -- the `--splitter2` twin of rr_test
	unsigned int knn_k;            //!< `--knn-k`: k for the kNN classifier (with `--criterion` wrapper-knn; default 3)
	double ridge_lambda;           //!< ridge regression L2 weight penalty (default 0.01; bias unpenalized)
	std::string extern_cmd;        //!< external-tool bridge command (with `--criterion` wrapper-extern or extern-value); USER-SUPPLIED CODE EXECUTION — see classifier_extern.hpp / criterion_extern_value.hpp
	bool extern_monotonic;         //!< the user's assertion that the extern-value measure is cardinality-monotonic (admits bb-* searches; optimality rests on the assertion)
	bool extern_no_data;           //!< suppress the extern-value DATA block: the tool is a pure set function over index sets
	unsigned int features_n;       //!< `--features` N: data-free entry — an abstract universe of N features, no `--data` at all (extern-value `--extern-no-data` only; "Tier 2")
	double linsvm_c;               //!< linear SVM regularization C (default 1.0)
	std::string linsvm_loss;       //!< linear SVM loss: l2 (squared hinge, default) | l1 (hinge)
	unsigned int linsvm_maxiter;   //!< linear SVM dual CD sweep cap (default 1000)
	double linsvm_tol;             //!< linear SVM projected-gradient stopping threshold (default 0.1)
	std::string rbfsvm_optimize;   //!< none | pre | alternating: native RBF SVM C/gamma optimization (mirrors `--svm-optimize`)
	double rbfsvm_c;               //!< RBF SVM regularization C (default 1.0)
	double rbfsvm_gamma;           //!< RBF width on standardized features (default 0 = auto: 1/d)
	std::string rbfsvm_loss;       //!< RBF SVM loss: l1 (hinge, the C-SVC correspondence, default) | l2 (squared hinge)
	unsigned int rbfsvm_maxiter;   //!< RBF SVM kernel dual CD sweep cap (default 1000)
	double rbfsvm_tol;             //!< RBF SVM projected-gradient stopping threshold (default 0.1)
	unsigned int rf_trees;         //!< Random Forest wrapper: number of trees (default 100)
	unsigned int rf_mtry;          //!< features tried per node; 0 = floor(sqrt(subset size))
	unsigned int rf_max_depth;     //!< Random Forest tree depth cap; 0 = unlimited
	unsigned int rf_min_leaf;      //!< minimum samples per leaf (default 1)
	unsigned int search2_d;                  //!< seed subset size for `--search2` (0 = use `--target-d`)
	double svm_c;       //!< `--svm-c`: LibSVM regularization C (with `--criterion` wrapper-svm); 0 = LibSVM library default, or the value found by `--svm-optimize`
	double svm_gamma;   //!< `--svm-gamma`: LibSVM RBF/poly/sigmoid kernel width (with `--criterion` wrapper-svm); 0 = LibSVM library default, or the value found by `--svm-optimize`
	double svm_coef0;   //!< `--svm-coef0`: LibSVM poly/sigmoid kernel bias term (with `--criterion` wrapper-svm); 0 = LibSVM library default, or the value found by `--svm-optimize`
	double logreg_lambda;                    //!< logistic regression L2 weight penalty (biases unpenalized)
	unsigned int logreg_maxiter;             //!< logistic regression Newton iteration cap
	double logreg_tol;                       //!< logistic regression gradient-norm stopping tolerance
	double regr_multiplier;                  //!< weighted-neighbours kernel multiplier (0 = library default)
	unsigned int hybrid_keep_perc;           //!< hybrid evaluator: percentage of prefiltered candidates kept (demo53: 50)
	unsigned int hybrid_keep_max;            //!< hybrid evaluator: hard cap on kept candidates
	unsigned int step_cap;                   //!< sampled evaluator: max candidate evaluations per forward step (0 = no cap)
	unsigned int step_cap_backward;          //!< sampled evaluator: max candidate evaluations per backward step (0 = full sweeps, exact floating correction)
	double step_cap_frac;                    //!< sampled evaluator: constant-percentage schedule, cap = ceil(frac*pool) (0 = off, use step_cap); forward only
	bool step_frozen;                        //!< sampled evaluator: freeze statistics after warm-up (frozen-ranking ablation)
	double step_explore;                     //!< sampled evaluator: exploration-floor fraction of each proposal batch
	double step_tau;                         //!< sampled evaluator: softmax temperature (0 = robust automatic scale)
	unsigned int step_decay;                 //!< sampled evaluator: per-feature statistics averaging horizon (evaluations)
	unsigned long warmup_probes;             //!< sampled evaluator: random probes seeding the statistics before the search (0 = off)
	unsigned int warmup_card;                //!< sampled evaluator: warm-up probe cardinality
	unsigned int restarts_nonimproving;      //!< random-restart protocol: stop after N consecutive non-improving runs (0 = off; demo22)
	unsigned long restarts_max;              //!< random-restart safety cap on total runs (0 = unlimited)
	bool bias_estimate;                      //!< estimate classifier bias on the selected subset (demo56)
	unsigned int target_d;                   //!< target subset size (`--target-d`); 0 = optimize subset size (wrapper criteria only -- monotonic filter criteria and B&B/os require a nonzero value)
	unsigned int os_delta;                   //!< oscillation cycle depth (`--search` os; default 1)
	unsigned int dos_delta;                  //!< oscillation cycle depth (`--search` dos; default 3)
	unsigned int sffs_delta;                 //!< floating-frontier bound: SFFS/SBFS stop at target-d +/- delta (0 = classic full frontier)
	unsigned int generalization;             //!< g-tuples per sequential step (demo21); 1 = classic
	unsigned long long mc_trials;            //!< montecarlo stopping: max trials (0 = unlimited if mc_seconds set)
	unsigned long mc_seconds;                //!< montecarlo stopping: wall-clock budget (0 = none)
	unsigned int mc_card_min /*!< montecarlo probe cardinality range lower bound (with mc_card_max, 0,0 = full range) */, mc_card_max; //!< montecarlo probe cardinality range upper bound (with mc_card_min, 0,0 = full range)
	double mc_card_prob;                     //!< montecarlo per-feature inclusion probability (0 = off)
	double bb_gamma;                         //!< FBB prediction optimism (default 1.0)
	unsigned int bb_delta;                   //!< FBB minimum prediction evidence (default 1)
	unsigned long tracker_capacity;          //!< 0 = type default (dupless: top_k; others: unlimited)
	double tracker_margin;                   //!< dupless-family inclusion margin (0 = only improving results)
	int daf_rank;                            //!< DAF ranking variant to extract from feature-stats (-1 = off; 0|1)
	unsigned int daf_count;                  //!< number of top-DAF features to report (0 = all)
	unsigned int daf_validate_every;         //!< validate incrementally built top-DAF subsets every K steps (0 = off)
	unsigned int daf_pairs;                  //!< pairwise DAF statistics over the top-K DAF0 features (0 = off)
	unsigned int daf_pairs_top;              //!< number of top pairs (by pair-DAF0) to report (0 = all)
	std::string report;                      //!< standard (default) | detailed (adds the research sections: curve_over_d, ROC/PR, confusion, tuned parameters)
	std::string dump_scores;                 //!< file to receive per-holdout-sample true/predicted labels + per-class scores of the final subset (the first `--validate` classifier); "" = off
	std::string curves_svg;                  //!< file to receive a dependency-free SVG rendering of the `--report` detailed curves (ROC, PR, criterion-vs-holdout over d); "" = off
	unsigned int threads;                    //!< number of worker threads for candidate evaluation; 1 = sequential evaluation, deterministic ordering (default 1)
	unsigned int top_k;                      //!< number of best tracked subsets to report (default 5); also the dupless tracker's default capacity
	unsigned int seed;                       //!< random seed (`--seed`); affects randomized components only, e.g. the initial subset of `--search` os (default 1)
	bool stable_output;                      //!< omit non-deterministic output (timings) so results are byte-comparable across runs
	bool repeat_over_splits;                 //!< run the search once per outer split (trial protocol, demo54)
	Config() : csv_delimiter("auto"), csv_header("auto"), csv_missing_to(std::numeric_limits<double>::quiet_NaN()),
	           splitter("5050"), scaler("void"), scaler_fit("all"),
	           knn_distance("L2"), svm_kernel("rbf"), svm_optimize("none"), evaluator("straight"), step_sampler("softmax"),
	           step_stats("zcontrast"), f1_positive(1), tracker("dupless"), output("text"), verbosity("normal"),
	           progress_mode("text"), missing_value(std::numeric_limits<double>::quiet_NaN()), cv_folds(3),
	           holdout_train(70), rr_splits(1), rr_train(50), rr_test(50), cv2_folds(3), holdout2_train(70), rr2_splits(1),
	           rr2_train(50), rr2_test(50), knn_k(3), ridge_lambda(0.01), extern_monotonic(false), extern_no_data(false), features_n(0),
	           linsvm_c(1.0), linsvm_loss("l2"), linsvm_maxiter(1000), linsvm_tol(0.1), rbfsvm_optimize("none"),
	           rbfsvm_c(1.0), rbfsvm_gamma(0.0), rbfsvm_loss("l1"), rbfsvm_maxiter(1000), rbfsvm_tol(0.1), rf_trees(100),
	           rf_mtry(0), rf_max_depth(0), rf_min_leaf(1), search2_d(0), svm_c(0.0), svm_gamma(0.0), svm_coef0(0.0),
	           logreg_lambda(0.01), logreg_maxiter(100), logreg_tol(1e-6), regr_multiplier(0.0), hybrid_keep_perc(50),
	           hybrid_keep_max(1000), step_cap(100), step_cap_backward(0), step_cap_frac(0.0), step_frozen(false),
	           step_explore(0.2), step_tau(0.0), step_decay(100), warmup_probes(0), warmup_card(25),
	           restarts_nonimproving(0), restarts_max(0), bias_estimate(false), target_d(0), os_delta(1), dos_delta(3),
	           sffs_delta(0), generalization(1), mc_trials(1000), mc_seconds(0), mc_card_min(0), mc_card_max(0),
	           mc_card_prob(0.0), bb_gamma(1.0), bb_delta(1), tracker_capacity(0), tracker_margin(0.0), daf_rank(-1),
	           daf_count(0), daf_validate_every(0), daf_pairs(0), daf_pairs_top(50), report("standard"), threads(1),
	           top_k(5), seed(1), stable_output(false), repeat_over_splits(false) {}
	bool detailed_report() const {return report=="detailed";} //!< true for `--report` detailed (the research sections; standard output is unchanged by them)
	bool data_is_stdin() const {return data=="-";} //!< true for `--data` - (dataset piped in on standard input)
	bool data_free() const {return features_n>0;} //!< true for `--features` N: no data file exists, the universe is abstract
	//! what the readers open: the stdin buffer when `--data` -, the given path otherwise
	const std::string& data_path() const {return data_resolved.empty()?data:data_resolved;}
	bool has_missing_value() const {return missing_value==missing_value;} //!< true when `--missing-value` was given (NaN-off sentinel, self-compare)
	bool has_csv_missing_to() const {return csv_missing_to==csv_missing_to;} //!< true when `--csv-missing-to` was given (NaN-off sentinel, self-compare)
	bool is_filter_criterion() const {return criterion=="bhattacharyya" || criterion=="divergence" || criterion=="gmahalanobis" || criterion=="multinom-bhattacharyya" || criterion=="multinom-mi";} //!< true for the five built-in statistical-model filters (no classifier training)
	bool is_wrapper_criterion() const {return criterion.compare(0,8,"wrapper-")==0;} //!< true for any "wrapper-*" criterion (classifier- or regressor-based)
	bool is_regression_criterion() const {return criterion=="wrapper-regressor" || criterion=="wrapper-ridge";} //!< true for the two regression wrappers (dependent variable = the class label, "regression simulating classification")
	bool has_wrapper_metric() const {return (!wrapper_metric.empty() && wrapper_metric!="accuracy") || !class_weights.empty();} //!< false = the plain-accuracy path; true when `--wrapper-metric`/`--class-weights` select the metric palette
	bool nested_splits() const {return !splitter2.empty();} //!< true when `--splitter2` defines an inner (depth-1) splitting stack
	bool is_bb_search() const {return search.compare(0,3,"bb-")==0;} //!< true for any of the four Branch & Bound variants (`--search` bb-*)
	//! true for the sequential-family searches (sfs/sbs/sffs/sbfs/sfrs/sbrs/os/dos) -- the ones driven through an EVALUATOR/`--evaluator`
	bool is_sequential_search() const {return search=="sfs"||search=="sbs"||search=="sffs"||search=="sbfs"
	                                       ||search=="sfrs"||search=="sbrs"||search=="os"||search=="dos";}
};

//! builds the `--progress` observer; empty pointer = default text delivery (the library's own progress text)
static std::shared_ptr<FST::Search_Observer<RETURNTYPE,SUBSET> > make_observer(const Config &cfg, std::ostream &progress)
{
	if(cfg.progress_mode=="quiet") return std::make_shared<FST::Search_Observer<RETURNTYPE,SUBSET> >(); // no-op observer swallows progress
	if(cfg.progress_mode=="json") return std::make_shared<Observer_JSON>(progress);
	return std::shared_ptr<FST::Search_Observer<RETURNTYPE,SUBSET> >(); // "text"
}

//! everything the run phase needs, assembled by the factories below
struct Pipeline {
	std::shared_ptr<ACCESSOR_MEM> da_mem;  //!< concrete-side handle: initialize(), setSplittingDepth()
	std::shared_ptr<DATAACCESSOR> da;      //!< the type-erased handle every component sees
	std::shared_ptr<CRITERION> crit;       //!< counting decorator around the selected criterion
	std::shared_ptr<CLASSIFIER> clf;       //!< the wrapper criterion's classifier (empty for filters); `--validate` same reuses it
	std::shared_ptr<REGRESSOR> regr;       //!< the type-erased regressor handle (`--validate` same on the regression pipelines)
	std::string regr_label;                //!< its validation-entry label ("regressor" | "ridge")
	std::shared_ptr<RWN> rwn;              //!< concrete regressor handle (only where the concrete API is needed: the kernel-multiplier sweep)
#ifdef FST_SVM
	std::shared_ptr<SVM> svm;              //!< concrete SVM handle (parameter optimization needs the non-erased type)
#endif
	std::shared_ptr<RBFSVM> rbfsvm;        //!< concrete native RBF SVM handle (`--rbfsvm-optimize` needs the non-erased type)
	std::shared_ptr<std::atomic<long long> > evals; //!< shared evaluation counter, incremented by the counting criterion decorator; feeds the "evaluations" stats field
	std::function<bool(RETURNTYPE&,const std::shared_ptr<SUBSET>&)> split_eval; //!< re-evaluates a subset on the accessor's current split
	std::function<void()> crit_reinit;     //!< re-estimates a filter criterion's model on the current split (per-trial re-init, demo55-B); empty for wrappers
	std::shared_ptr<SEARCH> search;        //!< the constructed search algorithm
	std::shared_ptr<TRACKER> tracker;              //!< dupless-family handle (top_k source); null for stabileval
	std::shared_ptr<TRACKER_REG> tracker_reg;      //!< regularizer handle (optimize_within_margin)
	std::shared_ptr<TRACKER_STABIL> tracker_stabil;//!< stability-evaluator handle (manual add per trial)
	std::shared_ptr<TRACKER_STATS> tracker_stats;  //!< feature-stats handle (compute_stats, DAF extraction)
	std::shared_ptr<WEIGHTER_SAMPLED> sampled_stats; //!< sampled-evaluator statistics engine (the warm-up probe loop feeds it)
	PENGINE sampled_engine;                          //!< the sampled evaluator's candidate-evaluation engine; the threaded warm-up loop reuses it so criterion clones persist into the search
	std::vector<PENGINE> engines;                    //!< every candidate-evaluation engine built for this run (the search's and the step evaluator's), registered by make_engine() so that invalidate_engine_clones() reaches them all
	/*! \brief Drops the cached per-worker criterion clones of every engine (see Candidate_Evaluator::invalidate_clones()).
	    The threaded engine re-uses its clones across batches while the same criterion object is supplied, so
	    once the ORIGINAL criterion's evaluation context has changed - the next trial's depth-0 split, a
	    re-estimated filter model, a re-tuned classifier or regressor parameter - the clones must be re-made
	    from it, or the workers keep evaluating in the old context: without this call a threaded trial
	    protocol repeats its first trial in every trial. A no-op for the sequential engine. */
	void invalidate_engine_clones() const {for(std::vector<PENGINE>::const_iterator e=engines.begin();e!=engines.end();++e) (*e)->invalidate_clones();}
};

/*! \def FST4_STAB(NAME,EXPR)
    \brief Appends one RunResult::StabilityEntry{NAME, EXPR} to r.stability.

    Local sugar for the `--repeat-over-splits` post-processing block of main() only:
    reused as-is by 11 back-to-back invocations there, one per stabileval measure
    (value_mean/value_stddev/size_mean/size_stddev/CWrel/ATI/CW/ANHI/SH/PH/C),
    each calling a different Data_Splitter_Stabileval query on p.tracker_stabil.
    Defined immediately before its first use and \#undef'd immediately after its
    last -- it has no meaning outside that block. \p e is the enclosing scope's
    RunResult::StabilityEntry; each invocation is a full statement (the macro body
    supplies the trailing semicolons), so it is used as `FST4_STAB("name", expr)`
    with no semicolon of its own. */

//! everything one run_pipeline() call produces: the selected subset plus every optional
//! block gated by its own CLI flag (all empty/default when the flag is not given, so the
//! emitters skip them uniformly across text/json/csv)
struct RunResult {
	std::vector<DIMTYPE> features; //!< the final selected subset, sorted ascending
	RETURNTYPE value;               //!< the final subset's criterion value
	std::vector<RETURNTYPE> per_split;     //!< filled only when the criterion-protocol splitter yields >1 split
	//! one classifier's/regressor's holdout validation outcome (`--validate`)
	struct ValidationEntry {
		std::string classifier;             /*!< which `--validate` name (or "same"'s resolved name / the regression label) produced this entry */
		RETURNTYPE accuracy;                /*!< plain classification accuracy or regression test() value (always reported) */
		std::string metric;                 /*!< selected `--wrapper-metric` name, empty = accuracy only */
		RETURNTYPE metric_value=0.0;        /*!< the selected metric's value; meaningless when metric is empty */
		std::vector<RETURNTYPE> confusion;  /*!< row-major confusion_classes x confusion_classes matrix (row = true class); `--report` detailed final validation only */
		DIMTYPE confusion_classes=0;        /*!< matrix dimension of confusion; 0 = not filled */
	};
	std::vector<ValidationEntry> validation; //!< `--validate`: train on depth-0 train part, test on depth-0 test part; reports accuracy plus the selected `--wrapper-metric`
	//! one entry of the dupless tracker's `--top-k` best distinct subsets
	struct TopEntry {RETURNTYPE value; /*!< the subset's criterion value */ std::vector<DIMTYPE> features; /*!< the subset itself */};
	std::vector<TopEntry> top_k; //!< the `--top-k` best distinct tracked subsets (dupless tracker), best first
	//! one `--regularize-margins` sweep point: the best subset within `margin` of the maximum by the secondary criterion
	struct RegEntry {
		double margin;                          /*!< the sweep margin this entry belongs to (one of `--regularize-margins`) */
		RETURNTYPE value;                       /*!< the chosen subset's primary criterion value */
		RETURNTYPE secondary;                   /*!< its secondary-criterion value: weight sum (`--regularize-by` weights) or subset size (subsetsize) */
		std::vector<DIMTYPE> features;          /*!< the chosen subset */
		std::vector<ValidationEntry> validation;/*!< its `--validate` results, if any */
	};
	std::vector<RegEntry> regularization;    //!< `--regularize-margins` sweep results
	//! one `--repeat-over-splits` trial's search outcome
	struct TrialEntry {RETURNTYPE value; /*!< the trial's criterion value */ std::vector<DIMTYPE> features; /*!< the trial's selected subset */};
	std::vector<TrialEntry> trials;          //!< `--repeat-over-splits` per-trial results
	//! one named stability/similarity measure computed by the stabileval tracker (see the FST4_STAB macro)
	struct StabilityEntry {std::string measure; /*!< measure name, e.g. "CWrel", "ATI" (see FST4_STAB) */ RETURNTYPE value; /*!< the measure's value */};
	std::vector<StabilityEntry> stability;   //!< stabileval measures over the trials
	//! one `--daf-rank` ranked feature
	struct DafEntry {DIMTYPE feature; /*!< the feature index */ RETURNTYPE value; /*!< its DAF ranking value */};
	std::vector<DafEntry> daf;               //!< `--daf-rank`: DAF ranking from feature-stats
	std::vector<ValidationEntry> daf_validation; //!< `--daf-validate-every`: accuracy of incrementally built top-DAF subsets (classifier field holds the subset size)
	//! one `--daf-pairs` pairwise (2-feature) DAF statistic
	struct PairDafEntry {
		DIMTYPE f1;                 /*!< first feature of the pair */
		DIMTYPE f2;                 /*!< second feature of the pair */
		RETURNTYPE daf0;            /*!< pair-DAF0 presence contrast (always valid) */
		RETURNTYPE daf1;            /*!< pair-DAF1 presence contrast (only when daf1_valid) */
		RETURNTYPE synergy;         /*!< the pair's synergy statistic */
		bool daf1_valid;            /*!< whether daf1 could be computed for this pair */
	};
	std::vector<PairDafEntry> daf_pairs;     //!< `--daf-pairs`: pairwise DAF statistics (top pairs by pair-DAF0)
	//! one `--regr-multipliers` sweep point (demo11reg)
	struct SweepEntry {
		double multiplier;                       /*!< the kernel multiplier this entry was run with */
		RETURNTYPE value;                        /*!< its criterion (train) value */
		std::vector<DIMTYPE> features;           /*!< the subset selected under this multiplier */
		std::vector<ValidationEntry> validation; /*!< its `--validate` results, if any */
	};
	std::vector<SweepEntry> multiplier_sweep;//!< `--regr-multipliers` sweep results (demo11reg)
	double best_multiplier;                  //!< the sweep's best-by-train-value multiplier
	RETURNTYPE bias;                         //!< `--bias-estimate` result
	bool has_bias;                           //!< whether bias was filled (`--bias-estimate` was given)
	std::vector<DIMTYPE> classified;         //!< `--classify-samples` predicted classes
	// `--report` detailed sections (all empty in standard mode)
	//! one ROC/PR point at a given decision threshold (see extract_curves())
	struct CurvePoint {RETURNTYPE threshold, /*!< the score threshold this point was taken at */ x, /*!< roc: false-positive rate; pr: recall */ y; /*!< roc: true-positive rate; pr: precision */};
	//! one class's ROC and PR curves (see extract_classifier_curves())
	struct ClassCurve {DIMTYPE cls; /*!< the class these curves are for (one-vs-rest on K>2 data) */ std::vector<CurvePoint> roc, /*!< the ROC curve */ pr; /*!< the PR curve */};
	std::vector<ClassCurve> curves;          //!< ROC/PR of the first `--validate` classifier on the depth-0 holdout
	std::string curves_classifier;           //!< which classifier produced the curves
	//! one curve_over_d point: the best subset of a given size plus its holdout re-validation (the overfitting curve)
	struct CurveEntry {
		DIMTYPE d;                               /*!< the subset size */
		RETURNTYPE value;                        /*!< its best criterion (train) value */
		std::vector<DIMTYPE> features;           /*!< the subset itself */
		std::vector<ValidationEntry> validation; /*!< its holdout re-validation, if `--validate` is set */
	};
	std::vector<CurveEntry> curve_over_d;    //!< best-per-size search trajectory (criterion axis) + holdout re-validation (validated axis)
	std::string tuned_which;                 //!< "svm"|"rbfsvm" after an optimize protocol; empty = none
	double tuned_c=0.0 /*!< the tuned C parameter */, tuned_gamma=0.0 /*!< the tuned gamma parameter */, tuned_coef0=0.0; //!< the tuned coef0 parameter, meaningful only when tuned_has_coef0
	bool tuned_has_coef0=false; //!< whether tuned_coef0 applies (poly/sigmoid LibSVM kernels only)
	double wall_time_s;                      //!< wall-clock seconds spent in run_pipeline(); omitted from output under `--stable-output`
	long long evaluations;                   //!< total criterion evaluations (Pipeline::evals); omitted from output under `--stable-output`
	RunResult() : value(0), best_multiplier(0), bias(0), has_bias(false), wall_time_s(0), evaluations(0) {}
};

//! everything produced by a `--compare-config` second scenario (the demo55 protocol)
struct CompareResult {
	Config cfg2;      //!< the second scenario's configuration (parsed from the config file)
	Pipeline p2;      //!< its pipeline (holds the stabileval tracker the similarity is computed from)
	RunResult r2;     //!< its full run result (trials + per-scenario stability block)
	std::vector<RunResult::StabilityEntry> similarity; //!< IATI/ICW/IANHI: scenario A tracker vs scenario B tracker
};

//! splits a comma-separated flag value into tokens; empty fields are kept and an empty input yields one empty token
static std::vector<std::string> split_csv(const std::string &s)
{
	std::vector<std::string> out;
	std::string::size_type start=0;
	while(start<=s.size()) {
		std::string::size_type comma=s.find(',',start);
		if(comma==std::string::npos) {out.push_back(s.substr(start)); break;}
		out.push_back(s.substr(start,comma-start));
		start=comma+1;
	}
	return out;
}

//! splits a comma-separated flag value into doubles, reporting the offending flag on parse failure
static std::vector<double> split_csv_doubles(const std::string &s, const std::string &flag)
{
	std::vector<std::string> tokens=split_csv(s);
	std::vector<double> out;
	for(std::vector<std::string>::const_iterator it=tokens.begin();it!=tokens.end();++it) {
		char *end=0;
		const double v=std::strtod(it->c_str(),&end);
		if(end==it->c_str() || *end!='\0')
			throw FST::fst_error(flag+": '"+*it+"' is not a number");
		out.push_back(v);
	}
	return out;
}

//! builds the wrapper Metric_Spec from the CLI flags; the class-count-dependent
//! checks (weights arity, two-class metrics) need the loaded data
static FST::Metric_Spec<RETURNTYPE> make_metric_spec(const Config &cfg, const DIMTYPE classes)
{
	FST::Metric_Spec<RETURNTYPE> spec;
	if(!cfg.wrapper_metric.empty() && !FST::wrapper_metric_from_name(cfg.wrapper_metric,spec.metric))
		throw FST::fst_error("unknown --wrapper-metric '"+cfg.wrapper_metric+"'"); // unreachable: the CLI membership check guards
	if(!cfg.class_weights.empty()) {
		std::vector<double> w=split_csv_doubles(cfg.class_weights,"--class-weights");
		if(w.size()!=(size_t)classes)
			throw FST::fst_error("--class-weights needs exactly one weight per class (the data has "+std::to_string((long)classes)+" classes, "+std::to_string(w.size())+" weights given)");
		for(size_t i=0;i<w.size();i++) {
			if(!(w[i]>0.0)) throw FST::fst_error("--class-weights must be positive numbers");
			spec.class_weights.push_back((RETURNTYPE)w[i]);
		}
	}
	spec.f1_positive=cfg.f1_positive;
	// auc covers any class count: the exact binary AUC on two-class data, the one-vs-rest macro AUC on K>2
	if(spec.metric==FST::Wrapper_Metric::f1 && classes!=2)
		throw FST::fst_error("--wrapper-metric f1 is the positive-class F1 on two-class data (positive = class index 1 unless --f1-positive says otherwise); use f1-macro on multiclass data");
	if(spec.metric==FST::Wrapper_Metric::f1 && cfg.f1_positive>=(unsigned int)classes)
		throw FST::fst_error("--f1-positive "+std::to_string(cfg.f1_positive)+" is not a class index of this two-class data (FST class order is the data-file class order; use 0 or 1)");
	return spec;
}

// ---------------------------------------------------------------------------

//! extracts a subset's selected feature indices as a plain, ascending-sorted vector --
//! the shape every output emitter (text/json/csv) and result-file writer works from
static std::vector<DIMTYPE> subset_features(const std::shared_ptr<SUBSET> &sub)
{
	std::vector<DIMTYPE> v;
	for(DIMTYPE f : sub->selected_features()) v.push_back(f);
	std::sort(v.begin(),v.end());
	return v;
}

//! formats a double at fixed 12-significant-digit precision -- the one number formatter
//! every emitter uses, so numeric output stays consistent (and reproducible)
//! across text/json/csv
static std::string fmt_double(const double v)
{
	char buf[48];
	std::snprintf(buf,sizeof(buf),"%.12g",v);
	return std::string(buf);
}

//! escapes a string for embedding inside a JSON string literal (RFC 8259: quotes, backslash,
//! the named short escapes, and all remaining control characters via \\uXXXX)
static std::string json_escape(const std::string &s)
{
	std::string out;
	for(std::string::const_iterator it=s.begin();it!=s.end();++it) {
		switch(*it) {
			case '"': out+="\\\""; break;
			case '\\': out+="\\\\"; break;
			case '\n': out+="\\n"; break;
			case '\t': out+="\\t"; break;
			case '\r': out+="\\r"; break;
			case '\b': out+="\\b"; break;
			case '\f': out+="\\f"; break;
			default:
				if((unsigned char)*it<0x20) { // remaining control chars (RFC 8259 requires escaping all of 0x00-0x1F)
					char buf[8];
					std::snprintf(buf,sizeof(buf),"\\u%04x",(unsigned int)(unsigned char)*it);
					out+=buf;
				}
				else out+=*it;
		}
	}
	return out;
}

//! infers the data format from a file's extension (.trn/.arff/.csv/.tsv, case-insensitive);
//! throws when the extension is missing or unrecognized -- the caller then needs an explicit `--format`
static std::string infer_format(const std::string &path)
{
	const std::string::size_type dot=path.find_last_of('.');
	if(dot!=std::string::npos) {
		std::string ext=path.substr(dot+1);
		std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
		if(ext=="trn") return "trn";
		if(ext=="arff") return "arff";
		if(ext=="csv" || ext=="tsv") return "csv"; // .tsv = CSV with the tab delimiter (auto-detected)
	}
	throw FST::fst_error("cannot infer data format from file name '"+path+"'; specify --format trn|arff|csv");
}

// ---------------------------------------------------------------------------
// factories: string -> constructed component; initialize() choreography happens
// here, where the concrete types are still known

//! constructs the `--scaler` by name; the void scaler additionally implements `--missing-value`
//! substitution (its two-argument constructor), so `--missing-value` with any other scaler is
//! rejected here rather than silently ignored
//! \param cfg the parsed configuration (reads cfg.scaler and cfg.missing_value)
//! \return the constructed scaler; throws FST::fst_error on an unknown `--scaler` name or an
//!         unsupported `--missing-value`/`--scaler` combination
static std::shared_ptr<FST::Data_Scaler<DATATYPE> > make_scaler(const Config &cfg)
{
	if(cfg.scaler=="void") {
		// missing-value substitution (demo62): values equal to the code are replaced by the
		// per-feature average of the available values (dataset-wide, or per depth-0 TRAIN part under
		// `--scaler-fit` train); implemented by the void scaler's two-argument ctor
		if(cfg.has_missing_value()) return std::shared_ptr<FST::Data_Scaler<DATATYPE> >(new FST::Data_Scaler_void<DATATYPE>((DATATYPE)cfg.missing_value,1));
		return std::shared_ptr<FST::Data_Scaler<DATATYPE> >(new FST::Data_Scaler_void<DATATYPE>());
	}
	if(cfg.has_missing_value()) throw FST::fst_error("--missing-value substitution is implemented by the void scaler; drop --scaler "+cfg.scaler);
	if(cfg.scaler=="to01") return std::shared_ptr<FST::Data_Scaler<DATATYPE> >(new FST::Data_Scaler_to01<DATATYPE>());
	if(cfg.scaler=="white") return std::shared_ptr<FST::Data_Scaler<DATATYPE> >(new FST::Data_Scaler_white<DATATYPE,IDXTYPE>());
	throw FST::fst_error("unknown scaler '"+cfg.scaler+"'");
}

//! constructs the splitter for one splitting depth; inner=true selects the `--splitter2`-* parameter set
static PSPLITTER make_splitter_named(const std::string &name, const bool inner, const Config &cfg)
{
	const unsigned int folds   = inner ? cfg.cv2_folds : cfg.cv_folds;
	const unsigned int hold    = inner ? cfg.holdout2_train : cfg.holdout_train;
	const unsigned int rrs     = inner ? cfg.rr2_splits : cfg.rr_splits;
	const unsigned int rrtrain = inner ? cfg.rr2_train : cfg.rr_train;
	const unsigned int rrtest  = inner ? cfg.rr2_test : cfg.rr_test;
	const std::string flag = inner ? "--splitter2" : "--splitter";
	if(name=="5050") return PSPLITTER(new SPLITTER5050());
	if(name=="cv") return PSPLITTER(new SPLITTERCV(folds));
	if(name=="holdout") return PSPLITTER(new SPLITTERHOLDOUT(hold));
	if(name=="resub") return PSPLITTER(new SPLITTERRESUB());
	if(name=="leave1out") return PSPLITTER(new SPLITTERL1O());
	if(name=="randrand" || name=="randfix") {
		if(rrtrain+rrtest>100)
			throw FST::fst_error(flag+" randrand/randfix train and test percentages must sum to at most 100 (got "
			                     +std::to_string((long long)rrtrain)+"+"+std::to_string((long long)rrtest)+")");
		if(name=="randrand") return PSPLITTER(new SPLITTERRR(rrs,rrtrain,rrtest));
		return PSPLITTER(new SPLITTERRANDFIX(rrs,rrtrain,rrtest));
	}
	throw FST::fst_error("unknown splitter '"+name+"'");
}

//! constructs and initialize()s the data accessor for the resolved \p format: builds the
//! outer (and, under `--splitter2`, inner) splitter stack and the scaler, then dispatches to
//! the data-free universe accessor or one of the three file-format loaders (trn/arff/csv)
//! \param cfg the resolved CLI configuration (data path, splitter/scaler/format flags, etc.)
//! \param format the resolved format string ("trn"|"arff"|"csv"), never used for the data-free entry
//! \return the initialize()d accessor; throws FST::fst_error on an unknown format
static std::shared_ptr<ACCESSOR_MEM> make_accessor(const Config &cfg, const std::string &format)
{
	std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>);
	splitters->push_back(make_splitter_named(cfg.splitter,false,cfg));   // splitting depth 0 (outer)
	if(cfg.nested_splits()) splitters->push_back(make_splitter_named(cfg.splitter2,true,cfg)); // depth 1 (inner)
	std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc=make_scaler(cfg);
	std::shared_ptr<ACCESSOR_MEM> acc;
	// `--features`: the data-free universe accessor; the splitter/scaler arguments exist for
	// structural compatibility only (Data_Accessor_Universe never exercises them)
	if(cfg.data_free()) acc.reset(new ACCESSOR_UNIVERSE(cfg.features_n,splitters,dsc));
	// data_path(): the stdin buffer under `--data` -, the given path otherwise
	else if(format=="trn") acc.reset(new ACCESSOR_TRN(cfg.data_path(),splitters,dsc));
	else if(format=="arff") acc.reset(new ACCESSOR_ARFF(cfg.data_path(),splitters,dsc));
	else if(format=="csv") {
		FST::Data_File_CSV_Options csvopts;
		csvopts.class_column=cfg.csv_class;
		csvopts.delimiter=cfg.csv_delimiter;
		csvopts.header=cfg.csv_header;
		// `--csv-missing-to`: missing feature tokens become the code at load; substitution (if any)
		// is the void scaler's job via `--missing-value` — the loader mapping is deliberately pure
		if(cfg.has_csv_missing_to()) {
			csvopts.missing_to_active=true; csvopts.missing_to=(float)cfg.csv_missing_to;
			csvopts.missing_to_unpaired=!cfg.has_missing_value(); // no substitution configured: the summary says so
		}
		acc.reset(new ACCESSOR_CSV(cfg.data_path(),csvopts,splitters,dsc));
	}
	else throw FST::fst_error("unknown data format '"+format+"'");
	// `--scaler-fit` train: per-split scaling (fit on each depth-0 TRAIN part; closes the
	// normalization leakage of whole-dataset fitting). Plain void scaling is the identity,
	// so the mode is enabled only when it can have an effect (to01/white, or void+`--missing-value`)
	if(cfg.scaler_fit=="train" && (cfg.scaler!="void" || cfg.has_missing_value()))
		acc->set_scaler_fit_train(true);
	return acc;
}

//! constructs a kNN classifier with k=\p k and the `--knn-distance` metric (L1/L2/Lp);
//! shared by `--criterion` wrapper-knn, `--validate`, and the ensemble evaluator's per-k members
static std::shared_ptr<CLASSIFIER> make_knn(const Config &cfg, const unsigned int k)
{
	if(cfg.knn_distance=="L2") {std::shared_ptr<KNN_L2> c(new KNN_L2); c->set_k(k); return c;}
	if(cfg.knn_distance=="L1") {std::shared_ptr<KNN_L1> c(new KNN_L1); c->set_k(k); return c;}
	if(cfg.knn_distance=="Lp") {std::shared_ptr<KNN_LP> c(new KNN_LP); c->set_k(k); return c;}
	throw FST::fst_error("unknown kNN distance '"+cfg.knn_distance+"'");
}

//! constructs an initialize()d filter criterion by name (hybrid-evaluator prefilter)
static std::shared_ptr<CRITERION> make_filter(const std::string &name, const std::shared_ptr<DATAACCESSOR> &da)
{
	if(name=="bhattacharyya") {std::shared_ptr<CRIT_BHAT> c(new CRIT_BHAT); c->initialize(da); return c;}
	if(name=="divergence") {std::shared_ptr<CRIT_DIV> c(new CRIT_DIV); c->initialize(da); return c;}
	if(name=="gmahalanobis") {std::shared_ptr<CRIT_GMAHA> c(new CRIT_GMAHA); c->initialize(da); return c;}
	if(name=="multinom-bhattacharyya") {std::shared_ptr<CRIT_MBHAT> c(new CRIT_MBHAT); c->initialize(da); return c;}
	if(name=="multinom-mi") {std::shared_ptr<CRIT_MMI> c(new CRIT_MMI); c->initialize(da); return c;}
	throw FST::fst_error("unknown filter criterion '"+name+"' (available: bhattacharyya, divergence, gmahalanobis, multinom-bhattacharyya, multinom-mi)");
}

//! constructs a classifier by name (used by wrapper criteria and by `--validate`); when \p pipe
//! is given, the concrete LibSVM / native RBF-SVM handle is stashed on it for the optimize protocols
static std::shared_ptr<CLASSIFIER> make_classifier(const std::string &name, const Config &cfg, const std::shared_ptr<DATAACCESSOR> &da,
	Pipeline *pipe=0)
{
	if(name=="knn") return make_knn(cfg,cfg.knn_k);
	if(name=="bayes") {
		std::shared_ptr<BAYES> c(new BAYES);
		c->initialize(da); // pre-computes model structures (demo26 choreography)
		return c;
	}
	if(name=="multinom") {
		std::shared_ptr<MULTINOM_NB> c(new MULTINOM_NB);
		c->initialize(da); // multinomial model buffers (demo30 choreography)
		return c;
	}
	if(name=="logreg") {
		std::shared_ptr<LOGREG> c(new LOGREG);
		c->set_lambda(cfg.logreg_lambda);
		c->set_max_iter(cfg.logreg_maxiter);
		c->set_tolerance(cfg.logreg_tol);
		c->initialize(da); // records data shape, allocates the index buffer
		return c;
	}
	if(name=="lda") {
		std::shared_ptr<LDA> c(new LDA);
		c->initialize(da); // pre-allocates model and matrix buffers
		return c;
	}
	if(name=="extern") {
		std::shared_ptr<EXTERN> c(new EXTERN);
		c->set_command(cfg.extern_cmd);
		c->initialize(da); // records data shape; the child process spawns lazily on first train()
		return c;
	}
	if(name=="linsvm") {
		std::shared_ptr<LINSVM> c(new LINSVM);
		c->set_parameter_C((REALTYPE)cfg.linsvm_c);
		c->set_l2_loss(cfg.linsvm_loss=="l2");
		c->set_max_iter(cfg.linsvm_maxiter);
		c->set_tolerance((REALTYPE)cfg.linsvm_tol);
		c->initialize(da); // records data shape, allocates the index buffer
		return c;
	}
	if(name=="rbfsvm") {
		std::shared_ptr<RBFSVM> c(new RBFSVM);
		c->set_parameter_C((REALTYPE)cfg.rbfsvm_c);
		c->set_parameter_gamma((REALTYPE)cfg.rbfsvm_gamma); // 0 = auto: 1/d on standardized features
		c->set_l2_loss(cfg.rbfsvm_loss=="l2");
		c->set_max_iter(cfg.rbfsvm_maxiter);
		c->set_tolerance((REALTYPE)cfg.rbfsvm_tol);
		c->initialize(da); // records data shape, allocates the index buffer
		if(pipe) pipe->rbfsvm=c;
		return c;
	}
	if(name=="rf") {
		std::shared_ptr<RF> c(new RF);
		c->set_tree_count(cfg.rf_trees);
		c->set_mtry((DIMTYPE)cfg.rf_mtry);
		c->set_max_depth(cfg.rf_max_depth);
		c->set_min_leaf((IDXTYPE)cfg.rf_min_leaf);
		c->set_seed_salt(cfg.seed); // training seeds derive from (subset content, `--seed`, split index) — pure-function evaluations
		c->initialize(da); // records data shape, allocates the index buffer
		return c;
	}
	if(name=="svm") {
#ifdef FST_SVM
		std::shared_ptr<SVM> c(new SVM);
		if(cfg.svm_kernel=="linear") c->set_kernel_type(LINEAR);
		else if(cfg.svm_kernel=="rbf") c->set_kernel_type(RBF);
		else if(cfg.svm_kernel=="poly") c->set_kernel_type(POLY);
		else if(cfg.svm_kernel=="sigmoid") c->set_kernel_type(SIGMOID);
		else throw FST::fst_error("unknown SVM kernel '"+cfg.svm_kernel+"'");
		c->initialize(da); // pre-allocates data transfer buffers (demo23 choreography)
		if(cfg.svm_c>0.0) c->set_parameter_C(cfg.svm_c);
		if(cfg.svm_gamma>0.0) c->set_parameter_gamma(cfg.svm_gamma);
		if(cfg.svm_coef0!=0.0) c->set_parameter_coef0(cfg.svm_coef0);
		if(pipe) pipe->svm=c;
		return c;
#else
		throw FST::fst_error("this fst4 build has no LibSVM support (fetch libsvm/svm.cpp and svm.h per libsvm/readme.txt, then rebuild; CMake enables FST_SVM automatically when they are present)");
#endif
	}
	throw FST::fst_error("unknown classifier '"+name+"' (available: knn, bayes, logreg, multinom, lda, rf, linsvm, rbfsvm, extern, svm)");
}

//! constructs and installs one filter criterion incl. the (uniform) per-split
//! re-initialization callbacks — the five make_criterion filter blocks share it
//! (the lambdas capture the CONCRETE type so that the per-split initialize() call
//! binds statically, exactly as a hand-written per-criterion block would)
template<class CRIT_T>
static std::shared_ptr<CRITERION> setup_filter_criterion(const std::shared_ptr<DATAACCESSOR> &da,
	std::function<bool(RETURNTYPE&,const std::shared_ptr<SUBSET>&)> &split_eval, std::function<void()> &crit_reinit)
{
	std::shared_ptr<CRIT_T> c(new CRIT_T);
	c->initialize(da);
	split_eval=[c,da](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {c->initialize(da); return c->evaluate(val,sub);};
	crit_reinit=[c,da]() {c->initialize(da);};
	return c;
}

//! constructs the criterion (initialize()d on the accessor's current split) and fills
//! the pipeline's split_eval / crit_reinit callbacks and classifier handles
static std::shared_ptr<CRITERION> make_criterion(const Config &cfg, Pipeline &p)
{
	const std::shared_ptr<DATAACCESSOR> &da=p.da;
	std::function<bool(RETURNTYPE&,const std::shared_ptr<SUBSET>&)> &split_eval=p.split_eval;
	std::function<void()> &crit_reinit=p.crit_reinit;
	std::shared_ptr<CLASSIFIER> &clf_out=p.clf;
	if(cfg.criterion=="bhattacharyya") return setup_filter_criterion<CRIT_BHAT>(da,split_eval,crit_reinit);
	if(cfg.criterion=="divergence") return setup_filter_criterion<CRIT_DIV>(da,split_eval,crit_reinit);
	if(cfg.criterion=="gmahalanobis") return setup_filter_criterion<CRIT_GMAHA>(da,split_eval,crit_reinit);
	if(cfg.criterion=="multinom-bhattacharyya") return setup_filter_criterion<CRIT_MBHAT>(da,split_eval,crit_reinit);
	if(cfg.criterion=="multinom-mi") return setup_filter_criterion<CRIT_MMI>(da,split_eval,crit_reinit);
	if(cfg.criterion=="wrapper-regressor") {
		// the regression pipeline (demo11reg): kNN-based weighted-neighbours regression,
		// dependent variable = the class label (regression simulating classification)
		std::shared_ptr<RWN> rwn(new RWN);
		rwn->set_dependent_index((IDXTYPE)-1);
		if(cfg.regr_multiplier>0.0) rwn->set_kernel_multiplier(cfg.regr_multiplier);
		p.rwn=rwn;
		p.regr=rwn; p.regr_label="regressor";
		std::shared_ptr<CRIT_WRAPPER_REG> c(new CRIT_WRAPPER_REG);
		c->initialize(rwn,da);
		if(cfg.has_wrapper_metric()) {
			const FST::Metric_Spec<RETURNTYPE> spec=make_metric_spec(cfg,da->getNoOfClasses());
			c->set_metric(spec);
			split_eval=[rwn,da,spec](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {
				if(!rwn->train(da,sub)) return false;
				return rwn->test_metric(val,da,spec);
			};
		} else {
			split_eval=[rwn,da](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {
				if(!rwn->train(da,sub)) return false;
				return rwn->test(val,da);
			};
		}
		return c;
	}
	if(cfg.criterion=="wrapper-ridge") {
		// closed-form L2-regularized linear regression; same pipeline shape as
		// wrapper-regressor (incl. the class-label dependent-variable convention), criterion
		// value = the selected regression metric (default r2, normalized after CLI parse)
		std::shared_ptr<RIDGE> ridge(new RIDGE);
		ridge->set_lambda(cfg.ridge_lambda);
		ridge->set_dependent_index((DIMTYPE)-1);
		p.regr=ridge; p.regr_label="ridge";
		std::shared_ptr<CRIT_WRAPPER_REG> c(new CRIT_WRAPPER_REG);
		ridge->initialize(da);
		c->initialize(ridge,da);
		const FST::Metric_Spec<RETURNTYPE> spec=make_metric_spec(cfg,da->getNoOfClasses());
		c->set_metric(spec);
		split_eval=[ridge,da,spec](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {
			if(!ridge->train(da,sub)) return false;
			return ridge->test_metric(val,da,spec);
		};
		return c;
	}
	if(cfg.criterion=="extern-value") {
		std::shared_ptr<CRIT_EXTVAL> c(new CRIT_EXTVAL);
		c->set_command(cfg.extern_cmd);
		c->set_monotonic(cfg.extern_monotonic);
		c->set_no_data(cfg.extern_no_data);
		c->initialize(da); // records data shape; the child spawns lazily on first evaluate()
		// the DATA-once protocol follows the filter refresh contract: initialize() marks the
		// tool's stored DATA stale, so per-trial re-init (crit_reinit) and per-call re-init
		// (split_eval) keep the tool in sync with the current split exactly like filter
		// model re-estimation does
		crit_reinit=[c,da]() {c->initialize(da);};
		split_eval=[c,da](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {c->initialize(da); return c->evaluate(val,sub);};
		return c;
	}
	if(cfg.is_wrapper_criterion()) {
		std::shared_ptr<CLASSIFIER> cls=make_classifier(cfg.criterion.substr(8),cfg,da,&p);
		clf_out=cls;
		std::shared_ptr<CRIT_WRAPPER> c(new CRIT_WRAPPER);
		c->initialize(cls,da);
		if(cfg.has_wrapper_metric()) {
			const FST::Metric_Spec<RETURNTYPE> spec=make_metric_spec(cfg,da->getNoOfClasses());
			c->set_metric(spec);
			split_eval=[cls,da,spec](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {
				if(!cls->train(da,sub)) return false;
				return cls->test_metric(val,da,spec);
			};
		} else {
			split_eval=[cls,da](RETURNTYPE &val, const std::shared_ptr<SUBSET> &sub)->bool {
				if(!cls->train(da,sub)) return false;
				return cls->test(val,da);
			};
		}
		return c;
	}
	throw FST::fst_error("unknown criterion '"+cfg.criterion+"'");
}

static PENGINE make_engine(const Config &cfg, Pipeline &p); // defined below (the candidate-evaluation engine)

//! constructs the sequential-step candidate evaluator selected by `--evaluator` (straight/hybrid/
//! ensemble/sampled), or -- when `--evaluator` is left at its default and the search is not one of
//! those step-evaluator families -- the plain sequential/thread-pooled evaluator picked by
//! `--threads`. sampled/hybrid/ensemble additionally stash engine/statistics handles on \p p so
//! later phases (the `--warmup-probes` loop) can reuse them
//! \param cfg the resolved CLI configuration (`--evaluator`, `--threads`, and the sampled/hybrid/ensemble sub-flags)
//! \param p the run's Pipeline, mutated in place to stash engine/statistics handles for later phases
//! \param eval_depth the splitting depth the constructed evaluator's criterion should run at
//!        (hybrid's prefilter is built at depth 0 regardless, then the main criterion resumes at this depth)
static std::shared_ptr<EVALUATOR> make_evaluator(const Config &cfg, Pipeline &p, const int eval_depth)
{
	if(cfg.evaluator=="sampled") {
		// budgeted stochastic step:
		// forward steps evaluate only `--step-cap` candidates proposed from online per-feature
		// statistics; the statistics engine is kept on the Pipeline so the `--warmup-probes`
		// loop in run_pipeline can seed it before the search starts
		std::shared_ptr<WEIGHTER_SAMPLED> stats(new WEIGHTER_SAMPLED((IDXTYPE)cfg.step_decay,
			cfg.step_stats=="rawmean"?WEIGHTER_SAMPLED::STATS_RAWMEAN:WEIGHTER_SAMPLED::STATS_ZCONTRAST));
		p.sampled_stats=stats;
		const EVAL_SAMPLED::SamplerMode sm = cfg.step_sampler=="uniform" ? EVAL_SAMPLED::SAMPLER_UNIFORM
		                                   : (cfg.step_sampler=="topk"   ? EVAL_SAMPLED::SAMPLER_TOPK
		                                                                 : EVAL_SAMPLED::SAMPLER_SOFTMAX);
		// `--threads`>1 injects the thread-pooled candidate-evaluation engine: proposals stay on the master
		// thread's RNG stream, only criterion evaluations fan out - results are bit-identical
		// to `--threads` 1 at matched `--seed` (see seq_step_sampled.hpp); the engine is kept on
		// the Pipeline so the threaded warm-up loop reuses it (criterion clones persist)
		PENGINE engine=make_engine(cfg,p);
		p.sampled_engine=engine;
		EVAL_SAMPLED *ev=new EVAL_SAMPLED(stats,(DIMTYPE)cfg.step_cap,cfg.step_explore,cfg.step_tau,sm,engine);
		ev->set_cap_frac(cfg.step_cap_frac); // >0 = constant-percentage schedule (ablation)
		ev->set_cap_backward((DIMTYPE)cfg.step_cap_backward); // >0 = budgeted backward steps
		return std::shared_ptr<EVALUATOR>(ev);
	}
	if(cfg.evaluator=="hybrid") {
		// demo53: a fast filter pre-orders the candidates and only the top `--hybrid-keep-perc`
		// percent reach the main criterion; the prefilter is initialized on the depth-0 data
		// (whole outer train part) while the main criterion evaluates on the inner splits
		p.da_mem->setSplittingDepth(0);
		std::shared_ptr<CRITERION> pref=make_filter(cfg.hybrid_prefilter,p.da);
		p.da_mem->setSplittingDepth(eval_depth);
		// `--threads`>1 injects the thread-pooled engine for the main-criterion phase; the
		// prefilter pre-sort stays on the master - results identical to `--threads` 1
		return std::shared_ptr<EVALUATOR>(new EVAL_HYBRID(pref,cfg.hybrid_keep_perc,cfg.hybrid_keep_max,make_engine(cfg,p)));
	}
	if(cfg.evaluator=="ensemble") {
		// demo50: kNN wrappers with different k vote on each candidate's rank
		std::vector<double> ks=split_csv_doubles(cfg.ensemble_knn_ks,"--ensemble-knn-ks");
		EVAL_ENSEMBLE::PEnsembleCriteria ens(new EVAL_ENSEMBLE::PAbstractCriteria);
		for(std::vector<double>::const_iterator k=ks.begin();k!=ks.end();++k) {
			if(*k<1.0 || *k!=(double)(unsigned int)(*k)) throw FST::fst_error("--ensemble-knn-ks entries must be positive integers");
			std::shared_ptr<CRIT_WRAPPER> w(new CRIT_WRAPPER);
			w->initialize(make_knn(cfg,(unsigned int)(*k)),p.da);
			ens->push_back(w);
		}
		// `--threads`>1 injects the thread-pooled engine (one batch per ensemble member per
		// step); vote combination stays on the master - results identical to `--threads` 1
		return std::shared_ptr<EVALUATOR>(new EVAL_ENSEMBLE(ens,make_engine(cfg,p)));
	}
	// the straight step: the merged class with the engine `--threads` selects (sequential
	// in-line or thread-pooled) injected at construction
	return std::shared_ptr<EVALUATOR>(new EVAL_SEQ(make_engine(cfg,p)));
}

//! the candidate-evaluation engine for the non-sequential searches (exhaustive,
//! Monte Carlo, B&B) and for every step evaluator (straight/hybrid/ensemble/sampled);
//! each engine is registered on \p p so that Pipeline::invalidate_engine_clones() can
//! drop its criterion clones when the criterion's evaluation context changes
static PENGINE make_engine(const Config &cfg, Pipeline &p)
{
	PENGINE e;
	if(cfg.threads<=1) e.reset(new FST::Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
	else e.reset(new FST::Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(cfg.threads));
	p.engines.push_back(e);
	return e;
}

//! constructs the result tracker; capacity 0 means the type default
//! (dupless: top_k, others: unlimited)
static void make_tracker(const Config &cfg, Pipeline &p)
{
	const IDXTYPE cap = cfg.tracker_capacity ? (IDXTYPE)cfg.tracker_capacity
	                    : (cfg.tracker=="dupless" ? (IDXTYPE)cfg.top_k : 0);
	if(cfg.tracker=="dupless") p.tracker.reset(new TRACKER(cap));
	else if(cfg.tracker=="regularizer") {p.tracker_reg.reset(new TRACKER_REG(cap)); p.tracker=p.tracker_reg;}
	else if(cfg.tracker=="feature-stats") {p.tracker_stats.reset(new TRACKER_STATS(cap)); p.tracker=p.tracker_stats;}
	else if(cfg.tracker=="stabileval") p.tracker_stabil.reset(new TRACKER_STABIL(cap)); // never attached: fed manually per trial (demo54)
	else throw FST::fst_error("unknown tracker '"+cfg.tracker+"'");
	if(p.tracker && cfg.tracker_margin>0.0) p.tracker->set_inclusion_margin(cfg.tracker_margin);
}

/*! walks (score, is_positive) pairs from the highest score down and emits one ROC and one
    PR point per distinct threshold (deterministic: stable sort keeps enumeration order in
    tie groups; the (0,0) ROC start point is implied). Sorts \p sc in place, descending by
    score. Returns false when either class is absent on the holdout — the curves are
    undefined then (the AUC convention). */
static bool extract_curves(std::vector<std::pair<RETURNTYPE,DIMTYPE> > &sc, std::vector<RunResult::CurvePoint> &roc, std::vector<RunResult::CurvePoint> &pr)
{
	const size_t n=sc.size();
	size_t n_pos=0;
	for(size_t i=0;i<n;i++) if(sc[i].second==1) n_pos++;
	const size_t n_neg=n-n_pos;
	if(n_pos==0 || n_neg==0) return false;
	std::stable_sort(sc.begin(),sc.end(),
		[](const std::pair<RETURNTYPE,DIMTYPE> &a, const std::pair<RETURNTYPE,DIMTYPE> &b){return a.first>b.first;});
	size_t tp=0, fp=0, i=0;
	while(i<n) { // tie groups share one point (threshold = the group's score)
		size_t j=i;
		while(j<n && !(sc[j].first<sc[i].first)) {if(sc[j].second==1) tp++; else fp++; j++;}
		RunResult::CurvePoint rp, pp;
		rp.threshold=pp.threshold=sc[i].first;
		rp.x=(RETURNTYPE)fp/(RETURNTYPE)n_neg; rp.y=(RETURNTYPE)tp/(RETURNTYPE)n_pos;
		pp.x=(RETURNTYPE)tp/(RETURNTYPE)n_pos; pp.y=(RETURNTYPE)tp/(RETURNTYPE)(tp+fp);
		roc.push_back(rp); pr.push_back(pp);
		i=j;
	}
	return true;
}

//! `--report` detailed: ROC/PR curves of a trained classifier on the depth-0 holdout —
//! binary from test_scores (positive class 1), K>2 one one-vs-rest curve per class
//! from the score matrix; classifiers without classify_scores (extern) yield nothing
static void extract_classifier_curves(const std::shared_ptr<CLASSIFIER> &vc, const std::shared_ptr<DATAACCESSOR> &da, std::vector<RunResult::ClassCurve> &curves)
{
	const DIMTYPE classes=da->getNoOfClasses();
	if(classes==2) {
		std::vector<std::pair<RETURNTYPE,DIMTYPE> > sc;
		if(!vc->test_scores(sc,da)) return;
		RunResult::ClassCurve c; c.cls=1;
		if(extract_curves(sc,c.roc,c.pr)) curves.push_back(c);
		return;
	}
	std::vector<RETURNTYPE> scores;
	std::vector<DIMTYPE> labels;
	DIMTYPE k;
	if(!vc->test_scores_matrix(scores,labels,k,da)) return;
	for(DIMTYPE cls=0;cls<k;cls++) {
		std::vector<std::pair<RETURNTYPE,DIMTYPE> > sc;
		sc.reserve(labels.size());
		for(size_t i=0;i<labels.size();i++) sc.push_back(std::make_pair(scores[i*(size_t)k+(size_t)cls],(DIMTYPE)((labels[i]==cls)?1:0)));
		RunResult::ClassCurve c; c.cls=cls;
		if(extract_curves(sc,c.roc,c.pr)) curves.push_back(c);
	}
}

/*! `--dump-scores`: the final subset's per-sample predictions on the depth-0 holdout — true
    label, predicted label, and one column per class score where the classifier exposes them.
    The walk is the class-major, block-order enumeration test_scores_matrix performs, so row i
    of the dump is point i of the ROC/PR input. The prediction comes from classify(), the
    classifier's OWN decision rule, not from an argmax over the score row: the two coincide for
    most classifiers but the LibSVM wrapper aggregates one-vs-one votes by its own convention.
    Classifiers without score support (the wrapper-extern label protocol) get the label columns
    only. Costs one extra pass over the holdout, no extra training. */
static void write_scores_dump(const std::string &path, const std::shared_ptr<CLASSIFIER> &vc,
                              const std::shared_ptr<DATAACCESSOR> &da, const std::string &label,
                              const std::vector<DIMTYPE> &features)
{
	std::vector<RETURNTYPE> scores;
	std::vector<DIMTYPE> score_labels;
	DIMTYPE score_classes=0;
	const bool has_scores=vc->test_scores_matrix(scores,score_labels,score_classes,da);
	std::ofstream out(path.c_str());
	if(!out) throw FST::fst_error("--dump-scores: cannot open '"+path+"' for writing");
	out << "# fst4 per-sample predictions of the selected subset on the depth-0 holdout" << std::endl
	    << "# classifier: " << label << std::endl
	    << "# subset:";
	for(std::vector<DIMTYPE>::size_type i=0;i<features.size();++i) out << " " << features[i];
	out << std::endl;
	if(!has_scores) out << "# this classifier exposes no per-class scores; label columns only" << std::endl;
	out << "sample,true_class,predicted_class";
	if(has_scores) for(DIMTYPE c=0;c<score_classes;c++) out << ",score_" << c;
	out << std::endl;
	typename DATAACCESSOR::PPattern pat;
	DIMTYPE predicted;
	const DIMTYPE stride=da->getNoOfFeatures();
	const DIMTYPE classes=da->getNoOfClasses();
	IDXTYPE cnt;
	std::size_t row=0;
	for(DIMTYPE c_test=0;c_test<classes;c_test++) {
		da->setClass(c_test);
		for(const auto &blk : da->blocks(FST::TEST)) { pat=blk.data; cnt=blk.size;
			for(IDXTYPE i=0;i<cnt;i++) {
				if(!vc->classify(predicted,&pat[(std::size_t)i*stride]))
					throw FST::fst_error("--dump-scores: classification of holdout sample "+std::to_string((long long)(row+1))+" failed");
				// the same "wrong class" remap test_confusion applies (classifier.hpp): kNN can
				// answer out of range when a pattern has no neighbours. Following that convention
				// keeps the dump and the confusion matrix of the SAME run consistent — counting
				// the dump's predicted!=true rows must reproduce the matrix's off-diagonal sum
				if(predicted>=classes) predicted=(c_test+1)%classes;
				if(has_scores && (row>=score_labels.size() || score_labels[row]!=c_test))
					throw FST::fst_error("--dump-scores: the score pass and the prediction pass disagree on the holdout enumeration"); // cannot happen: both walk classes x blocks in the same order
				out << row << "," << c_test << "," << predicted;
				if(has_scores) for(DIMTYPE k=0;k<score_classes;k++)
					out << "," << fmt_double((double)scores[row*(std::size_t)score_classes+(std::size_t)k]);
				out << std::endl;
				++row;
			}
			da->setClass(c_test); // necessary where da is the classifier's own training accessor (the kNN pattern, classifier.hpp)
		}
	}
	out.close();
	if(!out) throw FST::fst_error("--dump-scores: writing '"+path+"' failed");
}

//! detail!=nullptr (`--report` detailed final validation): classification entries gain the
//! confusion matrix, and the FIRST classification entry contributes the ROC/PR curves —
//! the `--validate` step: trains each named classifier on the depth-0 train part and
//! tests it on the depth-0 test part of the given subset (the demo final-validation protocol);
//! both reuse the just-trained classifier, no extra training. final=true marks the
//! final-subset validation, the only one `--dump-scores` writes for (the sweep and
//! curve_over_d re-validations call this too)
static std::vector<RunResult::ValidationEntry> run_validation(const Config &cfg, Pipeline &p, const std::shared_ptr<SUBSET> &sub, RunResult *detail=nullptr, const bool is_final=false)
{
	std::vector<RunResult::ValidationEntry> out;
	bool dumped=false;
	p.da_mem->setSplittingDepth(0);
	std::vector<std::string> names=split_csv(cfg.validate);
	for(std::vector<std::string>::const_iterator it=names.begin();it!=names.end();++it) {
		if(*it=="same" && p.regr) { // regression pipelines: validate with the wrapper's regressor
			RunResult::ValidationEntry e;
			e.classifier=p.regr_label;
			if(!p.regr->train(p.da,sub)) throw FST::fst_error("validation training of the regressor failed");
			if(!p.regr->test(e.accuracy,p.da)) throw FST::fst_error("validation testing of the regressor failed");
			if(cfg.has_wrapper_metric()) { // report the selected metric alongside the plain test() value
				const FST::Metric_Spec<RETURNTYPE> spec=make_metric_spec(cfg,p.da->getNoOfClasses());
				e.metric=FST::wrapper_metric_name(spec.metric); // `--class-weights` is rejected for regression, so no "weighted-" prefix can arise here
				if(!p.regr->test_metric(e.metric_value,p.da,spec)) throw FST::fst_error("validation metric '"+e.metric+"' is undefined on the holdout part");
			}
			out.push_back(e);
			continue;
		}
		std::shared_ptr<CLASSIFIER> vc;
		std::string label=*it;
		if(*it=="same") {
			if(!p.clf) throw FST::fst_error("--validate same requires a wrapper criterion (filter criteria have no classifier); name a classifier instead");
			vc=p.clf; label=cfg.criterion.substr(8);
		}
		else vc=make_classifier(*it,cfg,p.da);
		RunResult::ValidationEntry e;
		e.classifier=label;
		if(!vc->train(p.da,sub)) throw FST::fst_error("validation training of classifier '"+label+"' failed");
		if(!vc->test(e.accuracy,p.da)) throw FST::fst_error("validation testing of classifier '"+label+"' failed");
		if(cfg.has_wrapper_metric()) { // report the selected metric alongside plain accuracy (deployment quality vs. selection pressure)
			const FST::Metric_Spec<RETURNTYPE> spec=make_metric_spec(cfg,p.da->getNoOfClasses());
			e.metric=std::string(spec.class_weights.empty()?"":"weighted-")+FST::wrapper_metric_name(spec.metric);
			if(!vc->test_metric(e.metric_value,p.da,spec)) throw FST::fst_error("validation metric '"+e.metric+"' is undefined on the holdout part");
		}
		if(detail) { // `--report` detailed: confusion per entry; curves from the first classification entry
			DIMTYPE cc;
			std::vector<RETURNTYPE> conf;
			if(vc->test_confusion(conf,cc,p.da)) {e.confusion=conf; e.confusion_classes=cc;}
			if(detail->curves_classifier.empty()) {
				extract_classifier_curves(vc,p.da,detail->curves);
				if(!detail->curves.empty()) detail->curves_classifier=label;
			}
		}
		// `--dump-scores`: the first classification entry of the final validation, reusing
		// the classifier just trained here (gated independently of `--report` detailed)
		if(is_final && !dumped && !cfg.dump_scores.empty()) {
			write_scores_dump(cfg.dump_scores,vc,p.da,label,subset_features(sub));
			dumped=true;
		}
		out.push_back(e);
	}
	return out;
}

//! constructs the `--search` algorithm by name, applying its algorithm-specific parameters
//! (delta/gamma/generalization-level/Monte Carlo stopping/...); \p eval is used only by the
//! sequential-family searches (bif, exhaustive, Monte Carlo and the four B&B variants build
//! their own engine internally via make_engine(), registered on \p p)
//! \return the constructed search; throws FST::fst_error on an unknown `--search` name
static std::shared_ptr<SEARCH> make_search(const Config &cfg, Pipeline &p, const std::shared_ptr<EVALUATOR> &eval)
{
	if(cfg.search=="bif") {
		return std::shared_ptr<SEARCH>(new FST::Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
	}
	if(cfg.search=="sfs" || cfg.search=="sbs") {
		FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> *s=
			new FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(eval);
		s->set_search_direction(cfg.search=="sfs"?FST::FORWARD:FST::BACKWARD);
		s->set_generalization_level(cfg.generalization);
		if(!cfg.search2.empty()) s->enable_initial_subset(); // --search2: start from the BIF seed instead of the empty (forward) / full (backward) set
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="sffs" || cfg.search=="sbfs") {
		FST::Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> *s=
			new FST::Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(eval);
		s->set_search_direction(cfg.search=="sffs"?FST::FORWARD:FST::BACKWARD);
		s->set_delta(cfg.sffs_delta); // 0 (default) = classic unrestricted floating over the full frontier
		s->set_generalization_level(cfg.generalization);
		if(!cfg.search2.empty()) s->enable_initial_subset(); // --search2: start from the BIF seed instead of the empty (forward) / full (backward) set
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="sfrs" || cfg.search=="sbrs") {
		FST::Search_SFRS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> *s=
			new FST::Search_SFRS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(eval);
		s->set_search_direction(cfg.search=="sfrs"?FST::FORWARD:FST::BACKWARD);
		s->set_generalization_level(cfg.generalization);
		if(!cfg.search2.empty()) s->enable_initial_subset(); // --search2: start from the BIF seed instead of the empty (forward) / full (backward) set
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="os") {
		FST::Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> *s=
			new FST::Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(eval);
		s->set_delta(cfg.os_delta);
		s->set_generalization_level(cfg.generalization);
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="dos") {
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> *s=
			new FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(eval);
		s->set_delta(cfg.dos_delta);
		s->set_generalization_level(cfg.generalization);
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="exhaustive") {
		return std::shared_ptr<SEARCH>(new FST::Search_Exhaustive<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(make_engine(cfg,p)));
	}
	if(cfg.search=="montecarlo") {
		FST::Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> *s=
			new FST::Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(cfg.mc_trials,cfg.mc_seconds,1,make_engine(cfg,p));
		if(cfg.mc_card_prob>0.0) s->set_cardinality_randomization((float)cfg.mc_card_prob);
		else if(cfg.mc_card_min>0) s->set_cardinality_randomization(cfg.mc_card_min,cfg.mc_card_max);
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="bb-basic") {
		return std::shared_ptr<SEARCH>(new FST::Search_Branch_And_Bound_Basic<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(make_engine(cfg,p)));
	}
	if(cfg.search=="bb-improved") {
		return std::shared_ptr<SEARCH>(new FST::Search_Branch_And_Bound_Improved<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(make_engine(cfg,p)));
	}
	if(cfg.search=="bb-fast") {
		FST::Search_Branch_And_Bound_Fast<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,BBPREDICTOR> *s=
			new FST::Search_Branch_And_Bound_Fast<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,BBPREDICTOR>(make_engine(cfg,p));
		s->set_gamma(cfg.bb_gamma);
		s->set_delta(cfg.bb_delta);
		return std::shared_ptr<SEARCH>(s);
	}
	if(cfg.search=="bb-bbpp") {
		return std::shared_ptr<SEARCH>(new FST::Search_Branch_And_Bound_Partial_Prediction<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,BBPREDICTOR>(make_engine(cfg,p)));
	}
	throw FST::fst_error("unknown search '"+cfg.search+"'");
}

//! scans the current split's TRAIN data for features constant within some class -
//! the cause of singular covariance matrices in normal-model criteria (4.4)
static std::string list_constant_features(const std::shared_ptr<DATAACCESSOR> &da)
{
	std::string out;
	try {
		const DIMTYPE n=da->getNoOfFeatures();
		std::vector<bool> constant(n,true);
		for(DIMTYPE c=0;c<da->getNoOfClasses();c++) {
			da->setClass(c);
			std::vector<DATATYPE> first(n);
			std::vector<bool> seen(n,false);
			typename DATAACCESSOR::PPattern pat;
			IDXTYPE cnt;
			for(const auto &blk : da->blocks(FST::TRAIN)) { pat=blk.data; cnt=blk.size;
				for(IDXTYPE i=0;i<cnt;i++) for(DIMTYPE f=0;f<n;f++) {
					if(!constant[f]) continue;
					const DATATYPE v=pat[i*n+f];
					if(!seen[f]) {first[f]=v; seen[f]=true;}
					else if(v!=first[f]) constant[f]=false;
				}
			}
		}
		for(DIMTYPE f=0;f<n;f++) if(constant[f]) {if(!out.empty()) out+=" "; out+=std::to_string((long long)f);}
	}
	catch(...) {return std::string();} // diagnostics only - never mask the original error
	return out;
}

//! maps `--verbosity` ("silent"|"normal"|"detailed") to the library's FST::OutputDetail enum;
//! any other string (unreachable past CLI11's IsMember check) falls back to FST::NORMAL
static FST::OutputDetail parse_verbosity(const std::string &v)
{
	if(v=="silent") return FST::SILENT;
	if(v=="detailed") return FST::DETAILED;
	return FST::NORMAL;
}

// ---------------------------------------------------------------------------
// output emitters (results are sourced from the tracker and the final subset)

//! writes the "invocation" JSON object: an echo of the resolved configuration, restricted to
//! options that actually apply to this run (criterion-specific/search-specific/non-default
//! flags only) so the echo stays a faithful, minimal record of what produced the result --
//! part of emit_json()'s output, so \p os is always the real-stdout stream, never the
//! redirected std::cout
static void emit_invocation_json(std::ostream &os, const Config &cfg, const std::string &format)
{
	os << "  \"invocation\": {";
	if(cfg.data_free()) os << "\"features\": " << cfg.features_n; // the data-free entry: no `--data` was given, no format exists
	else {
		os << "\"data\": \"" << json_escape(cfg.data) << "\"";
		os << ", \"format\": \"" << format << "\"";
		// `--csv-missing-to` is echoed while the `--csv`-* DIALECT flags deliberately are not: the dialect
		// options are result-neutral by construction, whereas this one changes WHICH RECORDS load -
		// a run keeping the missing rows is not reproducible from an echo that omits it
		if(cfg.has_csv_missing_to()) os << ", \"csv_missing_to\": " << fmt_double(cfg.csv_missing_to);
	}
	os << ", \"splitter\": \"" << cfg.splitter << "\"";
	if(cfg.splitter=="cv") os << ", \"cv_folds\": " << cfg.cv_folds;
	if(cfg.splitter=="holdout") os << ", \"holdout_train\": " << cfg.holdout_train;
	if(cfg.splitter=="randrand" || cfg.splitter=="randfix")
		os << ", \"rr_splits\": " << cfg.rr_splits << ", \"rr_train\": " << cfg.rr_train << ", \"rr_test\": " << cfg.rr_test;
	if(cfg.nested_splits()) {
		os << ", \"splitter2\": \"" << cfg.splitter2 << "\"";
		if(cfg.splitter2=="cv") os << ", \"splitter2_cv_folds\": " << cfg.cv2_folds;
		if(cfg.splitter2=="holdout") os << ", \"splitter2_holdout_train\": " << cfg.holdout2_train;
		if(cfg.splitter2=="randrand" || cfg.splitter2=="randfix")
			os << ", \"splitter2_rr_splits\": " << cfg.rr2_splits << ", \"splitter2_rr_train\": " << cfg.rr2_train << ", \"splitter2_rr_test\": " << cfg.rr2_test;
	}
	if(!cfg.validate.empty()) os << ", \"validate\": \"" << json_escape(cfg.validate) << "\"";
	if(!cfg.wrapper_metric.empty()) os << ", \"wrapper_metric\": \"" << json_escape(cfg.wrapper_metric) << "\"";
	if(!cfg.class_weights.empty()) os << ", \"class_weights\": \"" << json_escape(cfg.class_weights) << "\"";
	if(cfg.f1_positive!=1) os << ", \"f1_positive\": " << cfg.f1_positive; // non-default only: the echo of runs that do not use it stays unchanged
	os << ", \"scaler\": \"" << cfg.scaler << "\"";
	if(cfg.scaler_fit!="all") os << ", \"scaler_fit\": \"" << cfg.scaler_fit << "\""; // non-default only: the echo of runs that do not use it stays unchanged
	if(cfg.has_missing_value()) os << ", \"missing_value\": " << fmt_double(cfg.missing_value);
	os << ", \"criterion\": \"" << cfg.criterion << "\"";
	if(cfg.criterion=="wrapper-knn") {
		os << ", \"knn_k\": " << cfg.knn_k;
		os << ", \"knn_distance\": \"" << cfg.knn_distance << "\"";
	}
	if(cfg.criterion=="wrapper-rf") {
		os << ", \"rf_trees\": " << cfg.rf_trees;
		if(cfg.rf_mtry>0) os << ", \"rf_mtry\": " << cfg.rf_mtry;
		if(cfg.rf_max_depth>0) os << ", \"rf_max_depth\": " << cfg.rf_max_depth;
		if(cfg.rf_min_leaf>1) os << ", \"rf_min_leaf\": " << cfg.rf_min_leaf;
	}
	if(cfg.criterion=="wrapper-ridge") os << ", \"ridge_lambda\": " << fmt_double(cfg.ridge_lambda);
	if(cfg.criterion=="wrapper-extern") os << ", \"extern_cmd\": \"" << json_escape(cfg.extern_cmd) << "\"";
	if(cfg.criterion=="extern-value") {
		os << ", \"extern_cmd\": \"" << json_escape(cfg.extern_cmd) << "\"";
		if(cfg.extern_monotonic) os << ", \"extern_monotonic\": true";
		if(cfg.extern_no_data) os << ", \"extern_no_data\": true";
	}
	if(cfg.criterion=="wrapper-linsvm") {
		os << ", \"linsvm_c\": " << fmt_double(cfg.linsvm_c) << ", \"linsvm_loss\": \"" << cfg.linsvm_loss << "\"";
		if(cfg.linsvm_maxiter!=1000) os << ", \"linsvm_maxiter\": " << cfg.linsvm_maxiter;
		if(cfg.linsvm_tol!=0.1) os << ", \"linsvm_tol\": " << fmt_double(cfg.linsvm_tol);
	}
	if(cfg.criterion=="wrapper-rbfsvm") {
		os << ", \"rbfsvm_c\": " << fmt_double(cfg.rbfsvm_c) << ", \"rbfsvm_gamma\": " << fmt_double(cfg.rbfsvm_gamma)
		   << ", \"rbfsvm_loss\": \"" << cfg.rbfsvm_loss << "\""; // gamma 0 = auto (1/d)
		if(cfg.rbfsvm_maxiter!=1000) os << ", \"rbfsvm_maxiter\": " << cfg.rbfsvm_maxiter;
		if(cfg.rbfsvm_tol!=0.1) os << ", \"rbfsvm_tol\": " << fmt_double(cfg.rbfsvm_tol);
		if(cfg.rbfsvm_optimize!="none") os << ", \"rbfsvm_optimize\": \"" << cfg.rbfsvm_optimize << "\"";
	}
	if(cfg.criterion=="wrapper-svm") {
		os << ", \"svm_kernel\": \"" << cfg.svm_kernel << "\"";
		if(cfg.svm_c>0.0) os << ", \"svm_c\": " << fmt_double(cfg.svm_c);
		if(cfg.svm_gamma>0.0) os << ", \"svm_gamma\": " << fmt_double(cfg.svm_gamma);
		if(cfg.svm_coef0!=0.0) os << ", \"svm_coef0\": " << fmt_double(cfg.svm_coef0);
		if(cfg.svm_optimize!="none") os << ", \"svm_optimize\": \"" << cfg.svm_optimize << "\"";
	}
	if(!cfg.search2.empty()) {
		os << ", \"search2\": \"" << cfg.search2 << "\"";
		if(cfg.search2_d>0) os << ", \"search2_d\": " << cfg.search2_d;
	}
	os << ", \"search\": \"" << cfg.search << "\"";
	os << ", \"target_d\": " << cfg.target_d;
	if(cfg.search=="os") os << ", \"os_delta\": " << cfg.os_delta;
	if(cfg.search=="dos") os << ", \"dos_delta\": " << cfg.dos_delta;
	if((cfg.search=="sffs"||cfg.search=="sbfs") && cfg.sffs_delta>0) os << ", \"sffs_delta\": " << cfg.sffs_delta;
	if(cfg.generalization>1) os << ", \"generalization_level\": " << cfg.generalization;
	if(cfg.evaluator=="sampled") {
		os << ", \"evaluator\": \"sampled\", \"step_cap\": " << cfg.step_cap
		   << ", \"step_explore\": " << fmt_double(cfg.step_explore)
		   << ", \"step_tau\": " << fmt_double(cfg.step_tau)
		   << ", \"step_decay\": " << cfg.step_decay
		   << ", \"step_sampler\": \"" << cfg.step_sampler << "\", \"step_stats\": \"" << cfg.step_stats << "\"";
		if(cfg.step_cap_frac>0.0) os << ", \"step_cap_frac\": " << fmt_double(cfg.step_cap_frac);
		if(cfg.step_cap_backward>0) os << ", \"step_cap_backward\": " << cfg.step_cap_backward;
		if(cfg.step_frozen) os << ", \"step_frozen\": true";
		if(cfg.warmup_probes>0) os << ", \"warmup_probes\": " << cfg.warmup_probes << ", \"warmup_card\": " << cfg.warmup_card;
	}
	if(cfg.search=="montecarlo") {
		os << ", \"mc_trials\": " << cfg.mc_trials << ", \"mc_seconds\": " << cfg.mc_seconds;
		if(cfg.mc_card_prob>0.0) os << ", \"mc_card_prob\": " << fmt_double(cfg.mc_card_prob);
		else if(cfg.mc_card_min>0) os << ", \"mc_card_min\": " << cfg.mc_card_min << ", \"mc_card_max\": " << cfg.mc_card_max;
	}
	if(cfg.criterion=="wrapper-regressor") {
		if(cfg.regr_multiplier>0.0) os << ", \"regr_multiplier\": " << fmt_double(cfg.regr_multiplier);
		if(!cfg.regr_multipliers.empty()) os << ", \"regr_multipliers\": \"" << json_escape(cfg.regr_multipliers) << "\"";
	}
	if(cfg.search=="bb-fast") os << ", \"bb_gamma\": " << fmt_double(cfg.bb_gamma) << ", \"bb_delta\": " << cfg.bb_delta;
	if(cfg.tracker!="dupless") os << ", \"tracker\": \"" << cfg.tracker << "\"";
	if(!cfg.tracker_attach.empty()) os << ", \"tracker_attach\": \"" << cfg.tracker_attach << "\"";
	if(cfg.tracker_capacity>0) os << ", \"tracker_capacity\": " << cfg.tracker_capacity;
	if(cfg.tracker_margin>0.0) os << ", \"tracker_margin\": " << fmt_double(cfg.tracker_margin);
	if(cfg.repeat_over_splits) os << ", \"repeat_over_splits\": true";
	if(!cfg.regularize_by.empty())
		os << ", \"regularize_by\": \"" << cfg.regularize_by << "\", \"regularize_margins\": \"" << json_escape(cfg.regularize_margins) << "\"";
	if(cfg.daf_rank>=0) {
		os << ", \"daf_rank\": " << cfg.daf_rank;
		if(cfg.daf_count>0) os << ", \"daf_count\": " << cfg.daf_count;
		if(cfg.daf_validate_every>0) os << ", \"daf_validate_every\": " << cfg.daf_validate_every;
	}
	os << ", \"threads\": " << cfg.threads;
	os << ", \"seed\": " << cfg.seed;
	os << ", \"top_k\": " << cfg.top_k;
	if(cfg.report!="standard") os << ", \"report\": \"" << cfg.report << "\"";
	if(!cfg.dump_scores.empty()) os << ", \"dump_scores\": \"" << json_escape(cfg.dump_scores) << "\""; // non-default only: the echo of runs that do not use it stays unchanged
	if(!cfg.curves_svg.empty()) os << ", \"curves_svg\": \"" << json_escape(cfg.curves_svg) << "\"";
	os << ", \"stable_output\": " << (cfg.stable_output?"true":"false");
	os << "}," << std::endl;
}

//! writes a JSON array of \p v's elements via operator<<, e.g. feature-index vectors
//! (RETURNTYPE values should go through emit_double_array() instead, for fmt_double() formatting)
template<typename T>
static void emit_array(std::ostream &os, const std::vector<T> &v)
{
	os << "[";
	for(typename std::vector<T>::size_type i=0;i<v.size();++i) {os << (i?", ":"") << v[i];}
	os << "]";
}

//! writes a JSON array of \p v's elements, each formatted through fmt_double()
static void emit_double_array(std::ostream &os, const std::vector<RETURNTYPE> &v)
{
	os << "[";
	for(std::vector<RETURNTYPE>::size_type i=0;i<v.size();++i) {os << (i?", ":"") << fmt_double(v[i]);}
	os << "]";
}

//! writes one RunResult::ValidationEntry as a JSON object; the confusion matrix is included
//! only when confusion_classes>0 (`--report` detailed final validation)
static void emit_validation_json(std::ostream &os, const RunResult::ValidationEntry &e)
{
	os << "{\"classifier\": \"" << json_escape(e.classifier) << "\", \"accuracy\": " << fmt_double(e.accuracy);
	if(!e.metric.empty()) os << ", \"metric\": \"" << json_escape(e.metric) << "\", \"metric_value\": " << fmt_double(e.metric_value);
	if(e.confusion_classes>0) { // `--report` detailed (row = true class)
		os << ", \"confusion\": [";
		for(DIMTYPE t=0;t<e.confusion_classes;t++) {
			if(t) os << ", ";
			os << "[";
			for(DIMTYPE q=0;q<e.confusion_classes;q++) {if(q) os << ", "; os << fmt_double(e.confusion[(size_t)t*(size_t)e.confusion_classes+(size_t)q]);}
			os << "]";
		}
		os << "]";
	}
	os << "}";
}

/*! \brief Writes the complete `--output` json result: version, invocation echo, data summary,
    and every RunResult block that applies (validation, curve_over_d, curves, tuned parameters,
    trials, stability, top_k, regularization, DAF blocks, multiplier sweep, `--compare-config`,
    and the wall-time/evaluations stats block unless `--stable-output`). Called with the real,
    un-redirected stdout stream -- with `--output` json this is the ONLY thing fst4 writes to
    stdout; all search/library prose goes to stderr (or is swallowed under `--verbosity` silent). */
static void emit_json(std::ostream &os, const Config &cfg, const std::string &format,
                      const Pipeline &p, const RunResult &r, const CompareResult *cmp=nullptr)
{
	os << "{" << std::endl;
	os << "  \"version\": \"" << FST4_VERSION << "\"," << std::endl;
	emit_invocation_json(os,cfg,format);
	os << "  \"data\": {\"file\": \"" << json_escape(cfg.data) << "\", \"format\": \"" << format << "\""
	   << ", \"classes\": " << p.da->getNoOfClasses()
	   << ", \"features\": " << p.da->getNoOfFeatures()
	   << ", \"samples\": " << p.da->getClassSizeSum()
	   << ", \"class_sizes\": ";
	{
		std::vector<IDXTYPE> cs;
		for(unsigned int c=0;c<p.da->getNoOfClasses();c++) cs.push_back(p.da->getClassSize(c));
		emit_array(os,cs);
	}
	os << "}," << std::endl;
	os << "  \"result\": {\"features\": ";
	emit_array(os,r.features);
	os << ", \"size\": " << r.features.size() << ", \"value\": " << fmt_double(r.value) << "}";
	if(r.per_split.size()>1) {
		os << "," << std::endl << "  \"per_split\": ";
		emit_double_array(os,r.per_split);
	}
	if(!r.validation.empty()) {
		os << "," << std::endl << "  \"validation\": [";
		for(std::vector<RunResult::ValidationEntry>::size_type i=0;i<r.validation.size();++i) {
			if(i) os << ", ";
			emit_validation_json(os,r.validation[i]);
		}
		os << "]";
	}
	if(!r.curve_over_d.empty()) { // `--report` detailed: criterion axis (value) vs validated axis (validation), per subset size
		os << "," << std::endl << "  \"curve_over_d\": [";
		for(std::vector<RunResult::CurveEntry>::size_type i=0;i<r.curve_over_d.size();++i) {
			const RunResult::CurveEntry &e=r.curve_over_d[i];
			if(i) os << ", ";
			os << "{\"d\": " << e.d << ", \"value\": " << fmt_double(e.value) << ", \"features\": ";
			emit_array(os,e.features);
			if(!e.validation.empty()) {
				os << ", \"validation\": [";
				for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {if(v) os << ", "; emit_validation_json(os,e.validation[v]);}
				os << "]";
			}
			os << "}";
		}
		os << "]";
	}
	if(!r.curves.empty()) { // `--report` detailed: roc rows = [threshold, fpr, tpr]; pr rows = [threshold, recall, precision]
		os << "," << std::endl << "  \"curves\": {\"classifier\": \"" << json_escape(r.curves_classifier) << "\", \"per_class\": [";
		for(std::vector<RunResult::ClassCurve>::size_type c=0;c<r.curves.size();++c) {
			const RunResult::ClassCurve &cc=r.curves[c];
			if(c) os << ", ";
			os << "{\"class\": " << cc.cls << ", \"roc\": [";
			for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.roc.size();++k) {if(k) os << ", "; os << "[" << fmt_double(cc.roc[k].threshold) << ", " << fmt_double(cc.roc[k].x) << ", " << fmt_double(cc.roc[k].y) << "]";}
			os << "], \"pr\": [";
			for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.pr.size();++k) {if(k) os << ", "; os << "[" << fmt_double(cc.pr[k].threshold) << ", " << fmt_double(cc.pr[k].x) << ", " << fmt_double(cc.pr[k].y) << "]";}
			os << "]}";
		}
		os << "]}";
	}
	if(!r.tuned_which.empty()) { // `--report` detailed: the optimize protocol's final hyperparameters
		os << "," << std::endl << "  \"tuned_parameters\": {\"classifier\": \"" << r.tuned_which << "\", \"c\": " << fmt_double(r.tuned_c) << ", \"gamma\": " << fmt_double(r.tuned_gamma);
		if(r.tuned_has_coef0) os << ", \"coef0\": " << fmt_double(r.tuned_coef0);
		os << "}";
	}
	if(r.has_bias) os << "," << std::endl << "  \"bias\": " << fmt_double(r.bias);
	if(!r.classified.empty()) {
		os << "," << std::endl << "  \"classified\": ";
		emit_array(os,r.classified);
	}
	if(!r.trials.empty()) {
		os << "," << std::endl << "  \"trials\": [";
		for(std::vector<RunResult::TrialEntry>::size_type i=0;i<r.trials.size();++i) {
			if(i) os << ", ";
			os << "{\"trial\": " << (i+1) << ", \"value\": " << fmt_double(r.trials[i].value) << ", \"size\": " << r.trials[i].features.size() << ", \"features\": ";
			emit_array(os,r.trials[i].features);
			os << "}";
		}
		os << "]";
	}
	if(!r.stability.empty()) {
		os << "," << std::endl << "  \"stability\": {";
		for(std::vector<RunResult::StabilityEntry>::size_type i=0;i<r.stability.size();++i) {
			if(i) os << ", ";
			os << "\"" << r.stability[i].measure << "\": " << fmt_double(r.stability[i].value);
		}
		os << "}";
	}
	os << "," << std::endl << "  \"top_k\": [";
	for(std::vector<RunResult::TopEntry>::size_type i=0;i<r.top_k.size();++i) {
		if(i) os << ", ";
		os << "{\"value\": " << fmt_double(r.top_k[i].value) << ", \"size\": " << r.top_k[i].features.size() << ", \"features\": ";
		emit_array(os,r.top_k[i].features);
		os << "}";
	}
	os << "]";
	if(!r.regularization.empty()) {
		os << "," << std::endl << "  \"regularization\": [";
		for(std::vector<RunResult::RegEntry>::size_type i=0;i<r.regularization.size();++i) {
			const RunResult::RegEntry &e=r.regularization[i];
			if(i) os << ", ";
			os << "{\"margin\": " << fmt_double(e.margin) << ", \"value\": " << fmt_double(e.value)
			   << ", \"" << (cfg.regularize_by=="weights"?"weight_sum":"subset_size") << "\": " << fmt_double(e.secondary)
			   << ", \"size\": " << e.features.size() << ", \"features\": ";
			emit_array(os,e.features);
			if(!e.validation.empty()) {
				os << ", \"validation\": [";
				for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {
					if(v) os << ", ";
					emit_validation_json(os,e.validation[v]);
				}
				os << "]";
			}
			os << "}";
		}
		os << "]";
	}
	if(!r.daf.empty()) {
		os << "," << std::endl << "  \"daf\": [";
		for(std::vector<RunResult::DafEntry>::size_type i=0;i<r.daf.size();++i) {
			if(i) os << ", ";
			os << "{\"feature\": " << r.daf[i].feature << ", \"value\": " << fmt_double(r.daf[i].value) << "}";
		}
		os << "]";
	}
	if(!r.daf_pairs.empty()) {
		os << "," << std::endl << "  \"daf_pairs\": [";
		for(std::vector<RunResult::PairDafEntry>::size_type i=0;i<r.daf_pairs.size();++i) {
			if(i) os << ", ";
			os << "{\"f1\": " << r.daf_pairs[i].f1 << ", \"f2\": " << r.daf_pairs[i].f2
			   << ", \"daf0\": " << fmt_double(r.daf_pairs[i].daf0)
			   << ", \"daf1\": "; if(r.daf_pairs[i].daf1_valid) os << fmt_double(r.daf_pairs[i].daf1); else os << "null";
			os << ", \"synergy\": " << fmt_double(r.daf_pairs[i].synergy) << "}";
		}
		os << "]";
	}
	if(!r.daf_validation.empty()) {
		os << "," << std::endl << "  \"daf_validation\": [";
		for(std::vector<RunResult::ValidationEntry>::size_type i=0;i<r.daf_validation.size();++i) {
			if(i) os << ", ";
			os << "{\"size\": " << r.daf_validation[i].classifier << ", \"accuracy\": " << fmt_double(r.daf_validation[i].accuracy) << "}";
		}
		os << "]";
	}
	if(!r.multiplier_sweep.empty()) {
		os << "," << std::endl << "  \"multiplier_sweep\": [";
		for(std::vector<RunResult::SweepEntry>::size_type i=0;i<r.multiplier_sweep.size();++i) {
			const RunResult::SweepEntry &e=r.multiplier_sweep[i];
			if(i) os << ", ";
			os << "{\"multiplier\": " << fmt_double(e.multiplier) << ", \"value\": " << fmt_double(e.value)
			   << ", \"size\": " << e.features.size() << ", \"features\": ";
			emit_array(os,e.features);
			if(!e.validation.empty()) {
				os << ", \"validation\": [";
				for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {
					if(v) os << ", ";
					emit_validation_json(os,e.validation[v]);
				}
				os << "]";
			}
			os << "}";
		}
		os << "]," << std::endl << "  \"best_multiplier\": " << fmt_double(r.best_multiplier);
	}
	if(cmp) { // `--compare-config`: the second scenario + cross-scenario tracker similarity (demo55)
		os << "," << std::endl << "  \"compare\": {\"config\": \"" << json_escape(cfg.compare_config) << "\"," << std::endl;
		os << "    \"scenario_b\": {\"criterion\": \"" << json_escape(cmp->cfg2.criterion) << "\", \"search\": \"" << json_escape(cmp->cfg2.search)
		   << "\", \"value\": " << fmt_double(cmp->r2.value) << ", \"size\": " << cmp->r2.features.size() << ", \"features\": ";
		emit_array(os,cmp->r2.features);
		if(!cmp->r2.trials.empty()) {
			os << ", \"trials\": [";
			for(std::vector<RunResult::TrialEntry>::size_type i=0;i<cmp->r2.trials.size();++i) {
				if(i) os << ", ";
				os << "{\"trial\": " << (i+1) << ", \"value\": " << fmt_double(cmp->r2.trials[i].value) << ", \"size\": " << cmp->r2.trials[i].features.size() << ", \"features\": ";
				emit_array(os,cmp->r2.trials[i].features);
				os << "}";
			}
			os << "]";
		}
		if(!cmp->r2.stability.empty()) {
			os << ", \"stability\": {";
			for(std::vector<RunResult::StabilityEntry>::size_type i=0;i<cmp->r2.stability.size();++i) {
				if(i) os << ", ";
				os << "\"" << cmp->r2.stability[i].measure << "\": " << fmt_double(cmp->r2.stability[i].value);
			}
			os << "}";
		}
		os << "}," << std::endl << "    \"similarity\": {";
		for(std::vector<RunResult::StabilityEntry>::size_type i=0;i<cmp->similarity.size();++i) {
			if(i) os << ", ";
			os << "\"" << cmp->similarity[i].measure << "\": " << fmt_double(cmp->similarity[i].value);
		}
		os << "}}";
	}
	if(!cfg.stable_output) {
		os << "," << std::endl << "  \"stats\": {\"wall_time_s\": " << fmt_double(r.wall_time_s)
		   << ", \"evaluations\": " << r.evaluations
		   << ", \"threads\": " << cfg.threads << "}";
	}
	os << std::endl << "}" << std::endl;
}

/*! \brief Writes the complete `--output` csv result: one long-format table, header
    "kind,rank,value,size,features", one row kind per RunResult block (result, top_k,
    per_split, validation, curve_over_d[_validation], roc, pr, confusion, tuned, bias,
    classified, trial, stability, regularization[_validation], daf, daf_pair, daf_validation,
    multiplier_sweep[_validation], best_multiplier, and the b_* / similarity rows under
    `--compare-config`), plus a trailing stats row unless `--stable-output`. Same real-stdout
    contract as emit_json(): with `--output` csv this is the only stdout content. */
static void emit_csv(std::ostream &os, const Config &cfg, const RunResult &r, const CompareResult *cmp=nullptr)
{
	os << "kind,rank,value,size,features" << std::endl;
	os << "result,0," << fmt_double(r.value) << "," << r.features.size() << ",";
	for(std::vector<DIMTYPE>::size_type i=0;i<r.features.size();++i) os << (i?" ":"") << r.features[i];
	os << std::endl;
	for(std::vector<RunResult::TopEntry>::size_type k=0;k<r.top_k.size();++k) {
		os << "top_k," << (k+1) << "," << fmt_double(r.top_k[k].value) << "," << r.top_k[k].features.size() << ",";
		for(std::vector<DIMTYPE>::size_type i=0;i<r.top_k[k].features.size();++i) os << (i?" ":"") << r.top_k[k].features[i];
		os << std::endl;
	}
	for(std::vector<RETURNTYPE>::size_type s=0;s<r.per_split.size();++s)
		os << "per_split," << (s+1) << "," << fmt_double(r.per_split[s]) << ",," << std::endl;
	for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<r.validation.size();++v)
		os << "validation," << (v+1) << "," << fmt_double(r.validation[v].accuracy) << ","
		   << (r.validation[v].metric.empty()?std::string():r.validation[v].metric+"="+fmt_double(r.validation[v].metric_value))
		   << "," << r.validation[v].classifier << std::endl;
	for(std::vector<RunResult::CurveEntry>::size_type i=0;i<r.curve_over_d.size();++i) { // `--report` detailed
		const RunResult::CurveEntry &e=r.curve_over_d[i];
		os << "curve_over_d," << e.d << "," << fmt_double(e.value) << "," << e.features.size() << ",";
		for(std::vector<DIMTYPE>::size_type f=0;f<e.features.size();++f) os << (f?" ":"") << e.features[f];
		os << std::endl;
		for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v)
			os << "curve_over_d_validation," << e.d << "," << fmt_double(e.validation[v].accuracy) << ","
			   << (e.validation[v].metric.empty()?std::string():e.validation[v].metric+"="+fmt_double(e.validation[v].metric_value))
			   << "," << e.validation[v].classifier << std::endl;
	}
	for(std::vector<RunResult::ClassCurve>::size_type c=0;c<r.curves.size();++c) { // `--report` detailed: roc = threshold,fpr,tpr; pr = threshold,recall,precision
		const RunResult::ClassCurve &cc=r.curves[c];
		for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.roc.size();++k)
			os << "roc," << cc.cls << "," << fmt_double(cc.roc[k].threshold) << "," << fmt_double(cc.roc[k].x) << "," << fmt_double(cc.roc[k].y) << std::endl;
		for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.pr.size();++k)
			os << "pr," << cc.cls << "," << fmt_double(cc.pr[k].threshold) << "," << fmt_double(cc.pr[k].x) << "," << fmt_double(cc.pr[k].y) << std::endl;
	}
	for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<r.validation.size();++v) // `--report` detailed: one row per true class
		if(r.validation[v].confusion_classes>0)
			for(DIMTYPE t=0;t<r.validation[v].confusion_classes;t++) {
				os << "confusion," << t << ",,";
				for(DIMTYPE q=0;q<r.validation[v].confusion_classes;q++) os << (q?" ":"") << fmt_double(r.validation[v].confusion[(size_t)t*(size_t)r.validation[v].confusion_classes+(size_t)q]);
				os << "," << r.validation[v].classifier << std::endl;
			}
	if(!r.tuned_which.empty()) { // `--report` detailed: C gamma [coef0]
		os << "tuned,," << fmt_double(r.tuned_c) << "," << fmt_double(r.tuned_gamma);
		if(r.tuned_has_coef0) os << " " << fmt_double(r.tuned_coef0);
		os << "," << r.tuned_which << std::endl;
	}
	if(r.has_bias) os << "bias,," << fmt_double(r.bias) << ",," << std::endl;
	for(std::vector<DIMTYPE>::size_type c=0;c<r.classified.size();++c)
		os << "classified," << (c+1) << "," << r.classified[c] << ",," << std::endl;
	for(std::vector<RunResult::TrialEntry>::size_type t=0;t<r.trials.size();++t) {
		os << "trial," << (t+1) << "," << fmt_double(r.trials[t].value) << "," << r.trials[t].features.size() << ",";
		for(std::vector<DIMTYPE>::size_type i=0;i<r.trials[t].features.size();++i) os << (i?" ":"") << r.trials[t].features[i];
		os << std::endl;
	}
	for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<r.stability.size();++s)
		os << "stability,," << fmt_double(r.stability[s].value) << ",," << r.stability[s].measure << std::endl;
	for(std::vector<RunResult::RegEntry>::size_type g=0;g<r.regularization.size();++g) {
		const RunResult::RegEntry &e=r.regularization[g];
		os << "regularization," << fmt_double(e.margin) << "," << fmt_double(e.value) << "," << e.features.size() << ",";
		for(std::vector<DIMTYPE>::size_type i=0;i<e.features.size();++i) os << (i?" ":"") << e.features[i];
		os << std::endl;
		for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v)
			os << "regularization_validation," << fmt_double(e.margin) << "," << fmt_double(e.validation[v].accuracy) << ","
			   << (e.validation[v].metric.empty()?std::string():e.validation[v].metric+"="+fmt_double(e.validation[v].metric_value))
			   << "," << e.validation[v].classifier << std::endl;
	}
	for(std::vector<RunResult::DafEntry>::size_type d=0;d<r.daf.size();++d)
		os << "daf," << (d+1) << "," << fmt_double(r.daf[d].value) << ",," << r.daf[d].feature << std::endl;
	for(std::vector<RunResult::PairDafEntry>::size_type d=0;d<r.daf_pairs.size();++d) {
		os << "daf_pair," << (d+1) << "," << fmt_double(r.daf_pairs[d].daf0) << ",";
		if(r.daf_pairs[d].daf1_valid) os << fmt_double(r.daf_pairs[d].daf1);
		os << " " << fmt_double(r.daf_pairs[d].synergy) << "," << r.daf_pairs[d].f1 << " " << r.daf_pairs[d].f2 << std::endl;
	}
	for(std::vector<RunResult::ValidationEntry>::size_type d=0;d<r.daf_validation.size();++d)
		os << "daf_validation," << r.daf_validation[d].classifier << "," << fmt_double(r.daf_validation[d].accuracy) << ",," << std::endl;
	for(std::vector<RunResult::SweepEntry>::size_type w=0;w<r.multiplier_sweep.size();++w) {
		const RunResult::SweepEntry &e=r.multiplier_sweep[w];
		os << "multiplier_sweep," << fmt_double(e.multiplier) << "," << fmt_double(e.value) << "," << e.features.size() << ",";
		for(std::vector<DIMTYPE>::size_type i=0;i<e.features.size();++i) os << (i?" ":"") << e.features[i];
		os << std::endl;
		for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v)
			os << "multiplier_sweep_validation," << fmt_double(e.multiplier) << "," << fmt_double(e.validation[v].accuracy) << ","
			   << (e.validation[v].metric.empty()?std::string():e.validation[v].metric+"="+fmt_double(e.validation[v].metric_value))
			   << "," << e.validation[v].classifier << std::endl;
	}
	if(!r.multiplier_sweep.empty())
		os << "best_multiplier,," << fmt_double(r.best_multiplier) << ",," << std::endl;
	if(cmp) { // `--compare-config`: scenario B rows (b_* kinds) + cross-scenario similarity (demo55)
		os << "b_result,0," << fmt_double(cmp->r2.value) << "," << cmp->r2.features.size() << ",";
		for(std::vector<DIMTYPE>::size_type i=0;i<cmp->r2.features.size();++i) os << (i?" ":"") << cmp->r2.features[i];
		os << std::endl;
		for(std::vector<RunResult::TrialEntry>::size_type t=0;t<cmp->r2.trials.size();++t) {
			os << "b_trial," << (t+1) << "," << fmt_double(cmp->r2.trials[t].value) << "," << cmp->r2.trials[t].features.size() << ",";
			for(std::vector<DIMTYPE>::size_type i=0;i<cmp->r2.trials[t].features.size();++i) os << (i?" ":"") << cmp->r2.trials[t].features[i];
			os << std::endl;
		}
		for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<cmp->r2.stability.size();++s)
			os << "b_stability,," << fmt_double(cmp->r2.stability[s].value) << ",," << cmp->r2.stability[s].measure << std::endl;
		for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<cmp->similarity.size();++s)
			os << "similarity,," << fmt_double(cmp->similarity[s].value) << ",," << cmp->similarity[s].measure << std::endl;
	}
	if(!cfg.stable_output)
		os << "stats,," << fmt_double(r.wall_time_s) << "," << r.evaluations << "," << cfg.threads << std::endl;
}

//! writes the complete `--output` text result (the default, human-readable format): the
//! fst4 report layout, extended with the `--report` detailed sections and the
//! `--compare-config` scenario-B/similarity block when present. Unlike emit_json()/emit_csv(),
//! text output is not machine-format-redirected -- \p os is real stdout regardless of
//! `--output`, since text mode never redirects std::cout to stderr
static void emit_text(std::ostream &os, const Config &cfg, const std::string &format,
                      const Pipeline &p, const RunResult &r, const CompareResult *cmp=nullptr)
{
	os << std::endl << "fst4 (FST4 " << FST4_VERSION << ") results:" << std::endl;
	if(cfg.data_free()) // `--features`: no file, no samples - the universe is abstract
		os << "data: universe features=" << p.da->getNoOfFeatures() << " (data-free, --features)" << std::endl;
	else
		os << "data: file=" << cfg.data << " format=" << format
		   << " classes=" << p.da->getNoOfClasses()
		   << " features=" << p.da->getNoOfFeatures()
		   << " samples=" << p.da->getClassSizeSum() << std::endl;
	os << "result: size=" << r.features.size() << " value=" << fmt_double(r.value) << " features=[";
	for(std::vector<DIMTYPE>::size_type i=0;i<r.features.size();++i) os << (i?" ":"") << r.features[i];
	os << "] (0-based)" << std::endl;
	if(r.per_split.size()>1) {
		os << "per-split values:";
		for(std::vector<RETURNTYPE>::size_type s=0;s<r.per_split.size();++s) os << " " << fmt_double(r.per_split[s]);
		os << std::endl;
	}
	for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<r.validation.size();++v) {
		os << "validated " << r.validation[v].classifier << " accuracy=" << fmt_double(r.validation[v].accuracy);
		if(!r.validation[v].metric.empty()) os << " " << r.validation[v].metric << "=" << fmt_double(r.validation[v].metric_value);
		os << " (trained on the depth-0 train part, tested on its test part)" << std::endl;
		if(r.validation[v].confusion_classes>0) { // `--report` detailed
			os << "  confusion matrix (row = true class):" << std::endl;
			for(DIMTYPE t=0;t<r.validation[v].confusion_classes;t++) {
				os << "   ";
				for(DIMTYPE q=0;q<r.validation[v].confusion_classes;q++) os << " " << fmt_double(r.validation[v].confusion[(size_t)t*(size_t)r.validation[v].confusion_classes+(size_t)q]);
				os << std::endl;
			}
		}
	}
	if(!r.curve_over_d.empty()) { // `--report` detailed: the criterion-axis vs validated-axis curve
		os << "curve over subset size (criterion value = selection-time estimate; validated = depth-0 holdout):" << std::endl;
		for(std::vector<RunResult::CurveEntry>::size_type i=0;i<r.curve_over_d.size();++i) {
			const RunResult::CurveEntry &e=r.curve_over_d[i];
			os << "  d=" << e.d << " value=" << fmt_double(e.value);
			for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {
				os << " validated_" << e.validation[v].classifier << "=" << fmt_double(e.validation[v].accuracy);
				if(!e.validation[v].metric.empty()) os << " " << e.validation[v].metric << "=" << fmt_double(e.validation[v].metric_value);
			}
			os << " features=[";
			for(std::vector<DIMTYPE>::size_type f=0;f<e.features.size();++f) os << (f?" ":"") << e.features[f];
			os << "]" << std::endl;
		}
	}
	for(std::vector<RunResult::ClassCurve>::size_type c=0;c<r.curves.size();++c) { // `--report` detailed
		const RunResult::ClassCurve &cc=r.curves[c];
		os << "ROC curve (classifier " << r.curves_classifier << ", positive class " << cc.cls << "; threshold FPR TPR):" << std::endl;
		for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.roc.size();++k)
			os << "  " << fmt_double(cc.roc[k].threshold) << " " << fmt_double(cc.roc[k].x) << " " << fmt_double(cc.roc[k].y) << std::endl;
		os << "PR curve (classifier " << r.curves_classifier << ", positive class " << cc.cls << "; threshold recall precision):" << std::endl;
		for(std::vector<RunResult::CurvePoint>::size_type k=0;k<cc.pr.size();++k)
			os << "  " << fmt_double(cc.pr[k].threshold) << " " << fmt_double(cc.pr[k].x) << " " << fmt_double(cc.pr[k].y) << std::endl;
	}
	if(!r.tuned_which.empty()) { // `--report` detailed
		os << "tuned " << r.tuned_which << " parameters: C=" << fmt_double(r.tuned_c) << " gamma=" << fmt_double(r.tuned_gamma);
		if(r.tuned_has_coef0) os << " coef0=" << fmt_double(r.tuned_coef0);
		os << std::endl;
	}
	if(r.has_bias)
		os << "estimated classifier bias on the selected subset: " << fmt_double(r.bias) << std::endl;
	if(!r.classified.empty()) {
		os << "classified samples (predicted class per input row):";
		for(std::vector<DIMTYPE>::size_type c=0;c<r.classified.size();++c) os << " " << r.classified[c];
		os << std::endl;
	}
	if(!r.trials.empty()) {
		os << "per-trial results (" << r.trials.size() << " runs):" << std::endl;
		for(std::vector<RunResult::TrialEntry>::size_type t=0;t<r.trials.size();++t) {
			os << "  trial " << (t+1) << ": value=" << fmt_double(r.trials[t].value) << " size=" << r.trials[t].features.size() << " features=[";
			for(std::vector<DIMTYPE>::size_type i=0;i<r.trials[t].features.size();++i) os << (i?" ":"") << r.trials[t].features[i];
			os << "]" << std::endl;
		}
	}
	if(!r.stability.empty()) {
		os << "stability measures over the trials:";
		for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<r.stability.size();++s)
			os << " " << r.stability[s].measure << "=" << fmt_double(r.stability[s].value);
		os << std::endl;
	}
	if(!r.regularization.empty()) {
		os << "regularized subsets (secondary criterion: " << (cfg.regularize_by=="weights"?"weight sum":"subset size") << "):" << std::endl;
		for(std::vector<RunResult::RegEntry>::size_type g=0;g<r.regularization.size();++g) {
			const RunResult::RegEntry &e=r.regularization[g];
			os << "  margin=" << fmt_double(e.margin) << ": value=" << fmt_double(e.value)
			   << " " << (cfg.regularize_by=="weights"?"weight_sum":"subset_size") << "=" << fmt_double(e.secondary)
			   << " size=" << e.features.size() << " features=[";
			for(std::vector<DIMTYPE>::size_type i=0;i<e.features.size();++i) os << (i?" ":"") << e.features[i];
			os << "]";
			for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {
				os << " validated_" << e.validation[v].classifier << "=" << fmt_double(e.validation[v].accuracy);
				if(!e.validation[v].metric.empty()) os << " " << e.validation[v].metric << "=" << fmt_double(e.validation[v].metric_value);
			}
			os << std::endl;
		}
	}
	if(!r.daf.empty()) {
		os << "DAF" << cfg.daf_rank << " feature ranking (best first):" << std::endl;
		for(std::vector<RunResult::DafEntry>::size_type d=0;d<r.daf.size();++d)
			os << "  " << (d+1) << ". feature " << r.daf[d].feature << " (value=" << fmt_double(r.daf[d].value) << ")" << std::endl;
	}
	if(!r.daf_pairs.empty()) {
		os << "Pairwise DAF (top pairs by pair-DAF0, best first):" << std::endl;
		for(std::vector<RunResult::PairDafEntry>::size_type d=0;d<r.daf_pairs.size();++d) {
			os << "  " << (d+1) << ". features (" << r.daf_pairs[d].f1 << "," << r.daf_pairs[d].f2 << ")"
			   << " pair-daf0=" << fmt_double(r.daf_pairs[d].daf0) << " pair-daf1=";
			if(r.daf_pairs[d].daf1_valid) os << fmt_double(r.daf_pairs[d].daf1); else os << "n/a";
			os << " synergy=" << fmt_double(r.daf_pairs[d].synergy) << std::endl;
		}
	}
	if(!r.daf_validation.empty()) {
		os << "top-DAF subset validation:";
		for(std::vector<RunResult::ValidationEntry>::size_type d=0;d<r.daf_validation.size();++d)
			os << " d=" << r.daf_validation[d].classifier << ": " << fmt_double(r.daf_validation[d].accuracy);
		os << std::endl;
	}
	if(!r.multiplier_sweep.empty()) {
		os << "kernel-multiplier sweep (best-by-train wins):" << std::endl;
		for(std::vector<RunResult::SweepEntry>::size_type w=0;w<r.multiplier_sweep.size();++w) {
			const RunResult::SweepEntry &e=r.multiplier_sweep[w];
			os << "  multiplier=" << fmt_double(e.multiplier) << ": value=" << fmt_double(e.value) << " size=" << e.features.size() << " features=[";
			for(std::vector<DIMTYPE>::size_type i=0;i<e.features.size();++i) os << (i?" ":"") << e.features[i];
			os << "]";
			for(std::vector<RunResult::ValidationEntry>::size_type v=0;v<e.validation.size();++v) {
				os << " validated_" << e.validation[v].classifier << "=" << fmt_double(e.validation[v].accuracy);
				if(!e.validation[v].metric.empty()) os << " " << e.validation[v].metric << "=" << fmt_double(e.validation[v].metric_value);
			}
			os << std::endl;
		}
		os << "best multiplier: " << fmt_double(r.best_multiplier) << std::endl;
	}
	if(!r.top_k.empty()) {
		os << "best " << r.top_k.size() << " tracked subsets:" << std::endl;
		for(std::vector<RunResult::TopEntry>::size_type k=0;k<r.top_k.size();++k) {
			os << "  " << (k+1) << ". value=" << fmt_double(r.top_k[k].value) << " size=" << r.top_k[k].features.size() << " features=[";
			for(std::vector<DIMTYPE>::size_type i=0;i<r.top_k[k].features.size();++i) os << (i?" ":"") << r.top_k[k].features[i];
			os << "]" << std::endl;
		}
	}
	if(cmp) { // `--compare-config`: scenario B + cross-scenario similarity (demo55)
		os << "scenario B (--compare-config " << cfg.compare_config << "): criterion=" << cmp->cfg2.criterion
		   << " search=" << cmp->cfg2.search << " value=" << fmt_double(cmp->r2.value)
		   << " size=" << cmp->r2.features.size() << " features=[";
		for(std::vector<DIMTYPE>::size_type i=0;i<cmp->r2.features.size();++i) os << (i?" ":"") << cmp->r2.features[i];
		os << "]" << std::endl;
		if(!cmp->r2.trials.empty()) {
			os << "scenario B per-trial results (" << cmp->r2.trials.size() << " runs):" << std::endl;
			for(std::vector<RunResult::TrialEntry>::size_type t=0;t<cmp->r2.trials.size();++t) {
				os << "  trial " << (t+1) << ": value=" << fmt_double(cmp->r2.trials[t].value) << " size=" << cmp->r2.trials[t].features.size() << " features=[";
				for(std::vector<DIMTYPE>::size_type i=0;i<cmp->r2.trials[t].features.size();++i) os << (i?" ":"") << cmp->r2.trials[t].features[i];
				os << "]" << std::endl;
			}
		}
		if(!cmp->r2.stability.empty()) {
			os << "scenario B stability measures over the trials:";
			for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<cmp->r2.stability.size();++s)
				os << " " << cmp->r2.stability[s].measure << "=" << fmt_double(cmp->r2.stability[s].value);
			os << std::endl;
		}
		os << "similarity of scenario A and scenario B:";
		for(std::vector<RunResult::StabilityEntry>::size_type s=0;s<cmp->similarity.size();++s)
			os << " " << cmp->similarity[s].measure << "=" << fmt_double(cmp->similarity[s].value);
		os << std::endl;
	}
	if(!cfg.stable_output)
		os << "stats: time=" << fmt_double(r.wall_time_s) << "s evaluations=" << r.evaluations << " threads=" << cfg.threads << std::endl;
}

// ---------------------------------------------------------------------------
// `--curves-svg`: a minimal dependency-free SVG plot writer: no plotting library, no external resources, no fonts beyond the generic
// sans-serif family — the file opens standalone in any browser and drops into LaTeX.

//! fixed palette cycled (by index % 6) across the per-class ROC/PR polylines in `--curves-svg` output
static const char *SVG_SERIES_COLORS[6]={"#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#8c564b"};

//! fixed-precision coordinate formatter — keeps the SVG byte-stable across runs
static std::string svg_num(double v)
{
	if(v==0.0) v=0.0; // normalizes -0.0, which would otherwise print as "-0.00"
	char b[32];
	std::snprintf(b,sizeof(b),"%.2f",v);
	return std::string(b);
}

//! short axis-tick formatter (4 significant digits is plenty for a tick label)
static std::string svg_tick(const double v)
{
	char b[32];
	std::snprintf(b,sizeof(b),"%.4g",v);
	return std::string(b);
}

//! XML-escapes text destined for an SVG `<text>` element (classifier names come from `--validate`)
static std::string svg_escape(const std::string &s)
{
	std::string out;
	for(std::string::const_iterator it=s.begin();it!=s.end();++it) {
		if(*it=='&') out+="&amp;";
		else if(*it=='<') out+="&lt;";
		else if(*it=='>') out+="&gt;";
		else out+=*it;
	}
	return out;
}

//! one plot panel: geometry plus the data->pixel mapping. The y mapping takes its range
//! per call so that a panel can carry two independently scaled axes (the over-d plot)
struct Svg_Panel {
	double left /*!< panel's left edge in SVG user units */, top /*!< panel's top edge */,
	       w /*!< panel width */, h; //!< panel height
	double xmin /*!< data-space x range lower bound */, xmax; //!< data-space x range upper bound
	//! maps a data-space x value to its pixel x coordinate within the panel (degenerate xmin==xmax centers)
	double px(const double v) const {return xmax>xmin ? left+(v-xmin)/(xmax-xmin)*w : left+w/2.0;}
	//! maps a data-space y value to its pixel y coordinate, using the given (possibly panel-external)
	//! y range so one panel can carry two independently scaled axes (degenerate ymin==ymax centers)
	double py(const double v, const double ymin, const double ymax) const
		{return ymax>ymin ? top+h-(v-ymin)/(ymax-ymin)*h : top+h/2.0;}
};

/*! `--curves-svg`: renders whatever curves `--report` detailed produced into one self-contained
    SVG — the ROC and PR curves of the validated classifier (one polyline per class on
    multiclass data, one-vs-rest) and the criterion-vs-holdout curve over subset sizes. The
    over-d panel carries TWO y-axes, left = criterion value and right = validated accuracy,
    because the train-criterion axis and the validated axis measure different things and
    must never be conflated on one scale. Returns false when the result carries no curve at all. */
static bool write_curves_svg(const std::string &path, const RunResult &r)
{
	bool has_roc=false, has_pr=false, has_dval=false;
	for(std::vector<RunResult::ClassCurve>::const_iterator it=r.curves.begin();it!=r.curves.end();++it) {
		if(!it->roc.empty()) has_roc=true;
		if(!it->pr.empty()) has_pr=true;
	}
	const bool has_d=!r.curve_over_d.empty();
	for(std::vector<RunResult::CurveEntry>::const_iterator it=r.curve_over_d.begin();it!=r.curve_over_d.end();++it)
		if(!it->validation.empty()) has_dval=true;
	if(!has_roc && !has_pr && !has_d) return false;

	const double W=560.0, PH=320.0, ML=64.0, MR=64.0, MT=44.0, MB=52.0;
	const double plotw=W-ML-MR, ploth=PH-MT-MB;
	const int npanels=(has_roc?1:0)+(has_pr?1:0)+(has_d?1:0);

	std::ofstream os(path.c_str());
	if(!os) throw FST::fst_error("--curves-svg: cannot open '"+path+"' for writing");
	os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl
	   << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_num(W) << "\" height=\"" << svg_num(npanels*PH+10.0)
	   << "\" viewBox=\"0 0 " << svg_num(W) << " " << svg_num(npanels*PH+10.0) << "\">" << std::endl
	   << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>" << std::endl;

	// frame, title, x ticks and x label; gridlines are drawn with the ticks
	auto draw_frame=[&os](const Svg_Panel &pn, const std::string &title, const std::string &xlab, const bool integer_x) {
		os << "<text x=\"" << svg_num(pn.left) << "\" y=\"" << svg_num(pn.top-20.0)
		   << "\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"bold\" fill=\"#222222\">" << svg_escape(title) << "</text>" << std::endl;
		os << "<rect x=\"" << svg_num(pn.left) << "\" y=\"" << svg_num(pn.top) << "\" width=\"" << svg_num(pn.w)
		   << "\" height=\"" << svg_num(pn.h) << "\" fill=\"none\" stroke=\"#333333\" stroke-width=\"1\"/>" << std::endl;
		for(int i=0;i<=4;i++) {
			double fx=pn.xmin+(pn.xmax-pn.xmin)*i/4.0;
			if(integer_x) fx=(double)(long long)(fx+0.5); // subset sizes are whole numbers
			const double x=pn.px(fx);
			os << "<line x1=\"" << svg_num(x) << "\" y1=\"" << svg_num(pn.top) << "\" x2=\"" << svg_num(x)
			   << "\" y2=\"" << svg_num(pn.top+pn.h) << "\" stroke=\"#eeeeee\" stroke-width=\"1\"/>" << std::endl;
			os << "<line x1=\"" << svg_num(x) << "\" y1=\"" << svg_num(pn.top+pn.h) << "\" x2=\"" << svg_num(x)
			   << "\" y2=\"" << svg_num(pn.top+pn.h+5.0) << "\" stroke=\"#333333\"/>" << std::endl;
			os << "<text x=\"" << svg_num(x) << "\" y=\"" << svg_num(pn.top+pn.h+18.0)
			   << "\" font-family=\"sans-serif\" font-size=\"10\" text-anchor=\"middle\" fill=\"#444444\">" << svg_tick(fx) << "</text>" << std::endl;
		}
		os << "<text x=\"" << svg_num(pn.left+pn.w/2.0) << "\" y=\"" << svg_num(pn.top+pn.h+38.0)
		   << "\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"middle\" fill=\"#222222\">" << svg_escape(xlab) << "</text>" << std::endl;
	};

	// y axis on either side, in its own colour so a dual-axis panel reads unambiguously
	auto draw_yaxis=[&os](const Svg_Panel &pn, const double ymin, const double ymax,
	                      const std::string &label, const char *color, const bool right) {
		const double ax = right ? pn.left+pn.w : pn.left;
		for(int i=0;i<=4;i++) {
			const double fy=ymin+(ymax-ymin)*i/4.0;
			const double y=pn.py(fy,ymin,ymax);
			os << "<line x1=\"" << svg_num(ax) << "\" y1=\"" << svg_num(y) << "\" x2=\"" << svg_num(right?ax+5.0:ax-5.0)
			   << "\" y2=\"" << svg_num(y) << "\" stroke=\"" << color << "\"/>" << std::endl;
			os << "<text x=\"" << svg_num(right?ax+9.0:ax-9.0) << "\" y=\"" << svg_num(y+3.5)
			   << "\" font-family=\"sans-serif\" font-size=\"10\" text-anchor=\"" << (right?"start":"end")
			   << "\" fill=\"" << color << "\">" << svg_tick(fy) << "</text>" << std::endl;
		}
		const double lx = right ? ax+52.0 : ax-50.0;
		const double ly = pn.top+pn.h/2.0;
		os << "<text x=\"" << svg_num(lx) << "\" y=\"" << svg_num(ly)
		   << "\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"middle\" fill=\"" << color
		   << "\" transform=\"rotate(-90 " << svg_num(lx) << " " << svg_num(ly) << ")\">" << svg_escape(label) << "</text>" << std::endl;
	};

	auto draw_polyline=[&os](const Svg_Panel &pn, const std::vector<std::pair<double,double> > &pts,
	                         const double ymin, const double ymax, const char *color, const bool marks) {
		if(pts.empty()) return;
		os << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"1.6\" points=\"";
		for(std::vector<std::pair<double,double> >::size_type i=0;i<pts.size();++i)
			os << (i?" ":"") << svg_num(pn.px(pts[i].first)) << "," << svg_num(pn.py(pts[i].second,ymin,ymax));
		os << "\"/>" << std::endl;
		if(marks) for(std::vector<std::pair<double,double> >::size_type i=0;i<pts.size();++i)
			os << "<circle cx=\"" << svg_num(pn.px(pts[i].first)) << "\" cy=\"" << svg_num(pn.py(pts[i].second,ymin,ymax))
			   << "\" r=\"2.4\" fill=\"" << color << "\"/>" << std::endl;
	};

	auto draw_legend=[&os](const double x, const double y, const std::vector<std::pair<std::string,const char*> > &items) {
		for(std::vector<std::pair<std::string,const char*> >::size_type i=0;i<items.size();++i) {
			const double ly=y+(double)i*15.0;
			os << "<line x1=\"" << svg_num(x) << "\" y1=\"" << svg_num(ly) << "\" x2=\"" << svg_num(x+18.0)
			   << "\" y2=\"" << svg_num(ly) << "\" stroke=\"" << items[i].second << "\" stroke-width=\"2.4\"/>" << std::endl;
			os << "<text x=\"" << svg_num(x+24.0) << "\" y=\"" << svg_num(ly+3.5)
			   << "\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#222222\">" << svg_escape(items[i].first) << "</text>" << std::endl;
		}
	};

	int panel=0;
	const std::string who = r.curves_classifier.empty() ? std::string() : (" — classifier: "+r.curves_classifier);

	if(has_roc) { // ROC: fixed [0,1]^2 so runs stay visually comparable; the (0,0) origin is implied by extract_curves
		Svg_Panel pn; pn.left=ML; pn.top=panel*PH+MT; pn.w=plotw; pn.h=ploth; pn.xmin=0.0; pn.xmax=1.0;
		draw_frame(pn,"ROC"+who,"false positive rate",false);
		draw_yaxis(pn,0.0,1.0,"true positive rate","#333333",false);
		os << "<line x1=\"" << svg_num(pn.px(0.0)) << "\" y1=\"" << svg_num(pn.py(0.0,0.0,1.0)) << "\" x2=\"" << svg_num(pn.px(1.0))
		   << "\" y2=\"" << svg_num(pn.py(1.0,0.0,1.0)) << "\" stroke=\"#bbbbbb\" stroke-width=\"1\" stroke-dasharray=\"4,4\"/>" << std::endl;
		std::vector<std::pair<std::string,const char*> > legend;
		for(std::vector<RunResult::ClassCurve>::size_type i=0;i<r.curves.size();++i) {
			if(r.curves[i].roc.empty()) continue;
			const char *color=SVG_SERIES_COLORS[i%6];
			std::vector<std::pair<double,double> > pts;
			pts.push_back(std::make_pair(0.0,0.0));
			for(std::vector<RunResult::CurvePoint>::const_iterator q=r.curves[i].roc.begin();q!=r.curves[i].roc.end();++q)
				pts.push_back(std::make_pair((double)q->x,(double)q->y));
			draw_polyline(pn,pts,0.0,1.0,color,false);
			legend.push_back(std::make_pair("class "+std::to_string((long long)r.curves[i].cls),color));
		}
		// bottom-anchored: grows upward with the entry count (empty guarded — has_roc implies
		// at least one entry, but size()-1 on an unsigned zero would place it off-canvas)
		if(legend.size()>1) draw_legend(pn.left+pn.w-96.0,pn.top+pn.h-20.0-15.0*(double)(legend.size()-1),legend);
		panel++;
	}

	if(has_pr) { // precision-recall, same fixed square
		Svg_Panel pn; pn.left=ML; pn.top=panel*PH+MT; pn.w=plotw; pn.h=ploth; pn.xmin=0.0; pn.xmax=1.0;
		draw_frame(pn,"Precision-recall"+who,"recall",false);
		draw_yaxis(pn,0.0,1.0,"precision","#333333",false);
		std::vector<std::pair<std::string,const char*> > legend;
		for(std::vector<RunResult::ClassCurve>::size_type i=0;i<r.curves.size();++i) {
			if(r.curves[i].pr.empty()) continue;
			const char *color=SVG_SERIES_COLORS[i%6];
			std::vector<std::pair<double,double> > pts;
			for(std::vector<RunResult::CurvePoint>::const_iterator q=r.curves[i].pr.begin();q!=r.curves[i].pr.end();++q)
				pts.push_back(std::make_pair((double)q->x,(double)q->y));
			draw_polyline(pn,pts,0.0,1.0,color,false);
			legend.push_back(std::make_pair("class "+std::to_string((long long)r.curves[i].cls),color));
		}
		if(legend.size()>1) draw_legend(pn.left+pn.w-96.0,pn.top+20.0,legend); // top-anchored: PR curves fall away from that corner
		panel++;
	}

	if(has_d) { // the overfitting plot: criterion value (left axis) against holdout accuracy (right axis)
		Svg_Panel pn; pn.left=ML; pn.top=panel*PH+MT; pn.w=plotw; pn.h=ploth;
		pn.xmin=(double)r.curve_over_d.front().d; pn.xmax=(double)r.curve_over_d.back().d;
		double cmin=0,cmax=0,vmin=0,vmax=0;
		bool cfirst=true, vfirst=true;
		std::string vname;
		std::vector<std::pair<double,double> > cpts, vpts;
		for(std::vector<RunResult::CurveEntry>::const_iterator e=r.curve_over_d.begin();e!=r.curve_over_d.end();++e) {
			const double x=(double)e->d, y=(double)e->value;
			cpts.push_back(std::make_pair(x,y));
			if(cfirst) {cmin=cmax=y; cfirst=false;} else {if(y<cmin) cmin=y; if(y>cmax) cmax=y;}
			if(!e->validation.empty()) { // the first `--validate` entry is the validated axis
				const double a=(double)e->validation[0].accuracy;
				vpts.push_back(std::make_pair(x,a));
				if(vfirst) {vmin=vmax=a; vname=e->validation[0].classifier; vfirst=false;}
				else {if(a<vmin) vmin=a; if(a>vmax) vmax=a;}
			}
		}
		// pad a degenerate (flat) range so the polyline lands mid-panel instead of on an edge
		if(cmax<=cmin) {cmin-=0.5; cmax+=0.5;}
		if(vmax<=vmin) {vmin-=0.5; vmax+=0.5;}
		draw_frame(pn,"Criterion value and holdout validation over subset size","subset size d",true);
		draw_yaxis(pn,cmin,cmax,"criterion value (train)",SVG_SERIES_COLORS[0],false);
		draw_polyline(pn,cpts,cmin,cmax,SVG_SERIES_COLORS[0],true);
		std::vector<std::pair<std::string,const char*> > legend;
		legend.push_back(std::make_pair("criterion (left axis)",SVG_SERIES_COLORS[0]));
		if(has_dval && !vpts.empty()) {
			draw_yaxis(pn,vmin,vmax,"validated accuracy (holdout)",SVG_SERIES_COLORS[1],true);
			draw_polyline(pn,vpts,vmin,vmax,SVG_SERIES_COLORS[1],true);
			legend.push_back(std::make_pair("accuracy, "+vname+" (right axis)",SVG_SERIES_COLORS[1]));
		}
		draw_legend(pn.left+10.0,pn.top+16.0,legend);
		panel++;
	}

	os << "</svg>" << std::endl;
	os.close();
	if(!os) throw FST::fst_error("--curves-svg: writing '"+path+"' failed");
	return true;
}

// ---------------------------------------------------------------------------

//! `--list`: the component registry with one-line descriptions and validity notes
static void print_components(std::ostream &os)
{
	os << "fst4 (FST4 " << FST4_VERSION << ") available components\n\n"
	   << "data formats (--format; default: inferred from the file extension)\n"
	   << "  trn                    FST TRN format\n"
	   << "  arff                   Weka ARFF format\n"
	   << "  csv                    delimited text, the simplest Excel-export path (.csv/.tsv; delimiter, header row and\n"
	   << "                         decimal-comma handling auto-detected — see --csv-class, --csv-delimiter, --csv-header;\n"
	   << "                         rows with missing values are skipped loudly, or kept with --csv-missing-to)\n"
	   << "  --data -               read the dataset from standard input instead of a file (--format is then\n"
	   << "                         required; with --output json|csv fst4 is fully pipe-transparent: data in on\n"
	   << "                         stdin, result out on stdout)\n"
	   << "  --features N           NO data at all: optimize over an abstract universe of N features with an\n"
	   << "                         external set function (--criterion extern-value --extern-no-data; everything\n"
	   << "                         data-centric is refused - FST4 as a black-box set-function optimizer)\n\n"
	   << "splitters (--splitter = outer/depth 0, --splitter2 = inner/depth 1)\n"
	   << "  5050                   1st half train, 2nd half test (deterministic)\n"
	   << "  cv                     k-fold cross-validation (--cv-folds / --splitter2-cv-folds)\n"
	   << "  holdout                first x% train, rest test (--holdout-train)\n"
	   << "  resub                  train = test = all data (resubstitution)\n"
	   << "  leave1out              leave-one-out cross-validation (intended as --splitter2)\n"
	   << "  randrand               random x% train / y% test, disjoint (--rr-splits/--rr-train/--rr-test); multi-split capable\n"
	   << "  randfix                like randrand with fixed test part across splits\n\n"
	   << "scalers (--scaler)\n"
	   << "  void                   no scaling (default)\n"
	   << "  to01                   normalize each feature to [0,1]\n"
	   << "  white                  whitening (zero mean, unit variance)\n"
	   << "  (--scaler-fit all|train: what the transform is learned from - all = the whole\n"
	   << "   dataset before splitting (default); train = the current depth-0\n"
	   << "   TRAIN part only, re-fitted per split = leakage-free, costs one rescale pass\n"
	   << "   per split change; --classify-samples vectors get the trained-on split's transform)\n\n"
	   << "criteria (--criterion)\n"
	   << "  bhattacharyya          normal-model filter; monotonic (B&B-compatible); two-class data only\n"
	   << "  divergence             normal-model filter; monotonic in theory but numerically unstable - prefer bhattacharyya; two-class only\n"
	   << "  gmahalanobis           normal-model filter (generalized Mahalanobis); monotonic; two-class only\n"
	   << "                         NOTE: normal-model filters fail on data with constant features (singular covariance)\n"
	   << "  multinom-bhattacharyya multinomial-model filter; monotonic; assumes count/text data (e.g. term frequencies)\n"
	   << "  multinom-mi            multinomial-model individual mutual information (ranking criterion); subset value =\n"
	   << "                         SUM of per-feature MI (monotonic, NOT a joint measure) - intended for --search bif\n"
	   << "  wrapper-knn            kNN classifier accuracy (--knn-k, --knn-distance L1|L2|Lp with p=3/2)\n"
	   << "  wrapper-bayes          normal Bayes classifier accuracy\n"
	   << "  wrapper-logreg         multinomial logistic regression accuracy (--logreg-lambda/-maxiter/-tol); deterministic\n"
	   << "                         damped-Newton training on internally standardized data; constant features are safe\n"
	   << "  wrapper-rf             Random Forest accuracy (--rf-trees/-mtry/-max-depth/-min-leaf); dependency-free CART+bagging,\n"
	   << "                         deterministic by design: training seeds derive from (subset, --seed, split) — never the fst_rand stream\n"
	   << "  wrapper-linsvm         linear SVM accuracy (--linsvm-c/-loss/-maxiter/-tol); dependency-free dual coordinate descent,\n"
	   << "                         fixed cyclic order = RNG-free/deterministic; one-vs-rest multiclass; no LibSVM needed (and no\n"
	   << "                         LINEAR-kernel freeze LibSVM can hit on some data)\n"
	   << "  wrapper-rbfsvm         RBF kernel SVM accuracy (--rbfsvm-c/-gamma/-loss/-maxiter/-tol, --rbfsvm-optimize); dependency-free\n"
	   << "                         kernel dual coordinate descent (C-SVC, penalized bias), fixed cyclic order = RNG-free/deterministic;\n"
	   << "                         one-vs-rest multiclass; gamma default 1/d on internally standardized features; no LibSVM needed\n"
	   << "  wrapper-lda            shared-covariance linear discriminant accuracy; closed-form/deterministic, K classes,\n"
	   << "                         one pooled-covariance inverse per training (fails on singular covariance, like normal filters)\n"
	   << "  wrapper-extern         external classifier over a persistent-process line protocol (--extern-cmd; see\n"
	   << "                         doc/FST4_CLI.md). WARNING: user-supplied code execution; reproducibility ends at the bridge\n"
	   << "  extern-value           external CRITERION over the DATA/VALUE line protocol (--extern-cmd): the train part\n"
	   << "                         streams once per split, then the tool answers index-only VALUE requests with the\n"
	   << "                         criterion value itself; --extern-no-data suppresses DATA entirely (pure set-function\n"
	   << "                         optimization); --extern-monotonic asserts cardinality-monotonicity and admits bb-*\n"
	   << "                         searches (optimality rests on the assertion). Same WARNING as wrapper-extern:\n"
	   << "                         code execution, reproducibility ends at the bridge\n"
	   << "  wrapper-multinom       multinomial naive-Bayes accuracy; assumes count/text data\n"
	   << "  wrapper-svm            LibSVM accuracy (--svm-kernel, --svm-c/-gamma/-coef0, --svm-optimize); rbf kernel is\n"
	   << "                         the most stable (linear/sigmoid may freeze LibSVM on some data)\n"
	   << "  wrapper-regressor      weighted-neighbours kNN regression; dependent variable = the class label (regression\n"
	   << "                         simulating classification - the demo11reg protocol)\n"
	   << "                         (--regr-multiplier / --regr-multipliers sweep); validate with 'same' only\n"
	   << "  wrapper-ridge          closed-form ridge regression (--ridge-lambda); same dependent-variable convention;\n"
	   << "                         criterion value defaults to r2 (see --wrapper-metric); deterministic, RNG-free\n"
	   << "  (all wrapper-* criteria accept --wrapper-metric: balanced-accuracy|f1|f1-macro|auc for classifiers,\n"
	   << "   r2|neg-rmse|neg-mae for the regressor, plus --class-weights; default accuracy;\n"
	   << "   --f1-positive picks the class f1 scores as positive on two-class data, default 1)\n\n"
	   << "searches (--search)\n"
	   << "  bif                    best individual features ranking; requires --target-d with monotonic criteria\n"
	   << "  sfs / sbs              sequential forward / backward selection\n"
	   << "  sffs / sbfs            floating forward / backward search\n"
	   << "  sfrs / sbrs            retreating (more thorough, slower) forward / backward search\n"
	   << "  os / dos               (dynamic) oscillating search (--os-delta / --dos-delta); os requires --target-d\n"
	   << "                         sequential family extras: --generalization-level, --evaluator, --tracker-attach\n"
	   << "  exhaustive             optimal exhaustive enumeration; any criterion; exponential cost\n"
	   << "  montecarlo             random probes (--mc-trials/--mc-seconds/--mc-card-min/-max/--mc-card-prob);\n"
	   << "                         with --tracker feature-stats + --daf-rank = the DAF ranking protocol\n"
	   << "  bb-basic / bb-improved / bb-fast / bb-bbpp\n"
	   << "                         optimal Branch & Bound variants; REQUIRE a monotonic filter criterion and --target-d;\n"
	   << "                         bb-fast tuning: --bb-gamma/--bb-delta; all four accept --threads N\n"
	   << "                         (candidate-level batches; results identical to --threads 1; NOTE threading\n"
	   << "                         pays only with expensive criteria - with the cheap built-in filters at\n"
	   << "                         small d the many small batches make --threads 1 the faster choice)\n"
	   << "\n"
	   << "step evaluators (--evaluator; sequential searches only; all accept --threads N,\n"
	   << "                 results identical to --threads 1 - proposals/pre-sorts/votes stay on the master)\n"
	   << "  straight               plain candidate evaluation (default); --threads N picks the thread-pooled variant\n"
	   << "  hybrid                 --hybrid-prefilter pre-orders candidates (on the master), top --hybrid-keep-perc%\n"
	   << "                         reach the criterion; --threads N thread-pools the main-criterion phase\n"
	   << "  ensemble               kNN wrappers with --ensemble-knn-ks vote on candidate ranking;\n"
	   << "                         --threads N thread-pools each member's candidate evaluations\n"
	   << "  sampled                budgeted stochastic step: forward steps evaluate only --step-cap candidates\n"
	   << "                         proposed from online per-feature statistics (--step-explore/--step-tau/\n"
	   << "                         --step-decay/--step-sampler/--step-stats; seed with --warmup-probes/--warmup-card;\n"
	   << "                         ablations: --step-cap-frac percentage schedule, --step-frozen no-update ranking);\n"
	   << "                         backward steps sweep in full unless --step-cap-backward budgets them\n"
	   << "                         symmetrically (the method of Somol and Grim, arXiv:2608.01502);\n"
	   << "                         --threads N evaluates each step's proposal batch in a thread pool\n"
	   << "                         (results bit-identical to --threads 1 at matched --seed)\n\n"
	   << "trackers (--tracker; --tracker-capacity/--tracker-margin/--tracker-attach)\n"
	   << "  dupless                best-k distinct subsets (default; feeds the top_k output block)\n"
	   << "  regularizer            records candidates for the --regularize-by/--regularize-margins sweep\n"
	   << "  stabileval             per-trial stability statistics; requires --repeat-over-splits\n"
	   << "  feature-stats          probe statistics; --daf-rank 0|1 extracts the DAF ranking (the published 2011 method),\n"
	   << "                         --daf-pairs K adds pairwise (group) statistics over the top-K DAF0 features\n\n"
	   << "result reporting\n"
	   << "  --output text|json|csv          result format; json/csv keep stdout machine-pure (prose goes to stderr)\n"
	   << "  --report standard|detailed      detailed adds research sections: curve_over_d (criterion vs holdout per\n"
	   << "                                  subset size), ROC/PR curves (tabular; per-class one-vs-rest on multiclass),\n"
	   << "                                  confusion matrices, tuned optimize-protocol hyperparameters\n"
	   << "  --dump-scores FILE              per-holdout-sample true/predicted labels + per-class scores (CSV;\n"
	   << "                                  the first --validate classifier) - external ROC/calibration/error analysis\n"
	   << "  --curves-svg FILE               self-contained SVG of the --report detailed curves (ROC, PR, criterion\n"
	   << "                                  vs holdout over d on separate axes); no plotting dependency\n\n"
	   << "run protocols\n"
	   << "  (default)              one search + optional --validate on the depth-0 holdout\n"
	   << "  --repeat-over-splits   one search per outer split (stability trials, demo54; wrappers need --splitter2)\n"
	   << "  --restarts-nonimproving N   random-restart search (demo22); requires --target-d\n"
	   << "  --regr-multipliers a,b,c    regression kernel sweep, best-by-train wins (demo11reg)\n"
	   << "  --search2 bif          BIF pre-ranking seeds the main search (demo25t/33)\n"
	   << "  --svm-optimize pre|alternating   LibSVM parameter grid tuning (demo12t/23)\n"
	   << "  --rbfsvm-optimize pre|alternating   native RBF SVM C/gamma tuning, the same protocol without LibSVM (demo29)\n"
	   << "  --bias-estimate        wrapper-classifier optimism estimate (demo56)\n"
	   << "  --classify-samples F   classify new samples on the selected subspace (demo63)\n";
}

/*! \brief Registers every fst4 CLI option/flag on \p app, binding each directly into a field
    of \p cfg (CLI11's add_option()/add_flag() write straight into the referenced variable on
    parse). Factored out of main() so that `--compare-config` can parse a second Config through
    an identical option set (the demo55 cross-scenario similarity protocol) -- see parse_compare_config().
    The third argument of each add_option()/add_flag() call is that option's authoritative
    help text (also what `--help` prints), so option semantics are best read there rather than
    duplicated in comments here. */
static void add_options(CLI::App &app, Config &cfg)
{
	// `--data` lost its ->required() when `--features` arrived (the data-free entry has no file at
	// all); the exactly-one-of-the-two rule is enforced after parsing, where the error message
	// can name both flags
	app.add_option("--data",cfg.data,"path to the data file (TRN, ARFF, or CSV format), or - to read the dataset from standard input (--format is then required, there being no file name to infer it from)")
		->check([](const std::string &s)->std::string{
			if(s=="-") return std::string();   // stdin: buffered by Stdin_Data_File in main()
			return CLI::ExistingFile(s);
		},"FILE or - (stdin)");
	app.add_option("--features",cfg.features_n,"data-free entry point: optimize over an abstract universe of N features with NO data file at all - requires --criterion extern-value --extern-no-data (the external tool answers index-only VALUE requests from its own logic: cost tables, simulations, external experiments; FST4 as a black-box set-function optimizer). Mutually exclusive with --data; everything data-centric (--validate, splitters, scalers, wrapper criteria) is refused")->check(CLI::Range(1u,1000000u));
	app.add_option("--format",cfg.format,"data file format (default: inferred from file extension; .csv/.tsv both mean csv)")->check(CLI::IsMember({"trn","arff","csv"}));
	app.add_option("--csv-class",cfg.csv_class,"CSV class column: a 0-based column index, or a header column name (default: the last column)");
	app.add_option("--csv-delimiter",cfg.csv_delimiter,"CSV field delimiter (default auto = detected from the first line; decimal commas accepted when the delimiter is not the comma)")->check(CLI::IsMember({"auto","comma","semicolon","tab"}));
	app.add_option("--csv-header",cfg.csv_header,"CSV header row present (default auto = detected from line 1; a name in --csv-class implies yes)")->check(CLI::IsMember({"auto","yes","no"}));
	app.add_option("--csv-missing-to",cfg.csv_missing_to,"map missing CSV feature values (empty, ?, NA, N/A, NaN, NULL) to this numeric code and keep the row, instead of skipping the whole row; pair with --missing-value <same code> to substitute the per-feature average of the available values (see --missing-value; demo62 protocol), or leave the code in place as a missingness indicator (rows with a missing class label are still skipped; void scaler only)");
	app.add_option("--splitter",cfg.splitter,"outer train/test data splitting strategy, splitting depth 0 (default 5050)")->check(CLI::IsMember({"5050","cv","holdout","resub","randrand","randfix","leave1out"}));
	app.add_option("--cv-folds",cfg.cv_folds,"number of cross-validation folds (with --splitter cv; default 3)")->check(CLI::Range(2u,1000u));
	app.add_option("--holdout-train",cfg.holdout_train,"percentage of data used for training (with --splitter holdout; default 70)")->check(CLI::Range(1u,99u));
	app.add_option("--rr-splits",cfg.rr_splits,"number of random splits (with --splitter randrand/randfix; default 1)")->check(CLI::Range(1u,100000u));
	app.add_option("--rr-train",cfg.rr_train,"percentage of samples drawn randomly for training (with --splitter randrand/randfix; default 50)")->check(CLI::Range(1u,99u));
	app.add_option("--rr-test",cfg.rr_test,"percentage of samples drawn for testing, disjoint from training (with --splitter randrand/randfix; default 50)")->check(CLI::Range(1u,99u));
	app.add_option("--splitter2",cfg.splitter2,"inner splitting strategy, splitting depth 1: the criterion protocol runs on these splits of the outer train part while --validate uses the outer holdout (demo choreography; default: none = criterion runs directly on depth-0 splits)")
		->check(CLI::IsMember({"5050","cv","holdout","resub","randrand","randfix","leave1out"}));
	app.add_option("--splitter2-cv-folds",cfg.cv2_folds,"number of inner cross-validation folds (with --splitter2 cv; default 3)")->check(CLI::Range(2u,1000u));
	app.add_option("--splitter2-holdout-train",cfg.holdout2_train,"inner training percentage (with --splitter2 holdout; default 70)")->check(CLI::Range(1u,99u));
	app.add_option("--splitter2-rr-splits",cfg.rr2_splits,"number of inner random splits (with --splitter2 randrand/randfix; default 1)")->check(CLI::Range(1u,100000u));
	app.add_option("--splitter2-rr-train",cfg.rr2_train,"inner random training percentage (with --splitter2 randrand/randfix; default 50)")->check(CLI::Range(1u,99u));
	app.add_option("--splitter2-rr-test",cfg.rr2_test,"inner random testing percentage (with --splitter2 randrand/randfix; default 50)")->check(CLI::Range(1u,99u));
	app.add_option("--scaler",cfg.scaler,"data normalization (default void = none)")->check(CLI::IsMember({"void","to01","white"}));
	app.add_option("--scaler-fit",cfg.scaler_fit,"what the scaler learns its transform from: all = the entire dataset before splitting (default; the FST3 behavior), train = the current depth-0 split's TRAIN part only, re-fitted on every depth-0 split change and applied to all data (leakage-free normalization)")->check(CLI::IsMember({"all","train"}));
	app.add_option("--missing-value",cfg.missing_value,"treat this exact value as a missing-data code and substitute the per-feature average of the available values - of the whole dataset by default, of the current depth-0 TRAIN part under --scaler-fit train (demo62 protocol; void scaler only)");
	app.add_option("--criterion",cfg.criterion,"feature selection criterion; bhattacharyya/divergence/gmahalanobis are normal-model filters (two-class data only), multinom-bhattacharyya is a multinomial-model filter (count/text data), wrapper-* estimate classifier accuracy")->required()
		->check(CLI::IsMember({"bhattacharyya","divergence","gmahalanobis","multinom-bhattacharyya","multinom-mi","wrapper-knn","wrapper-bayes","wrapper-lda","wrapper-logreg","wrapper-rf","wrapper-linsvm","wrapper-rbfsvm","wrapper-multinom","wrapper-svm","wrapper-extern","extern-value","wrapper-regressor","wrapper-ridge"}));
	app.add_option("--wrapper-metric",cfg.wrapper_metric,"wrapper criterion metric (with --criterion wrapper-*; default accuracy): balanced-accuracy|f1|f1-macro|auc for classification wrappers, r2|neg-rmse|neg-mae for wrapper-regressor; --validate reports it alongside accuracy")
		->check(CLI::IsMember({"accuracy","balanced-accuracy","f1","f1-macro","auc","r2","neg-rmse","neg-mae"}));
	app.add_option("--class-weights",cfg.class_weights,"comma-separated per-class cost weights in class-index order (with classification wrapper-* criteria); weighs confusion-matrix rows by true class, e.g. 1,5 makes class-1 errors five times as costly");
	app.add_option("--f1-positive",cfg.f1_positive,"class index scored as the positive class by --wrapper-metric f1 on two-class data (default 1; FST class order is the data-file class order). Affects f1 only - f1-macro averages over all classes, and auc keeps class 1 positive by its own convention")->check(CLI::Range(0u,1u));
	app.add_option("--knn-k",cfg.knn_k,"k for the kNN classifier - the wrapper-knn criterion and the knn validator alike (default 3)")->check(CLI::Range(1u,10000u));
	app.add_option("--extern-cmd",cfg.extern_cmd,"external tool command (with --criterion wrapper-extern or extern-value); launched ONCE per criterion clone through the system shell and driven over a line protocol - wrapper-extern: TRAIN/TEST label exchange (reference tests/extern_centroid.awk), extern-value: train data once per split (DATA), then index-only VALUE requests answered with the criterion value itself (reference tests/extern_value_meandiff.awk); see doc/FST4_CLI.md. WARNING: user-supplied code execution; reproducibility guarantees end at the bridge");
	app.add_flag("--extern-no-data",cfg.extern_no_data,"suppress the DATA block of the extern-value protocol: the external tool receives index-only VALUE requests and computes from its own logic - pure set-function optimization over an abstract feature universe (with --data the file then only defines the number of features; --features N drops the data file entirely)");
	app.add_flag("--extern-monotonic",cfg.extern_monotonic,"assert that the --criterion extern-value measure is monotonic w.r.t. subset cardinality (removing a feature never increases its value): admits the optimal bb-* searches - OPTIMALITY THEN RESTS ENTIRELY ON THIS ASSERTION; a false assertion yields silently suboptimal results");
	app.add_option("--linsvm-c",cfg.linsvm_c,"linear SVM regularization parameter C (with --criterion wrapper-linsvm; default 1.0)")->check(CLI::Range(1e-9,1e9));
	app.add_option("--linsvm-loss",cfg.linsvm_loss,"linear SVM loss (with --criterion wrapper-linsvm; default l2 = squared hinge)")->check(CLI::IsMember({"l1","l2"}));
	app.add_option("--linsvm-maxiter",cfg.linsvm_maxiter,"linear SVM dual coordinate-descent sweep cap (with --criterion wrapper-linsvm; default 1000)")->check(CLI::Range(1u,1000000u));
	app.add_option("--linsvm-tol",cfg.linsvm_tol,"linear SVM projected-gradient stopping threshold (with --criterion wrapper-linsvm; default 0.1)")->check(CLI::Range(1e-12,1.0));
	app.add_option("--rbfsvm-c",cfg.rbfsvm_c,"RBF SVM regularization parameter C (with --criterion wrapper-rbfsvm; default 1.0)")->check(CLI::Range(1e-9,1e9));
	app.add_option("--rbfsvm-gamma",cfg.rbfsvm_gamma,"RBF kernel width on the internally standardized features (with --criterion wrapper-rbfsvm; default 0 = auto: 1/subset-size)")->check(CLI::Range(0.0,1e9));
	app.add_option("--rbfsvm-loss",cfg.rbfsvm_loss,"RBF SVM loss (with --criterion wrapper-rbfsvm; default l1 = hinge, the C-SVC correspondence)")->check(CLI::IsMember({"l1","l2"}));
	app.add_option("--rbfsvm-maxiter",cfg.rbfsvm_maxiter,"RBF SVM kernel dual coordinate-descent sweep cap (with --criterion wrapper-rbfsvm; default 1000)")->check(CLI::Range(1u,1000000u));
	app.add_option("--rbfsvm-tol",cfg.rbfsvm_tol,"RBF SVM projected-gradient stopping threshold (with --criterion wrapper-rbfsvm; default 0.1)")->check(CLI::Range(1e-12,1.0));
	app.add_option("--rbfsvm-optimize",cfg.rbfsvm_optimize,"native RBF SVM C/gamma optimization (with --criterion wrapper-rbfsvm; the --svm-optimize protocol): pre = once on the full feature set before the search (and on the --search2 seed subset); alternating = additionally re-optimize and re-search while the criterion improves (default none)")
		->check(CLI::IsMember({"none","pre","alternating"}));
	app.add_option("--rf-trees",cfg.rf_trees,"Random Forest tree count (with --criterion wrapper-rf; default 100)")->check(CLI::Range(1u,100000u));
	app.add_option("--rf-mtry",cfg.rf_mtry,"features tried per tree node (with --criterion wrapper-rf; default 0 = floor(sqrt(subset size)))")->check(CLI::Range(0u,1000000u));
	app.add_option("--rf-max-depth",cfg.rf_max_depth,"Random Forest tree depth cap (with --criterion wrapper-rf; default 0 = unlimited)")->check(CLI::Range(0u,1000000u));
	app.add_option("--rf-min-leaf",cfg.rf_min_leaf,"minimum training samples per tree leaf (with --criterion wrapper-rf; default 1)")->check(CLI::Range(1u,1000000u));
	app.add_option("--logreg-lambda",cfg.logreg_lambda,"logistic regression L2 weight penalty, biases unpenalized (with --criterion wrapper-logreg; default 0.01)")->check(CLI::Range(0.0,1e6));
	app.add_option("--logreg-maxiter",cfg.logreg_maxiter,"logistic regression Newton iteration cap (with --criterion wrapper-logreg; default 100)")->check(CLI::Range(1u,1000000u));
	app.add_option("--logreg-tol",cfg.logreg_tol,"logistic regression gradient-norm stopping tolerance (with --criterion wrapper-logreg; default 1e-6)")->check(CLI::Range(1e-15,1.0));
	app.add_option("--knn-distance",cfg.knn_distance,"kNN distance; Lp uses p=3/2 (compile-time menu; default L2)")->check(CLI::IsMember({"L1","L2","Lp"}));
	app.add_option("--svm-kernel",cfg.svm_kernel,"LibSVM kernel (with --criterion wrapper-svm; default rbf - the most stable; linear/sigmoid may freeze LibSVM on some data)")->check(CLI::IsMember({"linear","rbf","poly","sigmoid"}));
	app.add_option("--svm-c",cfg.svm_c,"LibSVM C parameter (default: library default or --svm-optimize result)");
	app.add_option("--svm-gamma",cfg.svm_gamma,"LibSVM gamma parameter (default: library default or --svm-optimize result)");
	app.add_option("--svm-coef0",cfg.svm_coef0,"LibSVM coef0 parameter (default: library default or --svm-optimize result)");
	app.add_option("--svm-optimize",cfg.svm_optimize,"LibSVM C/gamma/coef0 grid optimization: pre = once on the full feature set before the search (and on the --search2 seed subset); alternating = additionally re-optimize and re-search while the criterion improves (default none)")
		->check(CLI::IsMember({"none","pre","alternating"}));
	app.add_option("--search2",cfg.search2,"seeding pre-search: bif ranks features individually and its best --search2-d features form the main search's initial subset (default: none)")->check(CLI::IsMember({"bif"}));
	app.add_option("--search2-d",cfg.search2_d,"seed subset size for --search2 (default 0 = use --target-d)");
	app.add_option("--search",cfg.search,"search algorithm: bif (individual ranking), sfs/sbs (sequential), sffs/sbfs (floating), sfrs/sbrs (retreating), os (oscillating), dos (dynamic oscillating), exhaustive, montecarlo, bb-basic/bb-improved/bb-fast/bb-bbpp (optimal Branch & Bound; monotonic filter criteria only); Dependency-Aware Feature ranking = montecarlo + --tracker feature-stats + --daf-rank")->required()
		->check(CLI::IsMember({"bif","sfs","sbs","sffs","sbfs","sfrs","sbrs","os","dos","exhaustive","montecarlo","bb-basic","bb-improved","bb-fast","bb-bbpp"}));
	app.add_option("--ridge-lambda",cfg.ridge_lambda,"ridge regression L2 weight penalty, bias unpenalized (with --criterion wrapper-ridge; default 0.01)")->check(CLI::Range(0.0,1e6));
	app.add_option("--regr-multiplier",cfg.regr_multiplier,"weighted-neighbours regression kernel multiplier (with --criterion wrapper-regressor; default: library default)")->check(CLI::Range(0.0,1000.0));
	app.add_option("--regr-multipliers",cfg.regr_multipliers,"comma-separated kernel multipliers: one full search per value, best-by-train-value wins (demo11reg protocol; with --criterion wrapper-regressor)");
	app.add_option("--evaluator",cfg.evaluator,"sequential-step candidate evaluator: straight (default), hybrid (a fast filter pre-orders candidates, only the best reach the main criterion), ensemble (kNN wrappers with different k vote), sampled (budgeted stochastic step - forward steps evaluate only --step-cap candidates proposed from online per-feature statistics; EXPERIMENTAL); sequential searches only; all accept --threads N with results identical to --threads 1 (bit-identical at matched --seed)")
		->check(CLI::IsMember({"straight","hybrid","ensemble","sampled"}));
	app.add_option("--hybrid-prefilter",cfg.hybrid_prefilter,"filter criterion used by the hybrid evaluator to pre-order candidates (bhattacharyya/divergence/gmahalanobis/multinom-bhattacharyya/multinom-mi; initialized on the whole depth-0 train part)");
	app.add_option("--hybrid-keep-perc",cfg.hybrid_keep_perc,"percentage of prefiltered candidates evaluated by the main criterion (with --evaluator hybrid; default 50)")->check(CLI::Range(1u,100u));
	app.add_option("--hybrid-keep-max",cfg.hybrid_keep_max,"hard cap on candidates evaluated by the main criterion (with --evaluator hybrid; default 1000)")->check(CLI::Range(1u,1000000u));
	app.add_option("--ensemble-knn-ks",cfg.ensemble_knn_ks,"comma-separated kNN k values forming the voting ensemble (with --evaluator ensemble, e.g. 1,5,9)");
	app.add_option("--step-cap",cfg.step_cap,"sampled evaluator: max candidate evaluations per forward step (with --evaluator sampled; default 100; 0 = no cap)");
	app.add_option("--step-cap-backward",cfg.step_cap_backward,"sampled evaluator: max candidate evaluations per BACKWARD step - removal candidates sampled ~ softmax(-score/tau) with the exploration floor over least-absence-observed members (default 0 = full backward sweeps = exact floating correction)");
	app.add_option("--step-cap-frac",cfg.step_cap_frac,"sampled evaluator: constant-PERCENTAGE budget schedule - the forward step cap becomes ceil(frac*pool) and --step-cap is ignored (0 = off; per-step cost then scales with dimensionality again - schedule ablation; forward steps only)")->check(CLI::Range(0.0,1.0));
	app.add_flag("--step-frozen",cfg.step_frozen,"sampled evaluator: freeze the per-feature statistics right after the warm-up probes - no in-search updates (frozen-ranking ablation; requires --warmup-probes)");
	app.add_option("--step-explore",cfg.step_explore,"sampled evaluator: fraction of each proposal batch drawn uniformly from the least-evaluated features - the exploration floor (with --evaluator sampled; default 0.2)")->check(CLI::Range(0.0,1.0));
	app.add_option("--step-tau",cfg.step_tau,"sampled evaluator: softmax temperature for weighted proposals (with --evaluator sampled; default 0 = robust automatic scale from the current score spread)");
	app.add_option("--step-decay",cfg.step_decay,"sampled evaluator: per-feature statistics averaging horizon in evaluations - exact mean up to this count, exponential forgetting beyond (with --evaluator sampled; default 100)")->check(CLI::Range(1u,1000000u));
	app.add_option("--step-sampler",cfg.step_sampler,"sampled evaluator: proposal policy - softmax (weighted + exploration floor, default), uniform (stochastic-greedy baseline), topk (hard cutoff ablation)")->check(CLI::IsMember({"softmax","uniform","topk"}));
	app.add_option("--step-stats",cfg.step_stats,"sampled evaluator: per-feature statistic - zcontrast (batch-standardized DAF contrast, default) or rawmean (size-weighted mean of the raw criterion values of subsets containing the feature, ablation)")->check(CLI::IsMember({"zcontrast","rawmean"}));
	app.add_option("--warmup-probes",cfg.warmup_probes,"sampled evaluator: number of random probe subsets evaluated before the search to seed the per-feature statistics (default 0 = off)");
	app.add_option("--warmup-card",cfg.warmup_card,"cardinality of the --warmup-probes probe subsets (default 25)")->check(CLI::Range(1u,1000000u));
	app.add_option("--restarts-nonimproving",cfg.restarts_nonimproving,"random-restart protocol (demo22): rerun the search from random --target-d-feature subsets until this many consecutive runs fail to improve (default 0 = off)");
	app.add_option("--restarts-max",cfg.restarts_max,"random-restart safety cap on total runs (default 0 = unlimited)");
	app.add_flag("--bias-estimate",cfg.bias_estimate,"estimate the wrapper classifier's bias (train-vs-test optimism) on the selected subset (demo56)");
	app.add_option("--classify-samples",cfg.classify_samples,"file with whitespace-separated full-length feature vectors: train on the depth-0 train part and print the predicted class per row (demo63); each row is transformed per feature exactly as the loaded data was (--scaler, --missing-value) before classification. NOTE: the printed row index counts the file's rows from 1 (file order), the one 1-based index in FST4 - features and classes are 0-based everywhere")->check(CLI::ExistingFile);
	app.set_config("--config","","read options from an INI/TOML-style file (keys = long option names without dashes, e.g. criterion=\"wrapper-knn\"); command-line values take precedence");
	app.add_option("--generalization-level",cfg.generalization,"number of features added/removed per sequential search step (g-tuples; sequential searches only; default 1)")->check(CLI::Range(1u,100u));
	app.add_option("--mc-trials",cfg.mc_trials,"Monte Carlo stopping condition: maximum number of random probes (with --search montecarlo; default 1000; 0 = unlimited, requires --mc-seconds)");
	app.add_option("--mc-seconds",cfg.mc_seconds,"Monte Carlo stopping condition: wall-clock budget in seconds (with --search montecarlo; default 0 = none; NOTE makes results load-dependent)");
	app.add_option("--mc-card-min",cfg.mc_card_min,"Monte Carlo probe subset cardinality range lower bound (with --search montecarlo; default: unrestricted)");
	app.add_option("--mc-card-max",cfg.mc_card_max,"Monte Carlo probe subset cardinality range upper bound (with --search montecarlo)");
	app.add_option("--mc-card-prob",cfg.mc_card_prob,"Monte Carlo per-feature inclusion probability, alternative to the min/max cardinality range (with --search montecarlo)")->check(CLI::Range(0.0,1.0));
	app.add_option("--bb-gamma",cfg.bb_gamma,"Fast Branch & Bound prediction optimism factor (with --search bb-fast; default 1.0)")->check(CLI::Range(0.0,1000.0));
	app.add_option("--bb-delta",cfg.bb_delta,"Fast Branch & Bound minimum prediction evidence count (with --search bb-fast; default 1)")->check(CLI::Range(1u,100000u));
	app.add_option("--target-d",cfg.target_d,"target subset size; 0 = optimize subset size (wrapper criteria only; monotonic filter criteria require a nonzero value)");
	app.add_option("--validate",cfg.validate,"final validation on the depth-0 holdout after the search: comma-separated classifier list (knn, bayes, lda, logreg, rf, linsvm, rbfsvm, multinom, svm, extern, or 'same' = the wrapper criterion's classifier; each takes its own --knn-*/--logreg-*/--rf-*/--linsvm-*/--rbfsvm-*/--svm-*/--extern-cmd settings), each trained on the outer train part and tested on the outer test part (default: none)");
	app.add_option("--tracker",cfg.tracker,"result tracker: dupless (best-k, default), regularizer (enables --regularize-margins), stabileval (per-trial stability statistics, requires --repeat-over-splits), feature-stats (probe statistics, enables --daf-rank and --daf-pairs)")
		->check(CLI::IsMember({"dupless","regularizer","stabileval","feature-stats"}));
	app.add_option("--tracker-attach",cfg.tracker_attach,"where the tracker records results: 'search' = accepted subsets only, 'evaluator' = every evaluated candidate (sequential searches only; default: search, except regularizer which defaults to evaluator)")
		->check(CLI::IsMember({"search","evaluator"}));
	app.add_option("--tracker-capacity",cfg.tracker_capacity,"maximum number of tracked results (default: --top-k for dupless, unlimited for the others)");
	app.add_option("--tracker-margin",cfg.tracker_margin,"inclusion margin: also track results whose value is within this distance of the running best (default 0)")->check(CLI::Range(0.0,1e9));
	app.add_flag("--repeat-over-splits",cfg.repeat_over_splits,"run the search once per outer (depth-0) split - the trial protocol used for stability evaluation (demo54); filter criteria re-estimate their model per trial, wrapper criteria need --splitter2 so that their own train/test estimation runs on inner splits of each trial's training part");
	app.add_option("--regularize-by",cfg.regularize_by,"secondary criterion for the post-search regularization sweep: subsetsize (prefer smaller subsets) or weights (prefer cheaper subsets per --feature-costs); requires --tracker regularizer")
		->check(CLI::IsMember({"subsetsize","weights"}));
	app.add_option("--regularize-margins",cfg.regularize_margins,"comma-separated criterion margins; for each, the best tracked subset within margin of the maximum according to the secondary criterion is reported (and validated if --validate is set)");
	app.add_option("--feature-costs",cfg.feature_costs,"comma-separated per-feature acquisition costs (one per feature; with --regularize-by weights)");
	app.add_option("--daf-rank",cfg.daf_rank,"extract the Dependency-Aware Feature ranking variant 0 or 1 from the feature-stats tracker (with --tracker feature-stats, typically --search montecarlo; default: off)")->check(CLI::Range(0,1));
	app.add_option("--daf-count",cfg.daf_count,"number of top-DAF features to report (default 0 = all)");
	app.add_option("--daf-validate-every",cfg.daf_validate_every,"validate incrementally built top-DAF subsets every K features using the first --validate classifier (default 0 = off)");
	app.add_option("--daf-pairs",cfg.daf_pairs,"pairwise DAF statistics: 2x2 presence contrasts (pair-DAF0/pair-DAF1/synergy) over all pairs of the top-K DAF0-ranked features (with --tracker feature-stats; default 0 = off)");
	app.add_option("--daf-pairs-top",cfg.daf_pairs_top,"number of top pairs (by pair-DAF0) to report (with --daf-pairs; default 50; 0 = all)");
	app.add_option("--sffs-delta",cfg.sffs_delta,"restrict the floating-search extent (with --search sffs/sbfs and a nonzero --target-d) in the search's OWN direction: sffs stops adding features at target-d+delta, sbfs stops removing them at target-d-delta, then the best tracked subset of the target size is reported; the floating backtracking steps stay bounded only by their improvement rule (default 0 = classic unrestricted floating over the full frontier - on very-high-dimensional data prefer a bound, e.g. 20)")->check(CLI::Range(0u,10000u));
	app.add_option("--os-delta",cfg.os_delta,"oscillation cycle depth (with --search os; default 1)")->check(CLI::Range(1u,10000u));
	app.add_option("--dos-delta",cfg.dos_delta,"oscillation cycle depth (with --search dos; default 3)")->check(CLI::Range(1u,10000u));
	app.add_option("--threads",cfg.threads,"number of worker threads for candidate evaluation; 1 = sequential evaluation, deterministic ordering (default 1)")->check(CLI::Range(1u,256u));
	app.add_option("--top-k",cfg.top_k,"number of best tracked subsets to report (default 5)")->check(CLI::Range(1u,100000u));
	app.add_option("--report",cfg.report,"result detail level (default standard = the complete result): detailed adds the research sections — curve_over_d (best-per-size criterion value + holdout re-validation of each, the overfitting curve), ROC/PR curves of the first --validate classifier on the depth-0 holdout (per-class one-vs-rest on multiclass data), confusion matrices, and the optimize protocols' tuned hyperparameters")
		->check(CLI::IsMember({"standard","detailed"}));
	app.add_option("--dump-scores",cfg.dump_scores,"write the final subset's per-sample predictions on the depth-0 holdout to this file (CSV with a header: sample index, true class, predicted class, and one score column per class when the classifier exposes scores) - feeds external ROC variants, calibration and error analysis. Uses the FIRST classifier named by --validate, adding no training; the row order is the same class-major enumeration the ROC/PR curves are built from");
	app.add_option("--curves-svg",cfg.curves_svg,"render the --report detailed curves into this file as a self-contained SVG (no plotting dependency, no external resources): the ROC and PR curves of the validated classifier and the criterion-vs-holdout curve over subset sizes, the latter with separate left/right axes so the two quantities are never conflated. Requires --report detailed, which computes the curves");
	app.add_option("--output",cfg.output,"result format on stdout (default text); with json/csv, search progress goes to stderr")->check(CLI::IsMember({"text","json","csv"}));
	app.add_option("--verbosity",cfg.verbosity,"search progress verbosity (default normal)")->check(CLI::IsMember({"silent","normal","detailed"}));
	app.add_option("--progress",cfg.progress_mode,"search progress delivery (default text): json = one JSON line per improved solution on the progress stream (needs --verbosity normal or higher), quiet = suppress search progress")->check(CLI::IsMember({"text","json","quiet"}));
	app.add_flag("--stable-output",cfg.stable_output,"omit non-deterministic output (timings) so results are byte-comparable across runs");
	app.add_option("--seed",cfg.seed,"random seed; affects randomized components only, e.g. the initial subset of --search os (default 1)");
	app.add_option("--compare-config",cfg.compare_config,"config file (same keys as --config) defining a SECOND feature-selection scenario on the same --data: both run in sequence and the cross-scenario tracker similarity (IATI/ICW/IANHI) is reported; both scenarios need --tracker stabileval --repeat-over-splits (demo55 protocol)")->check(CLI::ExistingFile);
}

//! parses a `--compare-config` file into a fresh Config through the full add_options() set --
//! the file must be self-contained (it supplies `--data`, `--criterion`, `--search`, ... itself,
//! since this parses a synthetic one-token "fst4" command line with no other arguments) --
//! and applies the same wrapper-ridge default-metric normalization main() applies to the
//! primary scenario, so both scenarios of a `--compare-config` run are treated identically
//! \return the second scenario's configuration; throws on a missing/malformed config file
//!         (app2.parse()'s CLI11 validation errors propagate as exceptions)
static Config parse_compare_config(const std::string &path)
{
	Config cfg2;
	CLI::App app2{"fst4 --compare-config scenario"};
	add_options(app2,cfg2);
	app2.set_config("--config",path)->required(); // force-read the file through the standard config machinery
	const char *argv2[]={"fst4"};
	app2.parse(1,const_cast<char**>(argv2));
	if(cfg2.criterion=="wrapper-ridge" && cfg2.wrapper_metric.empty()) cfg2.wrapper_metric="r2"; // same normalization as the primary scenario
	return cfg2;
}

} // namespace fst4_cli

/*! \brief fst4 CLI entry point.

    Parses the command line (add_options() via CLI11; `--list` short-circuits before parsing
    proper, since it needs no `--data`/`--criterion`/`--search`), then:
    \li redirects std::cout to stderr's buffer (or to a null stream under `--verbosity` silent)
        whenever `--output` is json or csv, keeping a separate `real_stdout` handle to the
        original stdout buffer -- this is the pipe-clean mechanism: ALL search/library prose
        goes through the (possibly redirected) std::cout/progress stream, while the final
        result is written to `real_stdout` only, so machine-format stdout carries nothing else;
    \li seeds FST::fst_srand() exactly once for the whole process (both scenarios of a
        `--compare-config` run share one continuous fst_rand() stream, per the demo55 protocol);
    \li runs the scenario through the locally-defined `run_pipeline` lambda (build accessor/
        criterion/evaluator/search/tracker, run the selected protocol, post-process), and, under
        `--compare-config`, a second scenario parsed by parse_compare_config() plus the
        cross-scenario tracker similarity (IATI/ICW/IANHI);
    \li dispatches the result to emit_json()/emit_csv()/emit_text() per `--output`, and, on an
        FST::fst_error, prints "FST ERROR: <message>, code=<code>" to stderr (with a singular-
        covariance hint naming the offending constant features when applicable) or, on any
        other std::exception, "non-FST ERROR: <what()>" to stderr.

    \return 0 on success, 1 if an exception was caught (FST::fst_error or std::exception) */
int main(int argc, char *argv[])
{
	using namespace fst4_cli;
	for(int i=1;i<argc;i++) if(std::string(argv[i])=="--list") {print_components(std::cout); return 0;}

	Config cfg;
	CLI::App app{"fst4 - Feature Selection Toolbox 4 command-line tool (reference: doc/FST4_CLI.md)"};
	add_options(app,cfg);
	CLI11_PARSE(app,argc,argv);
	// wrapper-ridge has no meaningful "plain accuracy": its natural criterion is R^2
	// — default the metric unless the user chose one
	if(cfg.criterion=="wrapper-ridge" && cfg.wrapper_metric.empty()) cfg.wrapper_metric="r2";

	// keep a handle on real stdout: with `--output` json/csv all library/search
	// prose is redirected to stderr (or swallowed when silent) so that stdout
	// carries nothing but the machine-readable result
	std::streambuf *stdout_buf=std::cout.rdbuf();
	std::ostream real_stdout(stdout_buf);
	static NullBuf nullbuf;
	std::ostream nullstream(&nullbuf);
	const bool silent=(cfg.verbosity=="silent");
	const bool machine_output=(cfg.output!="text");
	if(silent) std::cout.rdbuf(&nullbuf);
	else if(machine_output) std::cout.rdbuf(std::cerr.rdbuf());
	std::ostream &progress = silent ? nullstream : (machine_output ? std::cerr : std::cout);

	// the complete validate->build->search->post-process phase for ONE scenario, factored
	// so that `--compare-config` can run a second scenario through identical machinery.
	// FST::fst_srand is seeded by the caller ONCE: the demo55 protocol runs both scenarios on
	// one continuous fst_rand() stream, never reseeding in between.
	auto run_pipeline = [&](const Config &cfg, Pipeline &p) -> RunResult {
		// data-free entry (`--features`): format "none" — there is no file whose format could matter
		const std::string format = cfg.data_free() ? "none" : (cfg.format.empty() ? infer_format(cfg.data) : cfg.format);

		// validity rules
		if(cfg.data_free()) {
			// the universe is abstract: refuse everything whose meaning depends on data existing.
			// The searches themselves consume only n and criterion values, so they all remain
			// available (as do trackers, DAF statistics, `--threads`, `--search2` bif, the sampled
			// evaluator, and cost-based regularization — feature costs are abstract too)
			if(cfg.criterion!="extern-value" || !cfg.extern_no_data)
				throw FST::fst_error("--features N optimizes over an abstract data-free universe; the only criterion that can score its subsets is an external set function - add --criterion extern-value --extern-no-data --extern-cmd \"<command>\" (every built-in criterion computes from data)");
			if(!cfg.format.empty())
				throw FST::fst_error("--format names a data-file format; with --features there is no data file");
			if(cfg.splitter!="5050" || cfg.nested_splits())
				throw FST::fst_error("--splitter/--splitter2 partition data samples; with --features there are none (the universe pipeline runs one trivial split)");
			if(cfg.scaler!="void" || cfg.scaler_fit!="all" || cfg.has_missing_value())
				throw FST::fst_error("--scaler/--scaler-fit/--missing-value transform data values; with --features there are none");
			if(!cfg.validate.empty())
				throw FST::fst_error("--validate trains classifiers on data; with --features there is none - the external tool's value is the only quality measure");
			if(cfg.repeat_over_splits)
				throw FST::fst_error("--repeat-over-splits runs one search per data split; with --features there is only the one trivial split");
			if(!cfg.compare_config.empty())
				throw FST::fst_error("--compare-config compares scenarios over the same --data file; the data-free entry has none");
			if(cfg.bias_estimate || !cfg.classify_samples.empty())
				throw FST::fst_error("--bias-estimate/--classify-samples need data and classifiers; with --features there are none");
			if(cfg.detailed_report() || !cfg.dump_scores.empty() || !cfg.curves_svg.empty())
				throw FST::fst_error("--report detailed/--dump-scores/--curves-svg build on holdout validation; with --features there is no data to hold out - use --report standard");
			if(cfg.evaluator=="hybrid" || cfg.evaluator=="ensemble")
				throw FST::fst_error("the hybrid and ensemble evaluators build on data-computed criteria (filter prefilter, kNN members); with --features use --evaluator straight or sampled");
		}
		if(format!="csv" && (!cfg.csv_class.empty() || cfg.csv_delimiter!="auto" || cfg.csv_header!="auto" || cfg.has_csv_missing_to()))
			throw FST::fst_error("--csv-* options apply to the csv format only (data format resolved as '"+format+"')");
		if(cfg.has_csv_missing_to()) {
			// the mapped code enters the data as an ordinary value; only the void scaler leaves it
			// intact for `--missing-value` to substitute (to01/white would fold it into min/max or
			// mean/stddev), and a differing `--missing-value` code would never match the mapped one
			if(cfg.scaler!="void")
				throw FST::fst_error("--csv-missing-to would fold the missing-value code into the "+cfg.scaler+" scaling statistics; it works with --scaler void only (pair with --missing-value to substitute per-feature training averages)");
			if(cfg.has_missing_value() && cfg.missing_value!=cfg.csv_missing_to)
				throw FST::fst_error("--csv-missing-to "+fmt_double(cfg.csv_missing_to)+" and --missing-value "+fmt_double(cfg.missing_value)+" disagree - the mapped code would never be substituted; use the same value for both");
		}
		if((cfg.is_filter_criterion() || (cfg.criterion=="extern-value" && cfg.extern_monotonic)) && cfg.target_d==0)
			throw FST::fst_error("criterion '"+cfg.criterion+"' is monotonic with respect to subset size (subset-size optimization would select the full feature set); set --target-d to a nonzero value");
		if(cfg.search=="os" && cfg.target_d==0)
			throw FST::fst_error("--search os requires an initial subset of target cardinality; set --target-d to a nonzero value");
		if(cfg.is_bb_search()) {
			if(!cfg.is_filter_criterion() && !(cfg.criterion=="extern-value" && cfg.extern_monotonic))
				throw FST::fst_error("Branch & Bound guarantees optimality only for criteria monotonic with respect to subset size; '"+cfg.criterion+"' is not - use a normal-model filter criterion (bhattacharyya recommended; divergence is monotonic in theory but numerically unstable), a non-optimal search (exhaustive works with any criterion), or an external VALUE-mode measure asserted monotonic (--criterion extern-value --extern-monotonic; optimality then rests on the assertion)");
			if(cfg.target_d==0)
				throw FST::fst_error("Branch & Bound searches a fixed target cardinality; set --target-d to a nonzero value");
		}
		if(cfg.search=="montecarlo") {
			if(cfg.mc_card_prob>0.0 && cfg.mc_card_min>0)
				throw FST::fst_error("--mc-card-prob and --mc-card-min/--mc-card-max are mutually exclusive Monte Carlo cardinality randomization modes");
			if(cfg.mc_card_min>0 && cfg.mc_card_max<cfg.mc_card_min)
				throw FST::fst_error("--mc-card-max must be >= --mc-card-min");
			if(cfg.mc_trials==0 && cfg.mc_seconds==0)
				throw FST::fst_error("Monte Carlo needs a stopping condition: set --mc-trials and/or --mc-seconds to a nonzero value");
		}
		if(cfg.generalization>1 && !cfg.is_sequential_search())
			throw FST::fst_error("--generalization-level applies to the sequential search family (sfs/sbs/sffs/sbfs/sfrs/sbrs/os/dos) only");
		if(cfg.tracker_attach=="evaluator" && !cfg.is_sequential_search())
			throw FST::fst_error("--tracker-attach evaluator requires a sequential-family search (only those use a candidate-evaluator step object); "+cfg.search+" tracks at the search level");
		if(cfg.tracker=="regularizer" && !cfg.is_sequential_search() && cfg.tracker_attach!="search")
			throw FST::fst_error("--tracker regularizer records every evaluated candidate through the sequential-search evaluator; with --search "+cfg.search+" pass --tracker-attach search explicitly if search-level tracking is intended");
		if(cfg.tracker=="stabileval" && !cfg.repeat_over_splits)
			throw FST::fst_error("--tracker stabileval computes stability statistics over repeated trials; combine it with --repeat-over-splits (and an outer splitter yielding multiple splits, e.g. --splitter randrand --rr-splits 10; a wrapper criterion additionally needs --splitter2, e.g. --splitter2 cv)");
		if(cfg.repeat_over_splits && !cfg.validate.empty())
			throw FST::fst_error("--validate is not available with --repeat-over-splits: the trial protocol consumes the outer splits (per-trial criterion values serve as the accuracy estimate)");
		// A wrapper's evaluate() iterates the splits of the depth it runs at. Without --splitter2
		// that is depth 0 - the very cursor the trial loop walks - so every evaluation would
		// exhaust and reset the trial enumeration (randrand/randfix: the run never ends; cv:
		// a single trial) and each trial value would be a mean over ALL outer splits anyway.
		// The trial protocol therefore needs the wrapper's own estimation nested one level down.
		if(cfg.repeat_over_splits && !cfg.nested_splits() && (cfg.is_wrapper_criterion() || cfg.evaluator=="ensemble"))
			throw FST::fst_error("--repeat-over-splits with a wrapper criterion needs --splitter2: the outer (depth-0) splits are the trials, so the classifier's own train/test estimation must run on inner (depth-1) splits of each trial's training part - e.g. --splitter randrand --rr-splits 10 --splitter2 cv (the demo54 protocol). Without --splitter2 the wrapper's split loop and the trial loop would walk the same splits and the run would never finish");
		if(!cfg.dump_scores.empty() && cfg.validate.empty())
			throw FST::fst_error("--dump-scores writes the holdout predictions of a validation classifier; add --validate to name one (e.g. --validate same with a wrapper criterion)");
		// A regression pipeline validates only with its own regressor (the rule below refuses
		// classification validators outright), so no classifier is ever trained for the dump
		// to use — refuse here rather than silently writing nothing
		if(!cfg.dump_scores.empty() && cfg.is_regression_criterion())
			throw FST::fst_error("--dump-scores writes per-sample CLASS predictions and per-class scores; --criterion "+cfg.criterion
			                     +" predicts a continuous target and is validated by the regressor itself, so there are no classes to dump");
		if(!cfg.curves_svg.empty() && !cfg.detailed_report())
			throw FST::fst_error("--curves-svg renders the curves that --report detailed computes; add --report detailed");
		if(!cfg.curves_svg.empty() && (cfg.repeat_over_splits || !cfg.regr_multipliers.empty()))
			throw FST::fst_error("--curves-svg needs a single-search trajectory; the trial and sweep protocols (--repeat-over-splits, --regr-multipliers) have no single curve to plot");
		if(!cfg.regularize_margins.empty() && cfg.regularize_by.empty())
			throw FST::fst_error("--regularize-margins requires --regularize-by subsetsize|weights");
		if(!cfg.regularize_by.empty() && cfg.regularize_margins.empty())
			throw FST::fst_error("--regularize-by requires --regularize-margins (comma-separated margin list)");
		if(!cfg.regularize_by.empty() && cfg.tracker!="regularizer")
			throw FST::fst_error("the regularization sweep needs the tracked candidate list; add --tracker regularizer");
		if(cfg.regularize_by=="weights" && cfg.feature_costs.empty())
			throw FST::fst_error("--regularize-by weights requires --feature-costs (comma-separated per-feature costs)");
		if(cfg.daf_rank>=0 && cfg.tracker!="feature-stats")
			throw FST::fst_error("--daf-rank extracts the DAF ranking from the feature-stats tracker; add --tracker feature-stats (typically with --search montecarlo, the DAF ranking protocol)");
		if(cfg.daf_validate_every>0 && cfg.daf_rank<0)
			throw FST::fst_error("--daf-validate-every requires --daf-rank 0|1");
		if(cfg.daf_validate_every>0 && cfg.validate.empty())
			throw FST::fst_error("--daf-validate-every requires --validate to name the validation classifier");
		if(cfg.sffs_delta>0 && cfg.search!="sffs" && cfg.search!="sbfs")
			throw FST::fst_error("--sffs-delta bounds the floating search; it requires --search sffs or sbfs");
		if(cfg.sffs_delta>0 && cfg.target_d==0)
			throw FST::fst_error("--sffs-delta bounds the frontier around the target size; set --target-d to a nonzero value (with --target-d 0 the floating search always runs the full frontier)");
		if(cfg.daf_pairs==1)
			throw FST::fst_error("--daf-pairs needs at least 2 features to form pairs (or 0 = off)");
		if(cfg.daf_pairs>0 && cfg.tracker!="feature-stats")
			throw FST::fst_error("--daf-pairs computes pairwise statistics from the feature-stats tracker; add --tracker feature-stats (typically with --search montecarlo, the DAF protocol)");
		if(cfg.svm_optimize!="none" && cfg.criterion!="wrapper-svm")
			throw FST::fst_error("--svm-optimize tunes the wrapper criterion's SVM; it requires --criterion wrapper-svm");
		if(cfg.rbfsvm_optimize!="none" && cfg.criterion!="wrapper-rbfsvm")
			throw FST::fst_error("--rbfsvm-optimize tunes the wrapper criterion's native RBF SVM; it requires --criterion wrapper-rbfsvm");
		if((!cfg.wrapper_metric.empty() || !cfg.class_weights.empty()) && !cfg.is_wrapper_criterion())
			throw FST::fst_error("--wrapper-metric/--class-weights configure wrapper criteria (classifier or regressor based); filter criteria have no classifier - use a wrapper-* criterion");
		if(cfg.f1_positive!=1 && cfg.wrapper_metric!="f1")
			throw FST::fst_error("--f1-positive selects which class the f1 metric scores as positive; it requires --wrapper-metric f1 (f1-macro averages over all classes, and auc keeps class 1 positive by its own convention)");
		if((cfg.criterion=="wrapper-extern" || cfg.criterion=="extern-value") && cfg.extern_cmd.empty())
			throw FST::fst_error("--criterion "+cfg.criterion+" delegates computation to an external command; set --extern-cmd (reference implementations: tests/extern_centroid.awk for wrapper-extern, tests/extern_value_meandiff.awk for extern-value)");
		if(!cfg.extern_cmd.empty() && cfg.criterion!="wrapper-extern" && cfg.criterion!="extern-value")
			throw FST::fst_error("--extern-cmd configures the external-tool bridges; it requires --criterion wrapper-extern or extern-value");
		if(cfg.extern_monotonic && cfg.criterion!="extern-value")
			throw FST::fst_error("--extern-monotonic asserts the monotonicity of the external VALUE-mode measure; it requires --criterion extern-value");
		if(cfg.extern_no_data && cfg.criterion!="extern-value")
			throw FST::fst_error("--extern-no-data suppresses the extern-value DATA block; it requires --criterion extern-value");
		if(cfg.criterion=="wrapper-extern" && cfg.wrapper_metric=="auc")
			throw FST::fst_error("--wrapper-metric auc needs per-sample scores; the wrapper-extern protocol exchanges labels only (confusion-based metrics like balanced-accuracy or f1 work)");
		if(cfg.is_regression_criterion()) {
			if(!cfg.wrapper_metric.empty() && cfg.wrapper_metric!="accuracy" && cfg.wrapper_metric!="r2" && cfg.wrapper_metric!="neg-rmse" && cfg.wrapper_metric!="neg-mae")
				throw FST::fst_error("--criterion "+cfg.criterion+" predicts a continuous target; use a regression metric (r2, neg-rmse, neg-mae), not '"+cfg.wrapper_metric+"'");
			if(!cfg.class_weights.empty())
				throw FST::fst_error("--class-weights weighs classification confusion matrices; it does not apply to the regression pipeline");
		} else if(cfg.wrapper_metric=="r2" || cfg.wrapper_metric=="neg-rmse" || cfg.wrapper_metric=="neg-mae")
			throw FST::fst_error("--wrapper-metric "+cfg.wrapper_metric+" is a regression metric; it requires a regression wrapper criterion (wrapper-regressor or wrapper-ridge)");
		if(!cfg.class_weights.empty() && cfg.wrapper_metric=="balanced-accuracy")
			throw FST::fst_error("--class-weights with balanced-accuracy would reweight twice (balanced accuracy already weighs classes equally); pick one");
		if(!cfg.class_weights.empty() && cfg.wrapper_metric=="auc")
			throw FST::fst_error("--class-weights does not combine with auc (weighted AUC has several inequivalent definitions); drop one of the two flags");
		if(!cfg.regr_multipliers.empty() && cfg.criterion!="wrapper-regressor")
			throw FST::fst_error("--regr-multipliers sweeps the weighted-neighbours regression kernel; it requires --criterion wrapper-regressor");
		if(!cfg.regr_multipliers.empty() && cfg.repeat_over_splits)
			throw FST::fst_error("--regr-multipliers and --repeat-over-splits are mutually exclusive run protocols");
		if(cfg.is_regression_criterion() && !cfg.validate.empty()) {
			std::vector<std::string> vnames=split_csv(cfg.validate);
			for(std::vector<std::string>::const_iterator it=vnames.begin();it!=vnames.end();++it)
				if(*it!="same")
					throw FST::fst_error("--criterion "+cfg.criterion+" predicts a continuous target; classification validators do not apply - use --validate same (the regressor itself)");
		}
		if(cfg.evaluator!="straight") {
			if(!cfg.is_sequential_search())
				throw FST::fst_error("--evaluator "+cfg.evaluator+" is a sequential-step object; it requires a sequential-family search (sfs/sbs/sffs/sbfs/sfrs/sbrs/os/dos)");
			// all step evaluators are thread-capable: straight/hybrid/ensemble/sampled all
			// accept `--threads` N through the injected candidate-evaluation engine
		}
		if(cfg.evaluator=="hybrid" && cfg.hybrid_prefilter.empty())
			throw FST::fst_error("--evaluator hybrid requires --hybrid-prefilter (the fast filter criterion that pre-orders candidates)");
		if(cfg.evaluator=="ensemble" && cfg.ensemble_knn_ks.empty())
			throw FST::fst_error("--evaluator ensemble requires --ensemble-knn-ks (e.g. 1,5,9)");
		if(cfg.evaluator=="sampled" && cfg.generalization>1)
			throw FST::fst_error("--evaluator sampled implements g=1 steps only (g>1 would fall back to full sweeps and defeat the budget); drop --generalization-level");
		if(cfg.warmup_probes>0 && cfg.evaluator!="sampled")
			throw FST::fst_error("--warmup-probes seeds the sampled evaluator's statistics; it requires --evaluator sampled");
		if(cfg.warmup_probes>0 && cfg.repeat_over_splits)
			throw FST::fst_error("--warmup-probes with --repeat-over-splits is not supported: warm-up statistics would leak across trials");
		if(cfg.step_cap_frac>0.0 && cfg.evaluator!="sampled")
			throw FST::fst_error("--step-cap-frac configures the sampled evaluator; it requires --evaluator sampled");
		if(cfg.step_cap_backward>0 && cfg.evaluator!="sampled")
			throw FST::fst_error("--step-cap-backward configures the sampled evaluator; it requires --evaluator sampled");
		if(cfg.step_frozen && cfg.warmup_probes==0)
			throw FST::fst_error("--step-frozen freezes warm-started statistics; it requires --warmup-probes (frozen empty statistics would reduce to uniform sampling)");
		if(cfg.restarts_nonimproving>0) {
			if(cfg.target_d==0)
				throw FST::fst_error("--restarts-nonimproving draws random subsets of a fixed size; set --target-d to a nonzero value");
			if(cfg.repeat_over_splits || !cfg.regr_multipliers.empty())
				throw FST::fst_error("--restarts-nonimproving is mutually exclusive with the other run protocols (--repeat-over-splits, --regr-multipliers)");
		}
		if(cfg.bias_estimate && !cfg.is_wrapper_criterion())
			throw FST::fst_error("--bias-estimate measures the wrapper classifier's optimism; it requires a wrapper-* criterion");
		if(cfg.bias_estimate && cfg.is_regression_criterion())
			throw FST::fst_error("--bias-estimate applies to classification wrappers, not the regression pipeline");
		if(cfg.bias_estimate && cfg.repeat_over_splits)
			throw FST::fst_error("--bias-estimate is not available with --repeat-over-splits");
		if(!cfg.search2.empty()) {
			if(cfg.search2_d==0 && cfg.target_d==0)
				throw FST::fst_error("--search2 needs the seed subset size: set --search2-d (or a nonzero --target-d)");
			if(cfg.search=="os" && cfg.search2_d!=0 && cfg.search2_d!=cfg.target_d)
				throw FST::fst_error("--search os resumes from a subset of exactly --target-d features; --search2-d must match --target-d (or be left at 0)");
			// the non-oscillating sequential searches start FROM the seed and move in their own
			// direction only, so the seed must leave them room: a forward search extends it up to
			// --target-d (the floating ones may also start at the target size and float around it),
			// a backward one shrinks it down to --target-d
			{
				const unsigned int d2 = cfg.search2_d ? cfg.search2_d : cfg.target_d;
				if(cfg.target_d>0) {
					if(cfg.search=="sfs" && d2>=cfg.target_d)
						throw FST::fst_error("--search sfs extends the --search2 seed up to --target-d: set --search2-d smaller than --target-d (the default seed size equals --target-d, which leaves sfs nothing to do)");
					if(cfg.search=="sbs" && d2<=cfg.target_d)
						throw FST::fst_error("--search sbs shrinks the --search2 seed down to --target-d: set --search2-d larger than --target-d");
					if((cfg.search=="sffs" || cfg.search=="sfrs") && d2>cfg.target_d)
						throw FST::fst_error("--search "+cfg.search+" floats forward from the --search2 seed: --search2-d must not exceed --target-d");
					if((cfg.search=="sbfs" || cfg.search=="sbrs") && d2<cfg.target_d)
						throw FST::fst_error("--search "+cfg.search+" floats backward from the --search2 seed: --search2-d must not be below --target-d");
				}
			}
			if(cfg.repeat_over_splits)
				throw FST::fst_error("--search2 is not available with --repeat-over-splits (the seeding pre-search runs once, not per trial)");
		}
		if(!cfg.validate.empty()) {
			std::vector<std::string> vnames=split_csv(cfg.validate);
			for(std::vector<std::string>::const_iterator it=vnames.begin();it!=vnames.end();++it) {
				if(*it=="same") {
					if(!cfg.is_wrapper_criterion())
						throw FST::fst_error("--validate same requires a wrapper criterion (filter criteria have no classifier); name a classifier instead (knn, bayes, lda, logreg, rf, linsvm, rbfsvm, multinom, svm, extern)");
				}
				else if(*it!="knn" && *it!="bayes" && *it!="lda" && *it!="logreg" && *it!="rf" && *it!="linsvm" && *it!="rbfsvm" && *it!="multinom" && *it!="svm" && *it!="extern")
					throw FST::fst_error("--validate: unknown classifier '"+*it+"' (available: knn, bayes, lda, logreg, rf, linsvm, rbfsvm, multinom, svm, extern, same)");
				else if(*it=="extern" && cfg.extern_cmd.empty())
					throw FST::fst_error("--validate extern runs the external classifier bridge; name its command with --extern-cmd");
			}
		}
		if(cfg.splitter=="leave1out" && !cfg.validate.empty())
			throw FST::fst_error("--validate needs a depth-0 holdout with a meaningful test part; --splitter leave1out leaves a single sample for testing per split - use 5050/holdout/randrand as the outer splitter (leave1out is intended as --splitter2)");

		p.da_mem=make_accessor(cfg,format);
		p.da_mem->initialize(); // reads the data file (concrete-side call)
		p.da=p.da_mem;
		p.da_mem->setSplittingDepth(0);
		if(!p.da->getFirstSplit()) throw FST::fst_error("initial data split failed");
		if(cfg.nested_splits()) { // demo choreography: criterion protocol runs on the inner (depth 1) splits
			p.da_mem->setSplittingDepth(1);
			if(!p.da->getFirstSplit()) throw FST::fst_error("initial inner data split failed");
		}
		const int eval_depth = cfg.nested_splits() ? 1 : 0;

		const DIMTYPE n=p.da->getNoOfFeatures();
		if(cfg.target_d>=n)
			throw FST::fst_error("--target-d must be smaller than the number of features ("+std::to_string((long long)n)+")");

		std::shared_ptr<SUBSET> sub(new SUBSET(n));
		sub->deselect_all();

		std::shared_ptr<CRITERION> crit_inner=make_criterion(cfg,p);
		p.evals.reset(new std::atomic<long long>(0));
		p.crit.reset(new Criterion_Counting(crit_inner,p.evals));

#ifdef FST_SVM
		// LibSVM hyperparameter grid optimization on the full feature set before the
		// search (demo12t/23/25t choreography); 'alternating' additionally re-optimizes
		// interleaved with repeated searches after the first search completes
		if(cfg.svm_optimize!="none") {
			sub->select_all();
			if(!p.svm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,-2,5,progress))
				throw FST::fst_error("--svm-optimize: SVM hyperparameter optimization failed");
			sub->deselect_all();
		}
#endif
		// native RBF SVM C/gamma optimization: the same choreography without LibSVM
		if(cfg.rbfsvm_optimize!="none") {
			sub->select_all();
			if(!p.rbfsvm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,progress))
				throw FST::fst_error("--rbfsvm-optimize: RBF SVM hyperparameter optimization failed");
			sub->deselect_all();
		}

		// two-stage protocol (demo25t/33): BIF individual ranking seeds the main search
		if(!cfg.search2.empty()) {
			const DIMTYPE d2 = cfg.search2_d ? (DIMTYPE)cfg.search2_d : (DIMTYPE)cfg.target_d;
			FST::Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> bif;
			bif.set_output_detail(parse_verbosity(cfg.verbosity));
			bif.set_observer(make_observer(cfg,progress));
			RETURNTYPE seedval;
			if(!bif.search(d2,seedval,sub,p.crit,progress)) throw FST::fst_error("--search2 bif pre-search not finished");
			if(!silent) {
				std::ostringstream sos;
				sos << "Seed subset from --search2 bif (d=" << d2 << ", value=" << seedval << "): " << *sub << std::endl << std::endl;
				FST::syncout::print(progress,sos);
			}
#ifdef FST_SVM
			if(cfg.svm_optimize!="none") // demo25t: re-optimize SVM parameters on the seed subset
				if(!p.svm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,-2,5,progress))
					throw FST::fst_error("--svm-optimize: SVM hyperparameter re-optimization failed");
#endif
			if(cfg.rbfsvm_optimize!="none") // the native analogue on the seed subset
				if(!p.rbfsvm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,progress))
					throw FST::fst_error("--rbfsvm-optimize: RBF SVM hyperparameter re-optimization failed");
		}
		else if(cfg.search=="os") sub->make_random_subset(cfg.target_d);

		std::shared_ptr<EVALUATOR> eval=make_evaluator(cfg,p,eval_depth);
		p.search=make_search(cfg,p,eval);
		make_tracker(cfg,p);
		if(p.tracker) { // stabileval is fed manually per trial, never attached
			const std::string attach = !cfg.tracker_attach.empty() ? cfg.tracker_attach
			                           : (cfg.tracker=="regularizer" ? "evaluator" : "search"); // regularizer records every candidate (demo51/52t/61)
			if(attach=="evaluator") eval->enable_result_tracking(p.tracker);
			else p.search->enable_result_tracking(p.tracker);
		}
		p.search->set_output_detail(parse_verbosity(cfg.verbosity));
		// `--report` detailed: interpose the best-per-size recorder (curve_over_d source) in
		// front of the `--progress` observer; single-search protocols only (the trial and
		// sweep protocols have no single trajectory to plot)
		std::shared_ptr<Observer_Curve> curve_rec;
		{
			std::shared_ptr<FST::Search_Observer<RETURNTYPE,SUBSET> > obs=make_observer(cfg,progress);
			if(cfg.detailed_report() && !cfg.repeat_over_splits && cfg.regr_multipliers.empty()) {
				curve_rec.reset(new Observer_Curve(obs,progress));
				obs=curve_rec;
				// solution announcements are gated by the search's output detail; the recorder
				// needs them even under `--verbosity` silent — force NORMAL, the prose still goes
				// to the silent nullstream so nothing visible changes
				if(parse_verbosity(cfg.verbosity)==FST::SILENT) p.search->set_output_detail(FST::NORMAL);
			}
			p.search->set_observer(obs);
		}

		// sampled-evaluator warm start: random probes of
		// fixed cardinality seed the per-feature statistics before the search; probes are
		// evaluated through the counting decorator, so reported eval totals include them
		if(cfg.warmup_probes>0) {
			if(!p.sampled_stats) throw FST::fst_error("--warmup-probes requires --evaluator sampled");
			const DIMTYPE r=(DIMTYPE)(cfg.warmup_card<n?cfg.warmup_card:n-1);
			std::shared_ptr<SUBSET> probe(new SUBSET(n));
			p.sampled_stats->reset(n);
			if(cfg.threads>1 && p.sampled_engine) {
				// threaded warm-up through the sampled evaluator's own engine (clones made
				// here serve the search afterwards). Same determinism argument as the step:
				// probe generation stays on the master thread's RNG stream in unchanged
				// order (add_candidate copies the probe synchronously), evaluations fan
				// out, and the callback folds results into the statistics in add-order at
				// batch completion - identical to the sequential loop below. Probes
				// are fed in chunks to bound the engine's per-batch result storage; the
				// statistics buffer all of them and flush once, exactly as below.
				// The FIRST probe is evaluated on the master criterion in-line (as the
				// sequential loop does) so the master's lazily initialized state - and
				// thereby its setup printout - follows the identical trajectory.
				std::shared_ptr<WEIGHTER_SAMPLED> stats=p.sampled_stats;
				RETURNTYPE pv;
				probe->make_random_subset(r);
				if(!p.crit->evaluate(pv,probe)) throw FST::fst_error("warm-up probe evaluation failed");
				stats->add(pv,probe);
				const unsigned long chunk=1024;
				unsigned long remaining=cfg.warmup_probes-1;
				while(remaining>0) {
					const unsigned long batch = remaining<chunk ? remaining : chunk;
					p.sampled_engine->initialize(p.crit,(DIMTYPE)batch,
						[stats](const RETURNTYPE val, const std::shared_ptr<SUBSET> &csub, const DIMTYPE) {stats->add(val,csub);});
					for(unsigned long i=0;i<batch;i++) {
						probe->make_random_subset(r);
						if(!p.sampled_engine->add_candidate(probe)) throw FST::fst_error("warm-up probe evaluation failed");
					}
					if(!p.sampled_engine->collect()) throw FST::fst_error("warm-up probe evaluation failed");
					remaining-=batch;
				}
			} else {
				RETURNTYPE pv;
				for(unsigned long i=0;i<cfg.warmup_probes;i++) {
					probe->make_random_subset(r);
					if(!p.crit->evaluate(pv,probe)) throw FST::fst_error("warm-up probe evaluation failed");
					p.sampled_stats->add(pv,probe);
				}
			}
			p.sampled_stats->flush_batch();
			if(cfg.step_frozen) p.sampled_stats->freeze(); // frozen-ranking ablation: no in-search updates
			if(!silent) {
				std::ostringstream sos;
				sos << "Warm start: " << cfg.warmup_probes << " probes @ cardinality " << r << " folded into the sampled-step statistics"
				    << (cfg.step_frozen ? " (statistics FROZEN for the search)" : "") << "." << std::endl << std::endl;
				FST::syncout::print(progress,sos);
			}
		}

		if(!silent) {
			std::ostringstream sos;
			sos << "fst4 (FST4 " << FST4_VERSION << ") feature selection setup:" << std::endl
			    << *p.da << std::endl << *p.search << std::endl << *p.crit << std::endl << std::endl;
			FST::syncout::print(progress,sos);
		}

		RunResult r;
		RETURNTYPE critval=0;
		const std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();
		if(cfg.repeat_over_splits) {
			// trial protocol (demo54/55): one search per outer (depth-0) split; filter
			// criteria re-estimate their model on each trial's training data (demo55-B);
			// the stabileval tracker accumulates the per-trial results
			p.da_mem->setSplittingDepth(0);
			bool run=p.da->getFirstSplit();
			if(!run) throw FST::fst_error("no outer split available for --repeat-over-splits");
			unsigned int trial=0;
			while(run) {
				++trial;
				if(p.crit_reinit) p.crit_reinit();
				p.invalidate_engine_clones(); // the trial's depth-0 split (and a re-estimated filter model) must reach the worker clones
				if(cfg.nested_splits()) p.da_mem->setSplittingDepth(1);
				sub->deselect_all();
				if(cfg.search=="os") sub->make_random_subset(cfg.target_d);
				if(!p.search->search(cfg.target_d,critval,sub,p.crit,progress))
					throw FST::fst_error("search not finished (trial "+std::to_string((long long)trial)+")");
				RunResult::TrialEntry te;
				te.value=critval;
				te.features=subset_features(sub);
				r.trials.push_back(te);
				if(p.tracker_stabil) p.tracker_stabil->add(critval,sub);
				p.da_mem->setSplittingDepth(0);
				run=p.da->getNextSplit();
			}
			if(p.tracker_stabil && r.trials.size()>1) {
				const DIMTYPE nn=p.da->getNoOfFeatures();
				RunResult::StabilityEntry e;
				// FST4_STAB is documented at file scope, above struct RunResult
				#define FST4_STAB(NAME,EXPR) e.measure=NAME; e.value=(EXPR); r.stability.push_back(e);
				FST4_STAB("value_mean",p.tracker_stabil->value_mean())
				FST4_STAB("value_stddev",p.tracker_stabil->value_stddev())
				FST4_STAB("size_mean",p.tracker_stabil->size_mean())
				FST4_STAB("size_stddev",p.tracker_stabil->size_stddev())
				FST4_STAB("CWrel",p.tracker_stabil->stability_CWrel(nn))
				FST4_STAB("ATI",p.tracker_stabil->stability_ATI())
				FST4_STAB("CW",p.tracker_stabil->stability_CW())
				FST4_STAB("ANHI",p.tracker_stabil->stability_ANHI(nn))
				FST4_STAB("SH",p.tracker_stabil->stability_SH())
				FST4_STAB("PH",p.tracker_stabil->stability_PH(nn))
				FST4_STAB("C",p.tracker_stabil->stability_C())
				#undef FST4_STAB
			}
		}
		else if(cfg.restarts_nonimproving>0) {
			// random-restart protocol (demo22): re-run the search from random subsets of
			// `--target-d` features until N consecutive runs fail to improve the best value;
			// per-run results are reported in the trials block
			std::shared_ptr<SUBSET> bestsub(new SUBSET(n));
			unsigned int nonimp=0;
			unsigned long runs=0;
			bool first=true;
			while(nonimp<cfg.restarts_nonimproving && (cfg.restarts_max==0 || runs<cfg.restarts_max)) {
				++runs;
				sub->make_random_subset(cfg.target_d);
				RETURNTYPE val;
				if(!p.search->search(cfg.target_d,val,sub,p.crit,progress))
					throw FST::fst_error("search not finished (restart "+std::to_string((long long)runs)+")");
				RunResult::TrialEntry te;
				te.value=val;
				te.features=subset_features(sub);
				r.trials.push_back(te);
				if(first || val>critval) {critval=val; bestsub->stateless_copy(*sub); nonimp=0; first=false;}
				else ++nonimp;
			}
			sub->stateless_copy(*bestsub);
		}
		else if(!cfg.regr_multipliers.empty()) {
			// kernel-multiplier sweep (demo11reg): one full search per multiplier, the best
			// training criterion value wins; each swept model is validated on the holdout
			std::vector<double> mults=split_csv_doubles(cfg.regr_multipliers,"--regr-multipliers");
			std::shared_ptr<SUBSET> bestsub(new SUBSET(n));
			bool first=true;
			for(std::vector<double>::const_iterator m=mults.begin();m!=mults.end();++m) {
				p.rwn->set_kernel_multiplier(*m);
				p.invalidate_engine_clones(); // the worker clones hold regressor copies with the previous multiplier
				p.da_mem->setSplittingDepth(eval_depth);
				sub->deselect_all();
				RETURNTYPE val;
				if(!p.search->search(cfg.target_d,val,sub,p.crit,progress)) throw FST::fst_error("search not finished (multiplier sweep)");
				RunResult::SweepEntry e;
				e.multiplier=*m;
				e.value=val;
				e.features=subset_features(sub);
				if(!cfg.validate.empty()) e.validation=run_validation(cfg,p,sub);
				r.multiplier_sweep.push_back(e);
				if(first || val>critval) {critval=val; r.best_multiplier=*m; bestsub->stateless_copy(*sub); first=false;}
			}
			sub->stateless_copy(*bestsub);
			p.rwn->set_kernel_multiplier(r.best_multiplier); // the final validation below reruns with the winning multiplier
		}
		else {
			if(!p.search->search(cfg.target_d,critval,sub,p.crit,progress)) throw FST::fst_error("search not finished");
#ifdef FST_SVM
			// alternating subset-search / SVM-parameter optimization (demo23/62): repeat
			// while the training criterion improves; keep the best subset and restore the
			// best-found parameters before validation
			if(cfg.svm_optimize=="alternating") {
				std::shared_ptr<SUBSET> sub_temp(new SUBSET(n));
				double bestC=p.svm->get_parameter_C(), bestG=p.svm->get_parameter_gamma(), bestF=p.svm->get_parameter_coef0();
				bool stop=false;
				while(!stop) {
					const double C0=p.svm->get_parameter_C(), g0=p.svm->get_parameter_gamma(), f0=p.svm->get_parameter_coef0();
					if(!p.svm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,-2,5,progress))
						throw FST::fst_error("--svm-optimize: SVM hyperparameter re-optimization failed");
					if(p.svm->get_parameter_C()!=C0 || p.svm->get_parameter_gamma()!=g0 || p.svm->get_parameter_coef0()!=f0) {
						p.invalidate_engine_clones(); // the worker clones hold SVM copies with the previous parameters
						sub_temp->stateless_copy(*sub);
						RETURNTYPE critval_temp;
						if(!p.search->search(cfg.target_d,critval_temp,sub_temp,p.crit,progress)) throw FST::fst_error("search not finished");
						if(critval_temp>critval) {
							critval=critval_temp;
							sub->stateless_copy(*sub_temp);
							bestC=p.svm->get_parameter_C(); bestG=p.svm->get_parameter_gamma(); bestF=p.svm->get_parameter_coef0();
						}
						else stop=true;
					}
					else stop=true;
				}
				p.svm->set_parameter_C(bestC); p.svm->set_parameter_gamma(bestG); p.svm->set_parameter_coef0(bestF);
			}
#endif
			// the native RBF SVM alternating protocol: identical choreography, C/gamma only
			if(cfg.rbfsvm_optimize=="alternating") {
				std::shared_ptr<SUBSET> sub_temp(new SUBSET(n));
				REALTYPE bestC=p.rbfsvm->get_parameter_C(), bestG=p.rbfsvm->get_parameter_gamma();
				bool stop=false;
				while(!stop) {
					const REALTYPE C0=p.rbfsvm->get_parameter_C(), g0=p.rbfsvm->get_parameter_gamma();
					if(!p.rbfsvm->optimize_parameters(p.da,sub,100,100,-5,9,-15,3,progress))
						throw FST::fst_error("--rbfsvm-optimize: RBF SVM hyperparameter re-optimization failed");
					if(p.rbfsvm->get_parameter_C()!=C0 || p.rbfsvm->get_parameter_gamma()!=g0) {
						p.invalidate_engine_clones(); // the worker clones hold RBF SVM copies with the previous parameters
						sub_temp->stateless_copy(*sub);
						RETURNTYPE critval_temp;
						if(!p.search->search(cfg.target_d,critval_temp,sub_temp,p.crit,progress)) throw FST::fst_error("search not finished");
						if(critval_temp>critval) {
							critval=critval_temp;
							sub->stateless_copy(*sub_temp);
							bestC=p.rbfsvm->get_parameter_C(); bestG=p.rbfsvm->get_parameter_gamma();
						}
						else stop=true;
					}
					else stop=true;
				}
				p.rbfsvm->set_parameter_C(bestC); p.rbfsvm->set_parameter_gamma(bestG);
			}
		}
		const std::chrono::steady_clock::time_point t1=std::chrono::steady_clock::now();
		r.wall_time_s=std::chrono::duration<double>(t1-t0).count();
		r.evaluations=p.evals->load();
		r.value=critval;
		r.features=subset_features(sub);

		// classifier bias estimate on the selected subset (demo56): estimated in the
		// criterion-protocol split context, before validation retrains anything
		if(cfg.bias_estimate) {
			std::shared_ptr<CRIT_BIAS> wb(new CRIT_BIAS);
			wb->initialize(p.clf,p.da);
			p.da_mem->setSplittingDepth(eval_depth);
			if(!wb->evaluate(r.bias,sub)) throw FST::fst_error("bias estimation failed");
			r.has_bias=true;
		}

		// final validation on the depth-0 holdout (demo protocol: train on the outer
		// train part, test on the outer test part) — runs BEFORE the per_split loop so
		// the depth-0 split state established at setup is still pristine
		if(!cfg.validate.empty() && !cfg.repeat_over_splits)
			r.validation=run_validation(cfg,p,sub,cfg.detailed_report()?&r:nullptr,true); // is_final: the `--dump-scores` call site

		// `--report` detailed: the best-per-size trajectory re-validated on the same pristine
		// depth-0 holdout (the criterion-axis vs validated-axis curve),
		// plus the tuned hyperparameters of any optimize protocol
		if(curve_rec) for(const auto &kv : curve_rec->best_per_size()) {
			RunResult::CurveEntry e;
			e.d=kv.first;
			e.value=kv.second.first;
			e.features=subset_features(kv.second.second);
			if(!cfg.validate.empty()) e.validation=run_validation(cfg,p,kv.second.second);
			r.curve_over_d.push_back(e);
		}
		if(cfg.detailed_report()) {
#ifdef FST_SVM
			if(cfg.svm_optimize!="none") {r.tuned_which="svm"; r.tuned_c=p.svm->get_parameter_C(); r.tuned_gamma=p.svm->get_parameter_gamma(); r.tuned_coef0=p.svm->get_parameter_coef0(); r.tuned_has_coef0=true;}
#endif
			if(cfg.rbfsvm_optimize!="none") {r.tuned_which="rbfsvm"; r.tuned_c=(double)p.rbfsvm->get_parameter_C(); r.tuned_gamma=(double)p.rbfsvm->get_parameter_gamma();}
		}

		// classify new samples on the selected subspace (demo63): train the validation
		// classifier on the depth-0 train part, then classify each file row
		if(!cfg.classify_samples.empty()) {
			const std::string vname = cfg.validate.empty() ? std::string("same") : split_csv(cfg.validate)[0];
			std::shared_ptr<CLASSIFIER> vc;
			if(vname=="same") {
				if(!p.clf) throw FST::fst_error("--classify-samples needs a classifier: use a wrapper criterion or name one via --validate");
				vc=p.clf;
			}
			else vc=make_classifier(vname,cfg,p.da);
			std::ifstream f(cfg.classify_samples.c_str());
			if(!f) throw FST::fst_error("cannot open --classify-samples file '"+cfg.classify_samples+"'");
			std::vector<DATATYPE> vals;
			DATATYPE v;
			while(f>>v) vals.push_back(v);
			if(vals.empty() || vals.size()%n!=0)
				throw FST::fst_error("--classify-samples must contain whitespace-separated numbers, "+std::to_string((long long)n)
				                     +" per sample (full feature vectors); got "+std::to_string((long long)vals.size())+" values");
			p.da_mem->setSplittingDepth(0);
			if(!vc->train(p.da,sub)) throw FST::fst_error("--classify-samples classifier training failed");
			for(std::vector<DATATYPE>::size_type s=0;s<vals.size()/n;++s) {
				// route each user vector through the accessor's PER-FEATURE transform: the model
				// was trained on scaler-transformed data, so a raw vector would classify silently
				// wrong under to01/whitening (and `--missing-value` substitution would be skipped);
				// under `--scaler-fit train` this is the transform of the depth-0 split just trained on
				p.da_mem->scale_sample_inplace(&vals[s*n]);
				DIMTYPE cls;
				if(!vc->classify(cls,&vals[s*n])) throw FST::fst_error("classification of sample "+std::to_string((long long)(s+1))+" failed");
				r.classified.push_back(cls);
			}
		}

		// per-split re-evaluation of the final subset on the criterion-protocol splits
		// (schema block per_split; depth 1 when `--splitter2` is active)
		if(!cfg.repeat_over_splits) {
			p.da_mem->setSplittingDepth(eval_depth);
			if(p.da->getNoOfSplits()>1) {
				RETURNTYPE val;
				for(bool b=p.da->getFirstSplit();b;b=p.da->getNextSplit()) {
					if(!p.split_eval(val,sub)) throw FST::fst_error("per-split re-evaluation of the final subset failed");
					r.per_split.push_back(val);
				}
			}
		}

		// top-k from the result tracker
		if(p.tracker) {
			TRACKER::PResultRec rec;
			for(bool b=p.tracker->get_first(rec);b && r.top_k.size()<cfg.top_k;b=p.tracker->get_next(rec)) {
				RunResult::TopEntry e;
				e.value=rec->value;
				e.features=subset_features(rec->sub);
				r.top_k.push_back(e);
			}
		}

		// regularization sweep (demo51/52t/61): among tracked candidates within margin of
		// the achieved maximum, pick the one optimizing the secondary criterion, re-validate
		if(!cfg.regularize_margins.empty()) {
			std::shared_ptr<CRITERION> sec;
			if(cfg.regularize_by=="subsetsize")
				sec.reset(new CRIT_NEG(std::shared_ptr<CRITERION>(new CRIT_SUBSIZE))); // maximize -size = prefer smaller subsets
			else { // weights
				std::vector<double> costs=split_csv_doubles(cfg.feature_costs,"--feature-costs");
				if((DIMTYPE)costs.size()!=n)
					throw FST::fst_error("--feature-costs must list exactly one cost per feature ("+std::to_string((long long)n)
					                     +" values for this data, got "+std::to_string((long long)costs.size())+")");
				std::shared_ptr<CRIT_SUMW> sw(new CRIT_SUMW);
				sw->initialize((DIMTYPE)costs.size(),costs.data());
				sec.reset(new CRIT_NEG(sw)); // maximize -sum(costs) = prefer cheaper subsets
			}
			std::vector<double> margins=split_csv_doubles(cfg.regularize_margins,"--regularize-margins");
			for(std::vector<double>::const_iterator m=margins.begin();m!=margins.end();++m) {
				p.da_mem->setSplittingDepth(eval_depth); // criteria evaluate on the criterion-protocol split (demo51)
				RETURNTYPE r1,r2;
				std::shared_ptr<SUBSET> rsub;
				if(!p.tracker_reg->optimize_within_margin((RETURNTYPE)(*m),r1,r2,rsub,sec)) continue;
				RunResult::RegEntry e;
				e.margin=*m;
				e.value=r1;
				e.secondary=-r2; // the tracker maximizes the NEGATED secondary criterion
				e.features=subset_features(rsub);
				if(!cfg.validate.empty()) e.validation=run_validation(cfg,p,rsub);
				r.regularization.push_back(e);
			}
		}

		// feature-stats post-processing: DAF ranking extraction (the demo34/35t protocol)
		if(p.tracker_stats) {
			if(!p.tracker_stats->compute_stats(progress)) throw FST::fst_error("feature-stats tracker holds no results to compute statistics from");
			if(!silent) p.tracker_stats->print_stats(progress);
			if(cfg.daf_rank>=0) {
				RETURNTYPE v; DIMTYPE f;
				for(bool b=p.tracker_stats->getFirstDAF(v,f,cfg.daf_rank);
				    b && (cfg.daf_count==0 || r.daf.size()<cfg.daf_count);
				    b=p.tracker_stats->getNextDAF(v,f,cfg.daf_rank)) {
					RunResult::DafEntry e; e.feature=f; e.value=v;
					r.daf.push_back(e);
				}
				if(cfg.daf_validate_every>0) { // demo34: validate incrementally built top-DAF subsets
					p.da_mem->setSplittingDepth(0);
					std::vector<std::string> vnames=split_csv(cfg.validate);
					std::shared_ptr<CLASSIFIER> vc = (vnames[0]=="same") ? p.clf : make_classifier(vnames[0],cfg,p.da);
					std::shared_ptr<SUBSET> dsub(new SUBSET(n));
					dsub->deselect_all();
					for(std::vector<RunResult::DafEntry>::size_type i=0;i<r.daf.size();++i) {
						dsub->select(r.daf[i].feature);
						if((i+1)%cfg.daf_validate_every==0) {
							RunResult::ValidationEntry e;
							e.classifier=std::to_string((long long)(i+1));
							if(!vc->train(p.da,dsub)) throw FST::fst_error("top-DAF subset validation training failed");
							if(!vc->test(e.accuracy,p.da)) throw FST::fst_error("top-DAF subset validation testing failed");
							r.daf_validation.push_back(e);
						}
					}
				}
			}
			if(cfg.daf_pairs>0) { // pairwise group statistics over the top-K DAF0 features (works with or without `--daf-rank`)
				if(!p.tracker_stats->compute_pair_stats(cfg.daf_pairs,progress))
					throw FST::fst_error("pairwise DAF statistics could not be computed (fewer than 2 DAF0-valid features)");
				if(!silent) p.tracker_stats->print_pair_stats(progress,cfg.daf_pairs_top);
				RETURNTYPE d0,d1,syn; DIMTYPE pf1,pf2; bool d1v;
				for(bool b=p.tracker_stats->getFirstPairDAF(d0,d1,d1v,syn,pf1,pf2);
				    b && (cfg.daf_pairs_top==0 || r.daf_pairs.size()<cfg.daf_pairs_top);
				    b=p.tracker_stats->getNextPairDAF(d0,d1,d1v,syn,pf1,pf2)) {
					RunResult::PairDafEntry e; e.f1=pf1; e.f2=pf2; e.daf0=d0; e.daf1=d1; e.daf1_valid=d1v; e.synergy=syn;
					r.daf_pairs.push_back(e);
				}
			}
		}

		return r;
	}; // run_pipeline

	int exit_code=0;
	Pipeline p;
	CompareResult cmp;
	bool have_cmp=false;
	std::unique_ptr<Stdin_Data_File> stdin_data; // outlives both scenarios; deletes the buffer on scope exit
	try {
		// exactly one entry point: a data set (`--data`) or an abstract feature universe
		// (`--features`); enforced here rather than via ->required()/->excludes() so the
		// error can name both flags and their intent
		if(cfg.data.empty() && !cfg.data_free())
			throw FST::fst_error("--data is required (or --features N for data-free set-function optimization with --criterion extern-value --extern-no-data)");
		if(!cfg.data.empty() && cfg.data_free())
			throw FST::fst_error("--data and --features are mutually exclusive: --features declares an abstract data-free universe, a data file defines a concrete one");

		// `--data` -: buffer stdin BEFORE anything else reads it, and hand the readers the
		// buffer's path (cfg.data itself stays "-", so the reported invocation does too)
		if(cfg.data_is_stdin()) {
			if(cfg.format.empty())
				throw FST::fst_error("--data - reads the dataset from standard input, where there is no file name to infer the format from; add --format trn|arff|csv");
			stdin_data.reset(new Stdin_Data_File);
			cfg.data_resolved=stdin_data->path();
		}
		FST::fst_srand(cfg.seed); // once for the whole process — see the run_pipeline note
		RunResult r=run_pipeline(cfg,p);

		// `--curves-svg`: render the `--report` detailed curves (validity checked in run_pipeline;
		// false = detailed mode produced not a single curve, e.g. a regression pipeline with
		// no ROC and a search that announced no solutions)
		if(!cfg.curves_svg.empty() && !write_curves_svg(cfg.curves_svg,r))
			throw FST::fst_error("--curves-svg: the run produced no curves to draw (ROC/PR need --validate naming a classifier with score support; the over-d curve needs a search announcing solutions)");

		if(!cfg.compare_config.empty()) {
			// demo55 protocol: the second scenario runs in the same process, continuing
			// the rand() stream; the two stabileval trackers are then compared
			cmp.cfg2=parse_compare_config(cfg.compare_config);
			if(cmp.cfg2.data_is_stdin()) cmp.cfg2.data_resolved=cfg.data_resolved; // both scenarios read the one buffered stdin
			if(cfg.tracker!="stabileval" || cmp.cfg2.tracker!="stabileval" || !cmp.cfg2.repeat_over_splits)
				throw FST::fst_error("--compare-config: cross-scenario similarity compares per-trial result trackers; BOTH scenarios need the stabileval tracker and the repeat-over-splits protocol (add --tracker stabileval --repeat-over-splits on the command line, and tracker=\"stabileval\" plus repeat-over-splits=true in the config file)");
			if(cmp.cfg2.data!=cfg.data)
				throw FST::fst_error("--compare-config: both scenarios must select over the same --data file (feature-space identity is what makes the similarity measures meaningful); the config file names '"+cmp.cfg2.data+"'");
			if(!cmp.cfg2.compare_config.empty())
				throw FST::fst_error("--compare-config: the compared config file must not itself set compare-config");
			cmp.r2=run_pipeline(cmp.cfg2,cmp.p2);
			const DIMTYPE nn=p.da->getNoOfFeatures();
			RunResult::StabilityEntry e;
			e.measure="IATI";  e.value=p.tracker_stabil->similarity_IATI(*cmp.p2.tracker_stabil);     cmp.similarity.push_back(e);
			e.measure="ICW";   e.value=p.tracker_stabil->similarity_ICW(*cmp.p2.tracker_stabil);      cmp.similarity.push_back(e);
			e.measure="IANHI"; e.value=p.tracker_stabil->similarity_IANHI(nn,*cmp.p2.tracker_stabil); cmp.similarity.push_back(e);
			have_cmp=true;
		}

		const std::string format = cfg.data_free() ? "none" : (cfg.format.empty() ? infer_format(cfg.data) : cfg.format);
		if(cfg.output=="json") emit_json(real_stdout,cfg,format,p,r,have_cmp?&cmp:nullptr);
		else if(cfg.output=="csv") emit_csv(real_stdout,cfg,r,have_cmp?&cmp:nullptr);
		else emit_text(real_stdout,cfg,format,p,r,have_cmp?&cmp:nullptr);
	}
	catch(FST::fst_error &e) {
		const std::string msg=e.what();
		std::cerr<<"FST ERROR: "<< msg << ", code=" << e.code() << std::endl;
		if(msg.find("singular")!=std::string::npos && p.da) {
			// normal-model criteria fail on constant features (zero variance => singular
			// covariance); name the offenders if we can
			const std::string cf=list_constant_features(p.da);
			if(!cf.empty())
				std::cerr<<"HINT: the covariance matrix is singular because feature(s) "<<cf
				         <<" are constant on the current training split; remove them from the data, or use a wrapper criterion (wrapper-knn etc.), which does not invert covariance matrices"<<std::endl;
			else
				std::cerr<<"HINT: a singular covariance matrix usually means constant or linearly dependent features on the training split; try a different --splitter/--seed, remove the dependent features, or use a wrapper criterion"<<std::endl;
		}
		exit_code=1;
	}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; exit_code=1;}

	std::cout.rdbuf(stdout_buf);
	return exit_code;
}
