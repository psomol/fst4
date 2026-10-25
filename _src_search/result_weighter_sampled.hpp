#ifndef FSTRESULTWEIGHTERSAMPLED_H
#define FSTRESULTWEIGHTERSAMPLED_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    result_weighter_sampled.hpp
   \brief   Online per-feature statistics for budgeted (sampled) sequential steps — discounted batch-standardized DAF contrast
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0
   \note    FST4 was developed using MinGW-W64 g++ 15.2 and requires
   \note    \li a C++17 conforming compiler and standard library (no Boost, unlike FST3),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/,
                version 3.37; not distributed with FST4 — drop svm.cpp/svm.h into ./libsvm/, see libsvm/readme.txt)
   \note    Note that LibSVM is required for SVM related tools only.
            Ready-to-run SVM examples: demos/demo12t.cfg, demo23.cfg, demo25t.cfg
            and demo32t.cfg (configurations of the fst4 command-line tool).

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

#include <memory>
#include <vector>
#include <mutex>
#include <cmath>
#include <cassert>
#include <sstream>
#include <algorithm>
#include "error.hpp"
#include "global.hpp"

/*============== Template parameter type naming conventions ==============
--------- Numeric types: -------------------------------------------------
DATATYPE - data sample values - usually real numbers (but may be integers
          in text processing etc.)
REALTYPE - must be real numbers - for representing intermediate results of
          calculations like mean, covariance etc.
IDXTYPE - index values for enumeration of data samples - (nonnegative) integers,
          extent depends on numbers of samples in data
DIMTYPE - index values for enumeration of features (dimensions), or classes (not
          class sizes) - (nonnegative) integers, usually lower extent than IDXTYPE,
          but be aware of expressions like _classes*_features*_features !
          in linearized representations of feature matrices for all classes
BINTYPE - feature selection marker type - represents ca. <10 different feature
          states (selected, deselected, sel./desel. temporarily 1st nested loop, 2nd...)
RETURNTYPE - criterion value: real value, but may be extended in future to support
          multiple values
--------- Class types: ---------------------------------------------------
SUBSET       - class of class type Subset
CRITERION    - class implementing interface defined in abstract class Criterion
========================================================================== */

namespace FST {

/*! \brief Online per-feature statistics for budgeted (sampled) sequential steps —
           discounted batch-standardized DAF contrast

    The statistics engine behind Sequential_Step_Sampled: every criterion
    evaluation the step performs is fed here, and score(f) is what the step's
    proposal sampler concentrates its candidate budget on. See arXiv:2608.01502
    for the method.

    Collects criterion evaluations into a batch buffer (one batch = one sequential
    step's candidate sweep, or one warm-up probe round). CALLER CONTRACT: all
    subsets within one batch share the same cardinality and evaluation context —
    z-scores are only meaningful within such a batch. flush_batch() z-standardizes
    the buffered values and folds each evaluation into per-feature accumulators:
    for an evaluated subset with standardized score z, the 'is' accumulator of every
    selected feature and the 'isnot' accumulator of every unselected feature receive
    an update with z. score(f) = m_is(f) - m_isnot(f), a batch-relative DAF-style
    contrast that is comparable across subset sizes by construction (no per-size
    bins) and covers forward steps, backward steps and probes with one uniform rule:
    in a forward batch the added feature is selected in exactly one candidate, in a
    backward batch the removed feature is unselected in exactly one candidate, so
    the contrast picks up the correct sign automatically.

    The accumulators use an adaptive rate 1/min(t,horizon): exact running mean until
    a feature's side has seen `horizon` updates, exponential forgetting beyond it
    (keeps statistics fresh once the search trajectory drifts to high |S|).
    Unobserved features score 0 = the batch average — the correct neutral prior for
    centered z-scores, so no cold-start bias correction is needed.

    STATS_RAWMEAN is a simpler alternative statistic: the 1/|S|-size-weighted
    mean of the RAW criterion values over 'is' contexts only, with no
    standardization and no 'isnot' side. It is kept as a baseline to compare the
    contrast against; be aware of its flaw, namely that the absolute level it
    reports is dominated by WHEN during the search a feature happened to be
    evaluated (early small subsets and late large ones score on different
    scales), which is exactly what the z-contrast removes.

    This is a plain statistics engine, deliberately independent of the
    Result_Tracker hierarchy: it neither stores subsets for later retrieval nor
    takes part in candidate iteration. The sampling policy that consumes the
    scores lives entirely in Sequential_Step_Sampled.

    Thread safety: add() is mutex-protected (safe from threaded-evaluator worker
    callbacks); flush_batch() is to be called at batch boundaries only, where the
    step object is single-threaded by construction.
*/
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Weighter_Sampled {
public:
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared-ownership handle to the evaluated Subset passed into add()

	/*! \brief Selects which statistic score(f) reports: STATS_ZCONTRAST (default) is the
	           batch-standardized DAF-style contrast m_is-m_isnot described above; STATS_RAWMEAN
	           is the 1/|S|-size-weighted mean of raw criterion values over 'is' contexts only
	           (the simpler baseline statistic; the fst4 tool selects it with `--step-stats rawmean`). */
	enum StatsMode { STATS_ZCONTRAST, STATS_RAWMEAN };

	/*! \brief Construct with the given adaptive-rate horizon and scoring mode; all statistics
	           start at zero and are properly sized on the first add() or an explicit reset().
	    \param horizon exponential-forgetting horizon (see class doc); clamped to at least 1
	    \param mode STATS_ZCONTRAST or STATS_RAWMEAN (see StatsMode) */
	Result_Weighter_Sampled(const IDXTYPE horizon=100, const StatsMode mode=STATS_ZCONTRAST)
		: _n(0), _horizon(horizon<1?1:horizon), _mode(mode), _frozen(false), _total_evals(0), _flushed_batches(0)
		{notify("Result_Weighter_Sampled constructor.");}
	virtual ~Result_Weighter_Sampled() {notify("Result_Weighter_Sampled destructor.");}

	//! \brief (Re)dimension to \p n features and zero all statistics, the pending batch and the counters.
	//! Also clears the frozen flag. NOT thread-safe and NOT mutex-protected: call it only when no
	//! add() can run concurrently (add() itself calls it once, under the lock, on the very first use).
	void reset(const DIMTYPE n);
	DIMTYPE get_n() const {return _n;} //!< current feature-space dimensionality (0 until the first add() or reset())
	StatsMode get_mode() const {return _mode;} //!< the configured StatsMode (see StatsMode)
	IDXTYPE get_horizon() const {return _horizon;} //!< the (clamped, >=1) adaptive-rate horizon supplied at construction

	//! \brief Buffer one evaluated subset for the current batch (thread-safe; may be called from
	//! threaded-evaluator worker callbacks). \return false only if \p sub's feature-space size
	//! disagrees with the already established one — the caller should treat that as a fatal
	//! mismatch. Adding while frozen is a successful no-op, i.e. it returns true.
	bool add(const RETURNTYPE value, const PSubset sub);
	IDXTYPE flush_batch(); //!< z-score the buffered batch and fold it into the accumulators; returns the number of evaluations flushed (0 for an empty batch or while frozen)

	//! \brief Freezes the ranking: flushes any pending batch, then turns all further
	//! add()/flush_batch() calls into successful no-ops — the statistics (scores AND exploration
	//! counts) stay exactly as they were at this point, e.g. as a warm-up round left them.
	//! Takes the mutex only AFTER flush_batch() has released it, so never call it while holding
	//! the lock yourself.
	void freeze() {flush_batch(); std::lock_guard<std::mutex> lock(_add_mutex); _frozen=true;}
	bool is_frozen() const {return _frozen;} //!< true once freeze() has been called (see freeze())

	RETURNTYPE score(const DIMTYPE f) const; //!< current sampling score of feature f (see class doc)
	IDXTYPE count_is(const DIMTYPE f) const {assert(f<_n); return _fs[f].n_is;} //!< how many evaluated subsets contained f (exploration bookkeeping)
	IDXTYPE count_isnot(const DIMTYPE f) const {assert(f<_n); return _fs[f].n_isnot;} //!< how many evaluated subsets lacked f ("what happens without f" is the scarce observation for backward-step exploration)
	IDXTYPE total_evals() const {return _total_evals;} //!< cumulative count of evaluations passed to add() since construction/reset(), including any still buffered and not yet flushed
	IDXTYPE flushed_batches() const {return _flushed_batches;} //!< number of non-empty flush_batch() calls processed since construction/reset() (a count of batches, not of evaluations)

	/*! \brief Write a compact one-line summary (n, horizon, mode, frozen flag, total evaluations) to os.
	    Writes directly via operator<<, not through syncout::print — callers from threaded contexts
	    should route the result through syncout::print themselves, or prefer print_stats() below,
	    which already does. */
	std::ostream& print(std::ostream& os) const {os << "Result_Weighter_Sampled(n=" << _n << ", horizon=" << _horizon << ", stats=" << (_mode==STATS_ZCONTRAST?"zcontrast":"rawmean") << (_frozen?", FROZEN":"") << ", evals=" << _total_evals << ")"; return os;}
	bool print_stats(std::ostream& os, const DIMTYPE top=20) const; //!< diagnostic listing of the top-scored features

protected:
	/*! \brief Per-feature accumulators, one instance per feature index, updated by flush_batch().
	    m_is/m_isnot are adaptive-rate running means of the batch z-score, one side for evaluations
	    in which the feature was selected ('is') and one for evaluations in which it was not
	    ('isnot'); score() in STATS_ZCONTRAST mode returns their difference. raw_sum_is/raw_cnt_is
	    accumulate the STATS_RAWMEAN statistic instead. n_is/n_isnot count how many
	    evaluations landed on each side — also the exploration counts exposed via
	    count_is()/count_isnot(). */
	struct FeatureStat {
		RETURNTYPE m_is;        //!< adaptive-rate mean of batch-z over evaluations where the feature was selected (STATS_ZCONTRAST score source)
		RETURNTYPE m_isnot;     //!< adaptive-rate mean of batch-z over evaluations where the feature was not selected (STATS_ZCONTRAST score source)
		RETURNTYPE v_is;        //!< adaptive-rate mean of z^2 on the 'is' side; maintained as a dispersion estimate but not read by score()
		RETURNTYPE raw_sum_is;  //!< 1/|S|-weighted sum of raw criterion values over 'is' evaluations (STATS_RAWMEAN score source)
		RETURNTYPE raw_cnt_is;  //!< matching 1/|S|-weighted normalizer for raw_sum_is
		IDXTYPE n_is;    //!< number of evaluations in which the feature was selected (exploration count, see count_is())
		IDXTYPE n_isnot; //!< number of evaluations in which the feature was not selected (exploration count, see count_isnot())
	};

	/*! \brief One buffered, not-yet-flushed criterion evaluation: its raw value, the cardinality
	           of the evaluated subset, and the list of selected feature indices — everything
	           flush_batch() needs to z-standardize the batch and fold it into the accumulators. */
	struct BatchEntry {
		RETURNTYPE value;              //!< the raw criterion value returned for this evaluated subset
		DIMTYPE d;                     //!< cardinality of the evaluated subset (caller contract: identical for every entry in one batch)
		std::vector<DIMTYPE> selected; //!< indices of the subset's selected features
	};
	DIMTYPE _n; //!< current feature-space dimensionality, set by reset() (called explicitly, or lazily by the first add())
	IDXTYPE _horizon; //!< adaptive-rate horizon supplied at construction (clamped to at least 1)
	StatsMode _mode; //!< configured StatsMode, selects which statistic score() reports
	bool _frozen; //!< set by freeze(); once true, add()/flush_batch() are no-ops and statistics stay frozen at their warm-up values
	std::vector<FeatureStat> _fs; //!< per-feature accumulators, indexed 0.._n-1
	std::vector<BatchEntry> _batch; //!< buffered, not-yet-flushed evaluations added since the last flush_batch()
	std::vector<char> _mark;               //!< flush_batch() scratch bitmap (size _n); marks one entry's selected features while folding it in, then clears them again
	IDXTYPE _total_evals;     //!< cumulative count of evaluations passed to add() since construction/reset() (see total_evals())
	IDXTYPE _flushed_batches; //!< count of non-empty flush_batch() calls since construction/reset() (see flushed_batches())
	std::mutex _add_mutex; //!< guards _batch/_total_evals/_frozen so add() is safe to call from threaded-evaluator worker callbacks
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
void Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n)
{
	assert(n>0);
	_n=n;
	_fs.assign(_n,FeatureStat());
	for(DIMTYPE f=0;f<_n;f++) {
		_fs[f].m_is=0.0; _fs[f].m_isnot=0.0; _fs[f].v_is=0.0;
		_fs[f].raw_sum_is=0.0; _fs[f].raw_cnt_is=0.0;
		_fs[f].n_is=0; _fs[f].n_isnot=0;
	}
	_mark.assign(_n,0);
	_batch.clear();
	_frozen=false;
	_total_evals=0;
	_flushed_batches=0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::add(const RETURNTYPE value, const PSubset sub)
{
	assert(sub);
	std::lock_guard<std::mutex> lock(_add_mutex);
	if(_frozen) return true; // frozen ranking: the no-op is a success, callers must not treat it as an error
	if(_n==0) reset(sub->get_n());
	assert(_n==sub->get_n());
	if(_n!=sub->get_n()) return false;
	BatchEntry e;
	e.value=value;
	e.d=sub->get_d();
	e.selected.reserve(e.d);
	for(DIMTYPE f : sub->selected_features()) e.selected.push_back(f);
	_batch.push_back(std::move(e));
	_total_evals++;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
IDXTYPE Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::flush_batch()
{
	std::lock_guard<std::mutex> lock(_add_mutex);
	if(_frozen) {_batch.clear(); return 0;}
	const IDXTYPE B=(IDXTYPE)_batch.size();
	if(B==0) return 0;
	assert(_n>0);
	double mu=0.0;
	for(IDXTYPE i=0;i<B;i++) mu+=(double)_batch[i].value;
	mu/=(double)B;
	double var=0.0;
	for(IDXTYPE i=0;i<B;i++) {const double dv=(double)_batch[i].value-mu; var+=dv*dv;}
	var/=(double)B;
	const double sd=std::sqrt(var);
	const bool degenerate=!(sd>1e-12); // constant batch (or B==1) carries no ranking information -> z=0
	for(IDXTYPE i=0;i<B;i++) {
		const BatchEntry &e=_batch[i];
		assert(e.d==_batch[0].d); // caller contract: one batch = one cardinality
		const RETURNTYPE z=degenerate?(RETURNTYPE)0.0:(RETURNTYPE)(((double)e.value-mu)/sd);
		for(size_t k=0;k<e.selected.size();k++) _mark[e.selected[k]]=1;
		const RETURNTYPE dinv=(e.d>0)?(RETURNTYPE)(1.0/(double)e.d):(RETURNTYPE)0.0;
		for(DIMTYPE j=0;j<_n;j++) {
			FeatureStat &s=_fs[j];
			if(_mark[j]) {
				s.n_is++;
				const RETURNTYPE a=(RETURNTYPE)1.0/(RETURNTYPE)(s.n_is<_horizon?s.n_is:_horizon);
				s.m_is+=a*(z-s.m_is);
				s.v_is+=a*(z*z-s.v_is);
				s.raw_sum_is+=dinv*e.value;
				s.raw_cnt_is+=dinv;
			} else {
				s.n_isnot++;
				const RETURNTYPE a=(RETURNTYPE)1.0/(RETURNTYPE)(s.n_isnot<_horizon?s.n_isnot:_horizon);
				s.m_isnot+=a*(z-s.m_isnot);
			}
		}
		for(size_t k=0;k<e.selected.size();k++) _mark[e.selected[k]]=0;
	}
	_batch.clear();
	_flushed_batches++;
	return B;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::score(const DIMTYPE f) const
{
	assert(f<_n);
	const FeatureStat &s=_fs[f];
	if(_mode==STATS_RAWMEAN) return (s.raw_cnt_is>0.0)?(s.raw_sum_is/s.raw_cnt_is):(RETURNTYPE)0.0;
	return s.m_is-s.m_isnot;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Weighter_Sampled<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::print_stats(std::ostream& os, const DIMTYPE top) const
{
	std::ostringstream sos;
	if(_n<1) {sos << "Result_Weighter_Sampled: nothing to print." << std::endl; syncout::print(os,sos); return false;}
	std::vector<DIMTYPE> idx(_n);
	for(DIMTYPE f=0;f<_n;f++) idx[f]=f;
	const DIMTYPE k=(top>0 && top<_n)?top:_n;
	std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),
		[this](const DIMTYPE a,const DIMTYPE b){const RETURNTYPE sa=score(a),sb=score(b); return sa>sb || (sa==sb && a<b);});
	sos << "Sampled-step feature stats (top " << k << " of " << _n << ", evals=" << _total_evals << ", batches=" << _flushed_batches << "):" << std::endl;
	for(DIMTYPE i=0;i<k;i++) {
		const DIMTYPE f=idx[i];
		sos << " F:" << f << " score=" << score(f)
		    << " m_is=" << _fs[f].m_is << " m_isnot=" << _fs[f].m_isnot
		    << " n_is=" << _fs[f].n_is << " n_isnot=" << _fs[f].n_isnot << std::endl;
	}
	syncout::print(os,sos);
	return true;
}

} // namespace
#endif // FSTRESULTWEIGHTERSAMPLED_H ///:~
