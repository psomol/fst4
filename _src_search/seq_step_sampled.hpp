#ifndef FSTSEARCHSEQSTEPSAMPLED_H
#define FSTSEARCHSEQSTEPSAMPLED_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    seq_step_sampled.hpp
   \brief   Implements the budgeted (sampled) sequential selection step — per-step candidate evaluations capped, candidates proposed by online per-feature statistics
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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <sstream>
#include "error.hpp"
#include "global.hpp"
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
#include "seq_step.hpp"
#include "candidate_evaluator.hpp"
#include "candidate_evaluator_sequential.hpp"
#include "result_weighter_sampled.hpp"

/*============== Template parameter type naming conventions ==============
--------- Numeric types: -------------------------------------------------
RETURNTYPE - criterion value: real value
DIMTYPE - index values for enumeration of features (dimensions)
--------- Class types: ---------------------------------------------------
SUBSET       - class of class type Subset
CRITERION    - class implementing interface defined in abstract class Criterion
WEIGHTER     - Result_Weighter_Sampled instantiation (the statistics engine)
========================================================================== */

namespace FST {

/*! \brief Implements the budgeted (sampled) sequential selection step: forward
           steps evaluate only a sampled batch of candidates, backward steps
           sweep in full unless a backward budget is set

    The point of the step is that the number of criterion evaluations it spends
    is bounded by a user-set candidate budget INDEPENDENT of the feature-space
    size n, which keeps sequential selection tractable where the classic full
    sweep (n-d evaluations per step) is not. Candidates are not drawn blindly:
    an online per-feature statistics engine (Result_Weighter_Sampled), fed by
    every criterion evaluation the search performs, scores each feature and the
    proposal sampler concentrates the budget on the promising ones while an
    exploration floor keeps the rest reachable. For the method and its
    evaluation see arXiv:2608.01502.

    The class is a drop-in replacement for Sequential_Step_Straight in any
    search built on Sequential_Step (SFS/SFFS/SFRS, Oscillating and Dynamic
    Oscillating Search); the fst4 command-line tool selects it with
    `--evaluator sampled`.

    Forward steps with a free-feature pool larger than the cap propose exactly
    `cap` candidates per step instead of sweeping the whole pool:
    - a fraction `explore` of the batch is drawn uniformly (without replacement)
      from the least-evaluated features (exploration floor — solves cold start
      and keeps low-scoring features discoverable),
    - the remainder is drawn without replacement with probabilities
      softmax(score/tau) over the statistics engine's current per-feature scores;
      tau<=0 selects a robust automatic scale (IQR of the current free-feature
      scores / 1.349, floored) so the policy is self-tuning.
    SAMPLER_UNIFORM (the stochastic-greedy baseline of Mirzasoleiman et al. 2015)
    and SAMPLER_TOPK (a hard top-cap cutoff, prone to collapsing onto the same
    few features) are the alternative proposal policies.

    Backward steps are budgeted SYMMETRICALLY: when set_cap_backward(y_b>0) is
    active and the subset is larger than y_b, exactly y_b removal candidates are
    proposed — the exploration-floor fraction uniformly from the members whose
    ABSENCE is least observed (count_isnot), the remainder drawn without
    replacement ~ softmax(-score/tau) so LOW-scoring members are removed first.
    This matters because uncapped backward sweeps otherwise dominate the budget
    of a floating search at large d. The default y_b==0 lets backward steps
    sweep in full, which keeps the SFFS floating correction exact (the
    y_b -> |S| limit).
    CAUTION: weighting removal candidates by (1-w/sum(w)) over raw weights
    compresses to near-uniform and carries no signal; the softmax over
    z-contrast scores with the robust tau is the scale-aware alternative. Judge
    a configuration by the B-order diagnostic described below — avgorder must
    fall below ~0.5 for the proposals to carry signal at all.

    Small pools (pool <= cap), cap==0, generalized steps (g>1), and frozen-mode
    subsets fall back to the classic full sweep over sub->get*CandidateSubset()
    — identical in semantics to Sequential_Step_Straight. EVERY evaluated
    candidate (all paths) is fed to the statistics engine; the batch is flushed
    at the end of each step (batches never mix cardinalities — see
    Result_Weighter_Sampled's caller contract).

    All randomness is drawn via FST::fst_rand(), inheriting the process-wide
    seed contract (fst4 `--seed`; unseeded == `--seed` 1). Reruns with the same
    seed reproduce the same result.

    The candidate-evaluation engine (Candidate_Evaluator) is injected through
    the constructor; the default is sequential in-line evaluation. With
    Candidate_Evaluator_Threaded injected, each step's proposal batch is
    evaluated in the thread pool while the proposal sampling itself stays on
    the caller's thread: every fst_rand() draw of a step happens BEFORE the
    first evaluation, criterion evaluate() never draws (fst_rand state is
    thread_local, so workers could not perturb the master stream even if it
    did), and the engine replays results to the statistics engine/tracker/
    best-so-far in add-order (= proposal order) on the caller's thread when the
    batch completes. A threaded run therefore yields exactly the same result as
    a sequential one at the same seed, for any thread count (the weighter is
    never accessed concurrently and its batch semantics are unchanged).

    Diagnostics (printed at NORMAL verbosity): "F order"/"B order" lines report
    at which position in the proposal order the step winner appeared,
    normalized to (0,1] — the direct measure of whether the online statistics
    front-load winners. A value near 1/cnt means the sampler proposes the
    winner first; values around 0.5 mean the proposals are no better than a
    random order of the pool.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
class Sequential_Step_Sampled : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef std::shared_ptr<CRITERION> PCriterion;              //!< shared pointer to the criterion evaluated by this step
	typedef std::shared_ptr<SUBSET> PSubset;                    //!< shared pointer to a candidate/working feature subset
	typedef std::shared_ptr<WEIGHTER> PWeighter;                //!< shared pointer to the statistics engine (Result_Weighter_Sampled)
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< candidate-evaluation engine type (sequential in-line or thread-pooled)
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine;  //!< shared pointer to the evaluation engine
	/*! \brief Proposal-sampling policy for the forward/backward candidate batch (see the class docs above for
	           the exact per-mode draw procedure). SAMPLER_SOFTMAX (default) draws without replacement with
	           probability softmax(+-score/tau) plus an exploration floor; SAMPLER_UNIFORM draws uniformly
	           without replacement from the whole pool (the Mirzasoleiman et al. 2015 stochastic-greedy
	           baseline); SAMPLER_TOPK is a hard score cutoff with no exploration at all, which tends to
	           keep re-proposing the same few features. */
	enum SamplerMode { SAMPLER_SOFTMAX, SAMPLER_UNIFORM, SAMPLER_TOPK };
	/*! \brief Constructs the step with its statistics engine, per-step evaluation budget, proposal sampler and
	           (optional) evaluation engine.
	    \param stats shared statistics engine (Result_Weighter_Sampled); a default-constructed instance is
	           created when null
	    \param cap forward per-step candidate budget (set_cap_frac() overrides it with a percentage-of-pool
	           schedule); a pool not larger than cap falls back to a full sweep
	    \param explore fraction of each proposal batch drawn from the exploration floor (least-evaluated free
	           features, or least-observed-absent members on backward steps), clamped to [0,1]
	    \param tau softmax temperature; <=0 selects the robust self-tuning scale (IQR/1.349 of the current
	           free-feature/member scores) fresh on every step
	    \param sampler proposal policy, see SamplerMode
	    \param engine candidate-evaluation engine (Candidate_Evaluator); sequential in-line evaluation is used
	           when null */
	Sequential_Step_Sampled(PWeighter stats, const DIMTYPE cap=100, const double explore=0.2, const double tau=0.0, const SamplerMode sampler=SAMPLER_SOFTMAX, PEvaluatorEngine engine=PEvaluatorEngine())
		: Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(), _stats(stats), _engine(engine), _cap(cap),
		  _explore(explore<0.0?0.0:(explore>1.0?1.0:explore)), _tau(tau), _sampler(sampler),
		  count_forward(0), count_backward(0), order_forward(0.0), order_backward(0.0), all_evals(0)
	{
		if(!_stats) _stats.reset(new WEIGHTER);
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Sequential_Step_Sampled constructor.");
	}
	virtual ~Sequential_Step_Sampled() {notify("Sequential_Step_Sampled destructor.");}

	//! \brief Switches the forward budget to a constant-PERCENTAGE schedule: when \p frac >0 the
	//! forward step cap becomes ceil(frac*pool) instead of the constant count `cap`, so per-step
	//! cost scales with the pool again — the very property the constant-count default removes.
	//! \p frac is clamped to [0,1]; 0 (default) disables the schedule. Applies to FORWARD steps only.
	void set_cap_frac(const double frac) {_cap_frac=(frac<0.0?0.0:(frac>1.0?1.0:frac));}
	double get_cap_frac() const {return _cap_frac;} //!< current constant-percentage schedule fraction (0 = disabled, see set_cap_frac())

	//! \brief Sets the backward-step budget y_b: 0 (default) = full backward sweeps, which keeps the
	//! floating correction exact; >0 = propose exactly y_b removal candidates, sampled
	//! ~ softmax(-score/tau) with the exploration floor taken over count_isnot.
	void set_cap_backward(const DIMTYPE cap) {_cap_backward=cap;}
	DIMTYPE get_cap_backward() const {return _cap_backward;} //!< current backward-step budget y_b (0 = full sweeps, see set_cap_backward())

	/*! \brief Performs one selection step on \p sub: routes to the budgeted forward path, the budgeted
	           backward path or the classic full sweep, depending on \p sub's current direction mode, the
	           pool size and the configured budgets (see the class docs).
	    \param result set to the criterion value of the winning candidate on success
	    \param sub the working subset; the winning feature is added (forward mode) or removed (backward
	           mode) in place. Its direction (Subset::set_forward_mode()) selects the path.
	    \param crit the criterion each candidate is evaluated against; it must already be initialize()d
	    \param _generalization_level candidate subsets differ from \p sub by this many features; values >1
	           always take the full-sweep path
	    \param os progress and diagnostic messages are written here
	    \return false when there is nothing to do (empty pool), when a candidate could not be evaluated,
	            or when the statistics engine rejected an evaluation; true otherwise */
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout) override;

	PWeighter get_stats() {return _stats;} //!< shared statistics engine (fst4 feeds warm-up probes through it)

	/*! \brief Writes a human-readable identification of the step configuration (cap or the percentage schedule,
	           cap_backward if set, explore, tau or "auto", sampler, the statistics engine's mode and horizon,
	           and the thread count when the injected engine is threaded), followed by the active result tracker
	           if one is attached. */
	virtual std::ostream& print(std::ostream& os) const override {
		os << "Sequential_Step_Sampled(cap=";
		if(_cap_frac>0.0) os << "frac:" << _cap_frac; else os << _cap;
		if(_cap_backward>0) os << ", cap_backward=" << _cap_backward;
		os << ", explore=" << _explore << ", tau=";
		if(_tau>0.0) os << _tau; else os << "auto";
		os << ", sampler=" << (_sampler==SAMPLER_SOFTMAX?"softmax":(_sampler==SAMPLER_UNIFORM?"uniform":"topk"));
		os << ", stats=" << (_stats->get_mode()==WEIGHTER::STATS_ZCONTRAST?"zcontrast":"rawmean") << ", decay=" << _stats->get_horizon();
		if(_engine->is_threaded()) os << ", threads=" << _engine->get_max_threads();
		os << ")";
		if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
		return os;
	}
	//! \brief Prints the statistics engine's diagnostic top-scored-feature listing (Result_Weighter_Sampled::print_stats()) to \p os. \return true (unconditionally; the statistics engine is asserted non-null)
	virtual bool print_summary(std::ostream& os=std::cout) {assert(_stats); _stats->print_stats(os); return true;}

protected:
	PWeighter _stats; //!< shared online per-feature statistics engine driving proposal sampling (also reachable via get_stats(), e.g. for fst4's warm-up probes)
	PEvaluatorEngine _engine; //!< injected candidate-evaluation engine: sequential in-line (default) or thread-pooled
	DIMTYPE _cap; //!< forward per-step candidate budget (constant-count schedule; see set_cap_frac() for the percentage alternative)
	DIMTYPE _cap_backward=0; //!< >0 = budgeted backward steps; 0 = full backward sweeps (exact floating correction)
	double _cap_frac=0.0; //!< >0 = constant-percentage schedule overrides _cap (see set_cap_frac); forward only
	double _explore; //!< exploration-floor fraction of each proposal batch, clamped to [0,1] at construction (see class docs)
	double _tau; //!< softmax temperature as given at construction; <=0 triggers the per-step robust auto-scale (class docs)
	SamplerMode _sampler; //!< active proposal policy, see SamplerMode

private:
	/*! \brief Full candidate sweep (the Sequential_Step_Straight fallback): evaluates every candidate subset
	           returned by sub's getFirstCandidateSubset()/getNextCandidateSubset() through the injected engine,
	           feeds every evaluation to the statistics engine and (if active) the result tracker, and copies
	           the best-found candidate into \p sub. Used for small pools (pool<=cap), cap==0, generalized steps
	           (g>1) and frozen-mode subsets.
	    \param result set to the best-found candidate's criterion value on success
	    \param sub the working subset; updated in place to the best-found candidate
	    \param crit the criterion each candidate is evaluated against
	    \param g generalization level (candidate subsets differ from \p sub by \p g features)
	    \param pool number of free (forward) or selected (backward) features available to propose from
	    \param os progress/diagnostic messages are written here at NORMAL output detail or above */
	inline bool full_sweep(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE g, const DIMTYPE pool, std::ostream& os);
	/*! \brief Budgeted forward step: proposes exactly \p cap candidate features (see the class docs above for
	           the sampler-mode and exploration-floor mechanics), evaluates the proposed batch through the
	           injected engine, feeds every evaluation to the statistics engine and (if active) the result
	           tracker, and selects the best-scoring proposed feature into \p sub. All fst_rand() draws happen
	           before any candidate is handed to the engine, so proposal sampling is unaffected by threaded
	           evaluation (see the threading paragraph in the class docs).
	    \param result set to the winning candidate's criterion value on success
	    \param sub the working subset; the winning feature is added in place
	    \param crit the criterion each proposed candidate is evaluated against
	    \param pool number of free features available
	    \param cap number of candidates to propose (< pool; the caller has already ruled out the full-sweep fallback)
	    \param os progress/diagnostic messages are written here at NORMAL output detail or above */
	inline bool sampled_forward(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE pool, const DIMTYPE cap, std::ostream& os);
	/*! \brief Budgeted backward step, the removal mirror of sampled_forward (kept as a separate body on
	           purpose — see the comment above its definition): proposes exactly \p cap removal candidates
	           favoring low-scored members, evaluates them through the injected engine, feeds every evaluation
	           to the statistics engine and (if active) the result tracker, and removes the best-scoring
	           proposed member from \p sub.
	    \param result set to the winning candidate's criterion value on success
	    \param sub the working subset; the winning feature is removed in place
	    \param crit the criterion each proposed candidate is evaluated against
	    \param pool number of currently selected features (subset size)
	    \param cap number of removal candidates to propose (< pool)
	    \param os progress/diagnostic messages are written here at NORMAL output detail or above */
	inline bool sampled_backward(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE pool, const DIMTYPE cap, std::ostream& os);
	/*! \brief Folds one completed step's outcome into the order-instrumentation running averages
	           (order_forward/order_backward, count_forward/count_backward) and, at NORMAL verbosity, prints
	           the "F order"/"B order" diagnostic line to \p os (see the class docs' Diagnostics paragraph).
	    \param forward true for a forward step, false for backward
	    \param order 1-based position of the step winner within the proposal (sampled path) or sweep (full path) order
	    \param cnt number of candidates actually evaluated this step
	    \param pool candidate pool size this step drew from
	    \param os the diagnostic line, when printed, is written here */
	inline void note_step(const bool forward, const DIMTYPE order, const DIMTYPE cnt, const DIMTYPE pool, std::ostream& os);
	double frand() {return (double)fst_rand()/((double)FST_RAND_MAX+1.0);} //!< uniform [0,1); fst_rand()-based per the `--seed` contract

	PSubset tmp_cand;                  //!< sampled path: reusable candidate = S + {f} (lazy)

	// instrumentation: at which position in the proposal order do step winners appear.
	// All four counters plus all_evals accumulate over the LIFETIME of this object and are
	// never reset; reuse the same step object across searches only if that is what you want
	// the diagnostic averages to describe.
	DIMTYPE count_forward;             //!< number of completed forward steps folded into order_forward so far (cumulative, never reset)
	DIMTYPE count_backward;            //!< number of completed backward steps folded into order_backward so far (cumulative, never reset)
	RETURNTYPE order_forward;          //!< running average of forward-step winner order, normalized to (0,1] (see note_step(); cumulative, never reset)
	RETURNTYPE order_backward;         //!< running average of backward-step winner order, normalized to (0,1] (see note_step(); cumulative, never reset)
	unsigned long long all_evals;      //!< criterion evaluations performed by this step object so far (cumulative, never reset; 64-bit because a long is 32-bit on LLP64)
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
bool Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(crit);
	assert(_stats);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	if(_stats->get_n()!=sub->get_n()) _stats->reset(sub->get_n()); // first use (a warmed-up engine already matches n and is kept)
	const bool forward=sub->get_forward_mode();
	const DIMTYPE pool = forward ? (DIMTYPE)(sub->get_n()-sub->get_d()) : sub->get_d();
	if(pool==0) return false;
	// frozen-mode subsets carry protected features the raw-flag scans below must not touch —
	// the classic candidate traversal handles them transparently, so fall back to it
	if(sub->get_frozen_mode())
		return full_sweep(result,sub,crit,_generalization_level,pool,os);
	if(forward) {
		// effective per-step cap: constant count by default, ceil(frac*pool) under the
		// constant-percentage schedule
		DIMTYPE cap=_cap;
		if(_cap_frac>0.0) {cap=(DIMTYPE)std::ceil(_cap_frac*(double)pool); if(cap<1) cap=1;}
		if(cap==0 || pool<=cap || _generalization_level>1)
			return full_sweep(result,sub,crit,_generalization_level,pool,os);
		return sampled_forward(result,sub,crit,pool,cap,os);
	}
	// backward: budgeted symmetrically when y_b is set; y_b==0 keeps full backward sweeps
	// and thereby the exact floating correction
	if(_cap_backward==0 || pool<=_cap_backward || _generalization_level>1)
		return full_sweep(result,sub,crit,_generalization_level,pool,os);
	return sampled_backward(result,sub,crit,pool,_cap_backward,os);
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
bool Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER>::full_sweep(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE g, const DIMTYPE pool, std::ostream& os)
{
	const bool forward=sub->get_forward_mode();
	const bool track=parent::result_tracker_active();
	// per-candidate side effects ride the engine's ResultCallback: the sequential engine
	// fires it synchronously inside add_candidate, the threaded engine fires it in
	// add-order on this thread when the batch completes - either way the
	// statistics/tracker/instrumentation see candidates in sweep order
	RETURNTYPE bestval=0;
	DIMTYPE order=0,cnt=0;
	bool best_available=false,stats_ok=true;
	typename EVALUATORENGINE::ResultCallback on_result=[this,track,&bestval,&order,&cnt,&best_available,&stats_ok](const RETURNTYPE val, const PSubset &csub, const DIMTYPE) {
		all_evals++;
		if(!_stats->add(val,csub)) stats_ok=false;
		if(track) parent::_tracker->add(val,csub);
		cnt++;
		if(!best_available || val>bestval) {bestval=val; order=cnt; best_available=true;}
	};
	// the candidate count is needed only by the threaded engine, to size its result
	// storage (same convention as Sequential_Step_Straight)
	const DIMTYPE max_candidates = _engine->is_threaded() ? sub->getNoOfCandidateSubsets(g,false/*reverse*/) : 1;
	_engine->initialize(crit,max_candidates,on_result);
	bool b=sub->getFirstCandidateSubset(g,false/*reverse*/);
	if(!b) return false;
	if(parent::output_detailed()) {std::ostringstream sos; sos << "Step (full)...: sub=" << *sub << std::endl << std::flush; syncout::print(os,sos);}
	if(!_engine->add_candidate(sub)) return false;
	b=sub->getNextCandidateSubset();
	while(b) {
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Step (full)...: sub=" << *sub << std::endl << std::flush; syncout::print(os,sos);}
		if(!_engine->add_candidate(sub)) return false;
		b=sub->getNextCandidateSubset();
	}
	PSubset psub;
	DIMTYPE dummyfeature;
	if(!_engine->getBestResult(result,psub,dummyfeature)) return false;
	if(!stats_ok) return false;
	_stats->flush_batch();
	assert(psub && psub->get_n()==sub->get_n());
	sub->stateless_copy(*psub);
	note_step(forward,order,cnt,pool,os);
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
bool Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER>::sampled_forward(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE pool, const DIMTYPE cap, std::ostream& os)
{
	const DIMTYPE n=sub->get_n();
	const bool track=parent::result_tracker_active();
	assert(cap>0 && pool>cap);
	// free features (forward mode => raw selection flags read literally)
	std::vector<DIMTYPE> freef;
	freef.reserve(pool);
	for(DIMTYPE f=0;f<n;f++) if(!sub->selected_raw(f)) freef.push_back(f);
	assert((DIMTYPE)freef.size()==pool);
	const DIMTYPE want=cap;
	std::vector<DIMTYPE> prop;
	prop.reserve(want);
	std::vector<char> taken(n,0);
	if(_sampler==SAMPLER_TOPK) {
		// hard cutoff: the `cap` highest-scored free features (no exploration floor)
		std::vector<DIMTYPE> byscore(freef);
		std::partial_sort(byscore.begin(),byscore.begin()+want,byscore.end(),
			[this](const DIMTYPE a,const DIMTYPE b){const RETURNTYPE sa=_stats->score(a),sb=_stats->score(b); return sa>sb || (sa==sb && a<b);});
		for(DIMTYPE i=0;i<want;i++) prop.push_back(byscore[i]);
	} else {
		DIMTYPE u = (_sampler==SAMPLER_UNIFORM) ? want : (DIMTYPE)(_explore*(double)want+0.5);
		if(u>want) u=want;
		if(u>0) {
			// exploration floor: uniform draws from the least-evaluated features
			// (SAMPLER_UNIFORM: uniform draws from the WHOLE pool - the stochastic-greedy baseline)
			std::vector<DIMTYPE> bycnt(freef);
			const DIMTYPE le = (_sampler==SAMPLER_UNIFORM) ? pool : std::min<DIMTYPE>(pool,std::max<DIMTYPE>(4*u,u));
			if(le<pool)
				std::partial_sort(bycnt.begin(),bycnt.begin()+le,bycnt.end(),
					[this](const DIMTYPE a,const DIMTYPE b){const auto ca=_stats->count_is(a),cb=_stats->count_is(b); return ca<cb || (ca==cb && a<b);});
			for(DIMTYPE k=0;k<u;k++) { // partial Fisher-Yates over the least-evaluated prefix
				const DIMTYPE j=k+(DIMTYPE)(frand()*(double)(le-k));
				std::swap(bycnt[k],bycnt[j]);
				prop.push_back(bycnt[k]);
				taken[bycnt[k]]=1;
			}
		}
		const DIMTYPE rest=want-u;
		if(rest>0) {
			// weighted draws without replacement ~ softmax(score/tau) over the untaken pool
			std::vector<DIMTYPE> cand;
			cand.reserve(pool-u);
			for(size_t i=0;i<freef.size();i++) if(!taken[freef[i]]) cand.push_back(freef[i]);
			std::vector<double> sc(cand.size());
			for(size_t i=0;i<cand.size();i++) sc[i]=(double)_stats->score(cand[i]);
			double tau=_tau;
			if(tau<=0.0) { // robust self-tuning scale: IQR/1.349 of the current free-feature scores
				std::vector<double> tmp(sc);
				const size_t q1=tmp.size()/4, q3=(3*tmp.size())/4;
				std::nth_element(tmp.begin(),tmp.begin()+q1,tmp.end());
				const double a=tmp[q1];
				std::nth_element(tmp.begin(),tmp.begin()+q3,tmp.end());
				const double b=tmp[q3];
				tau=(b-a)/1.349;
				// floor the temperature (the negated comparison also catches a NaN IQR):
				// division by tau must stay finite. At the floor all scores that are equal
				// give exponent 0, i.e. uniform draws.
				if(!(tau>1e-9)) tau=1e-9;
			}
			const double smax=*std::max_element(sc.begin(),sc.end());
			std::vector<double> w(sc.size());
			double W=0.0;
			for(size_t i=0;i<sc.size();i++) {w[i]=std::exp((sc[i]-smax)/tau); W+=w[i];}
			for(DIMTYPE k=0;k<rest;k++) {
				const double r=frand()*W;
				size_t pick=w.size();
				double acc=0.0;
				for(size_t i=0;i<w.size();i++) {
					if(w[i]<=0.0) continue;
					acc+=w[i];
					if(r<acc) {pick=i; break;}
				}
				if(pick==w.size()) {for(size_t i=w.size();i-->0;) if(w[i]>0.0) {pick=i; break;}} // numeric-drift guard
				if(pick==w.size()) {
					// complete underflow: with tau at the 1e-9 floor, exp((sc-smax)/tau)
					// is +0.0 for every candidate off the exact-max ties; once those are
					// drawn, W==0 and neither the draw loop nor the drift guard (both
					// need w>0) can select. The tau->0 softmax limit over the remnant is
					// argmax(score) - pick it deterministically (lowest index wins ties;
					// w[i]<0 marks already-drawn entries). Without this branch,
					// pick==w.size() slips past the NDEBUG'd assert into cand[pick]/
					// w[pick] - an out-of-bounds read AND write whose heap garbage makes
					// proposals duplicated and dependent on the runtime environment.
					for(size_t i=0;i<w.size();i++) if(w[i]>=0.0 && (pick==w.size() || sc[i]>sc[pick])) pick=i;
				}
				assert(pick<w.size());
				prop.push_back(cand[pick]);
				W-=w[pick];
				w[pick]=-1.0; // drawn sentinel (negative so all no-replacement guards keep skipping it)
			}
		}
	}
	assert((DIMTYPE)prop.size()==want);
	// evaluate the proposed batch: candidate i = S + {prop[i]}; all fst_rand() draws of
	// this step happened above, so fanning the evaluations out to the threaded engine
	// leaves the master RNG stream untouched; the ResultCallback fires in add-order
	// (= proposal order) in both engines, so statistics, tracker and winner are the same
	// whichever engine is injected
	if(!tmp_cand || tmp_cand->get_n()!=n) tmp_cand.reset(new SUBSET(*sub));
	else tmp_cand->stateless_copy(*sub);
	RETURNTYPE bestval=0;
	DIMTYPE bestf=n,order=0,cnt=0;
	bool stats_ok=true;
	typename EVALUATORENGINE::ResultCallback on_result=[this,track,n,&bestval,&bestf,&order,&cnt,&stats_ok](const RETURNTYPE val, const PSubset &csub, const DIMTYPE f) {
		all_evals++;
		if(!_stats->add(val,csub)) stats_ok=false;
		if(track) parent::_tracker->add(val,csub);
		cnt++;
		if(bestf==n || val>bestval) {
			bestval=val;
			bestf=f;
			order=cnt;
		}
	};
	_engine->initialize(crit,want,on_result);
	for(size_t i=0;i<prop.size();i++) {
		const DIMTYPE f=prop[i];
		tmp_cand->select_raw(f);
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Step (sampled)...: +<" << f << "> sub=" << *tmp_cand << std::endl << std::flush; syncout::print(os,sos);}
		if(!_engine->add_candidate(tmp_cand,f)) return false;
		tmp_cand->deselect_raw(f);
	}
	if(!_engine->collect()) return false;
	if(!stats_ok) return false;
	_stats->flush_batch();
	if(bestf==n) return false;
	sub->select_raw(bestf);
	result=bestval;
	note_step(true,order,cnt,pool,os);
	return true;
}

// The backward mirror of sampled_forward, kept as a separate body ON PURPOSE: merging the two
// into one parameterized function would change the order in which each path consumes the
// fst_rand() stream, and with it every seeded result the two paths produce. Differences from
// the forward path:
// - pool = current members of S (raw flags read literally: selected_raw(f) is
//   mode-independent membership; in backward mode select_raw(f) REMOVES f and
//   deselect_raw(f) restores it — Subset::set_forward_mode inverts raw-op meaning),
// - proposal favors LOW scores: topk = lowest-scored members, softmax over -score/tau,
// - exploration floor targets members whose ABSENCE is least observed (count_isnot),
// - candidates are constructed by toggling `sub` in place (matching how the full-sweep
//   candidate traversal mutates `sub` during the step).
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
bool Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER>::sampled_backward(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE pool, const DIMTYPE cap, std::ostream& os)
{
	const DIMTYPE n=sub->get_n();
	const bool track=parent::result_tracker_active();
	assert(cap>0 && pool>cap);
	assert(!sub->get_forward_mode());
	// current members of S (raw membership is mode-independent: _bin[f]>0)
	std::vector<DIMTYPE> memb;
	memb.reserve(pool);
	for(DIMTYPE f=0;f<n;f++) if(sub->selected_raw(f)) memb.push_back(f);
	assert((DIMTYPE)memb.size()==pool);
	const DIMTYPE want=cap;
	std::vector<DIMTYPE> prop;
	prop.reserve(want);
	std::vector<char> taken(n,0);
	if(_sampler==SAMPLER_TOPK) {
		// hard cutoff: the `cap` LOWEST-scored members (removal analogue of the forward top-cut)
		std::vector<DIMTYPE> byscore(memb);
		std::partial_sort(byscore.begin(),byscore.begin()+want,byscore.end(),
			[this](const DIMTYPE a,const DIMTYPE b){const RETURNTYPE sa=_stats->score(a),sb=_stats->score(b); return sa<sb || (sa==sb && a<b);});
		for(DIMTYPE i=0;i<want;i++) prop.push_back(byscore[i]);
	} else {
		DIMTYPE u = (_sampler==SAMPLER_UNIFORM) ? want : (DIMTYPE)(_explore*(double)want+0.5);
		if(u>want) u=want;
		if(u>0) {
			// exploration floor: uniform draws from the members whose absence is least observed
			// (SAMPLER_UNIFORM: uniform draws from ALL members - the stochastic-greedy baseline)
			std::vector<DIMTYPE> bycnt(memb);
			const DIMTYPE le = (_sampler==SAMPLER_UNIFORM) ? pool : std::min<DIMTYPE>(pool,std::max<DIMTYPE>(4*u,u));
			if(le<pool)
				std::partial_sort(bycnt.begin(),bycnt.begin()+le,bycnt.end(),
					[this](const DIMTYPE a,const DIMTYPE b){const auto ca=_stats->count_isnot(a),cb=_stats->count_isnot(b); return ca<cb || (ca==cb && a<b);});
			for(DIMTYPE k=0;k<u;k++) { // partial Fisher-Yates over the least-observed prefix
				const DIMTYPE j=k+(DIMTYPE)(frand()*(double)(le-k));
				std::swap(bycnt[k],bycnt[j]);
				prop.push_back(bycnt[k]);
				taken[bycnt[k]]=1;
			}
		}
		const DIMTYPE rest=want-u;
		if(rest>0) {
			// weighted draws without replacement ~ softmax(-score/tau) over the untaken members
			// (LOW-scoring members are the preferred removal candidates)
			std::vector<DIMTYPE> cand;
			cand.reserve(pool-u);
			for(size_t i=0;i<memb.size();i++) if(!taken[memb[i]]) cand.push_back(memb[i]);
			std::vector<double> sc(cand.size());
			for(size_t i=0;i<cand.size();i++) sc[i]=(double)_stats->score(cand[i]);
			double tau=_tau;
			if(tau<=0.0) { // robust self-tuning scale: IQR/1.349 of the current member scores
				std::vector<double> tmp(sc);
				const size_t q1=tmp.size()/4, q3=(3*tmp.size())/4;
				std::nth_element(tmp.begin(),tmp.begin()+q1,tmp.end());
				const double a=tmp[q1];
				std::nth_element(tmp.begin(),tmp.begin()+q3,tmp.end());
				const double b=tmp[q3];
				tau=(b-a)/1.349;
				// floor the temperature (the negated comparison also catches a NaN IQR):
				// division by tau must stay finite. At the floor all scores that are equal
				// give exponent 0, i.e. uniform draws.
				if(!(tau>1e-9)) tau=1e-9;
			}
			const double smin=*std::min_element(sc.begin(),sc.end());
			std::vector<double> w(sc.size());
			double W=0.0;
			for(size_t i=0;i<sc.size();i++) {w[i]=std::exp((smin-sc[i])/tau); W+=w[i];}
			for(DIMTYPE k=0;k<rest;k++) {
				const double r=frand()*W;
				size_t pick=w.size();
				double acc=0.0;
				for(size_t i=0;i<w.size();i++) {
					if(w[i]<=0.0) continue;
					acc+=w[i];
					if(r<acc) {pick=i; break;}
				}
				if(pick==w.size()) {for(size_t i=w.size();i-->0;) if(w[i]>0.0) {pick=i; break;}} // numeric-drift guard
				if(pick==w.size()) {
					// complete underflow - the backward mirror of the forward branch above:
					// the tau->0 softmax(-score/tau) limit over the remnant is
					// argmin(score); pick it deterministically (lowest index wins ties;
					// w[i]<0 marks already-drawn entries). Prevents the out-of-bounds
					// cand[pick]/w[pick] access described there.
					for(size_t i=0;i<w.size();i++) if(w[i]>=0.0 && (pick==w.size() || sc[i]<sc[pick])) pick=i;
				}
				assert(pick<w.size());
				prop.push_back(cand[pick]);
				W-=w[pick];
				w[pick]=-1.0; // drawn sentinel (negative so all no-replacement guards keep skipping it)
			}
		}
	}
	assert((DIMTYPE)prop.size()==want);
	// evaluate the proposed batch in place: candidate i = S - {prop[i]}
	// (backward-mode raw ops: select_raw removes, deselect_raw restores; the engine
	// copies the candidate state inside add_candidate, so toggling `sub` back right
	// after the call is safe in the threaded engine too)
	RETURNTYPE bestval=0;
	DIMTYPE bestf=n,order=0,cnt=0;
	bool stats_ok=true;
	typename EVALUATORENGINE::ResultCallback on_result=[this,track,n,&bestval,&bestf,&order,&cnt,&stats_ok](const RETURNTYPE val, const PSubset &csub, const DIMTYPE f) {
		all_evals++;
		if(!_stats->add(val,csub)) stats_ok=false;
		if(track) parent::_tracker->add(val,csub);
		cnt++;
		if(bestf==n || val>bestval) {
			bestval=val;
			bestf=f;
			order=cnt;
		}
	};
	_engine->initialize(crit,want,on_result);
	for(size_t i=0;i<prop.size();i++) {
		const DIMTYPE f=prop[i];
		sub->select_raw(f);
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Step (sampled)...: -<" << f << "> sub=" << *sub << std::endl << std::flush; syncout::print(os,sos);}
		if(!_engine->add_candidate(sub,f)) return false;
		sub->deselect_raw(f);
	}
	if(!_engine->collect()) return false;
	if(!stats_ok) return false;
	_stats->flush_batch();
	if(bestf==n) return false;
	sub->select_raw(bestf);
	result=bestval;
	note_step(false,order,cnt,pool,os);
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class WEIGHTER>
void Sequential_Step_Sampled<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,WEIGHTER>::note_step(const bool forward, const DIMTYPE order, const DIMTYPE cnt, const DIMTYPE pool, std::ostream& os)
{
	assert(cnt>0);
	const RETURNTYPE r=(RETURNTYPE)order/(RETURNTYPE)cnt;
	if(forward) {
		count_forward++;
		order_forward=((count_forward-1)*order_forward+r)/count_forward;
		if(parent::output_normal()) {std::ostringstream sos; sos << std::endl << "F order = " << r << "(" << order << ") avgorder " << order_forward << " count = " << count_forward << " all_evals = " << all_evals << " pool = " << pool << " proposed = " << cnt << std::endl << std::flush; syncout::print(os,sos);}
	} else {
		count_backward++;
		order_backward=((count_backward-1)*order_backward+r)/count_backward;
		if(parent::output_normal()) {std::ostringstream sos; sos << std::endl << "B order = " << r << "(" << order << ") avgorder " << order_backward << " count = " << count_backward << " all_evals = " << all_evals << " pool = " << pool << " proposed = " << cnt << std::endl << std::flush; syncout::print(os,sos);}
	}
}

} // namespace
#endif // FSTSEARCHSEQSTEPSAMPLED_H ///:~
