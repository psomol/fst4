#ifndef FSTCANDIDATEEVALUATOR_H
#define FSTCANDIDATEEVALUATOR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    candidate_evaluator.hpp
   \brief   Abstract candidate-evaluation strategy: the seam through which one
            search algorithm body evaluates candidates either sequentially
            in-line or concurrently in a thread pool
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0

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
#include <functional>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"

namespace FST {

/*! \brief Abstract candidate-evaluation strategy (sequential-inline vs. thread-pooled)

	Search algorithms delegate candidate subset evaluation to an injected
	implementation of this interface, so that one algorithm body serves both the
	sequential and the threaded execution model. Which one is in effect is a
	property of the engine object passed to the search's constructor, not of the
	search class.

	Two modes of use:

	- \b batch: initialize() once per candidate group, add_candidate() per
	  candidate, then collect() or getBestResult(). The optional per-result
	  callback fires exactly once per candidate with its criterion value:
	  the sequential implementation fires it synchronously inside
	  add_candidate(), so evaluation stays interleaved with the caller's own
	  output, the threaded implementation fires it for all candidates in
	  add-order while completing the batch. Ties on the best value are broken
	  in favor of the earliest added candidate in both implementations.

	- \b streaming: evaluate_stream() pulls candidates from a generator until
	  it is exhausted - for searches whose candidate count is too large to
	  materialize (exhaustive, Monte Carlo). Result tracking - including
	  per-worker tracker clones and their final join in the threaded case -
	  is handled inside the implementation.

	\warning The threaded implementation clones the supplied criterion once per
	worker and re-uses the clones as long as the same criterion object is
	supplied. The clones' state is not re-synchronized with the original
	criterion afterwards, so an engine must not be carried across a change of the
	criterion's context (typically a change of the data split) - see the class
	documentation of Candidate_Evaluator_Threaded.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Candidate_Evaluator {
public:
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the criterion this engine evaluates candidates against
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a candidate feature subset
	typedef std::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker; //!< shared pointer to the result tracker fed during evaluate_stream()
	//! fills sub with the next candidate, returns false when no candidate remains (in the threaded implementation calls are serialized under a lock)
	typedef std::function<bool(SUBSET&)> Generator;
	//! fired once per evaluated batch candidate: (criterion value, candidate, feature tag as passed to add_candidate)
	typedef std::function<void(const RETURNTYPE, const PSubset&, const DIMTYPE)> ResultCallback;
	//! fired when a best-so-far improves during evaluate_stream(): (worker index, or -1 if fired from the master/sequential replay; 0-based ordinal of the candidate in GENERATION order; criterion value; candidate). Both engines fire this on the master thread in generation order, with index -1
	//! NOTE the ordinal is unsigned long long, NOT unsigned long: on LLP64 platforms (Windows) unsigned long is 32 bits and a long streaming search (Monte Carlo with a large trial count) would wrap it
	typedef std::function<void(const int, const unsigned long long, const RETURNTYPE, const PSubset&)> BestCallback;
	//! returns true if (value,subset) is better than (best value,best subset) - encodes per-search tie-breaking for evaluate_stream()
	typedef std::function<bool(const RETURNTYPE, const PSubset&, const RETURNTYPE, const PSubset&)> BetterPredicate;

	virtual ~Candidate_Evaluator() {}

	virtual unsigned int get_max_threads() const =0; //!< 1 in the sequential implementation
	virtual bool is_threaded() const =0; //!< true for the threaded engine (any `--threads` value), false for the sequential one -- distinguishes the engine kind, not whether get_max_threads()>1

	// batch mode
	// NOTE: crit is expected to be already initialize()d before use here
	virtual void initialize(const PCriterion &crit, const DIMTYPE max_candidates, ResultCallback on_result=ResultCallback()) =0; //!< starts a new batch; max_candidates must bound the number of add_candidate() calls to follow
	virtual bool add_candidate(const PSubset &sub, const DIMTYPE feature=0) =0; //!< adds one candidate to the current batch; \p sub is copied, \p feature is an opaque tag handed back by the callbacks. Returns false on evaluation failure (sequential) or when more than the announced max_candidates candidates are added (threaded)
	virtual bool collect() =0; //!< completes the batch (threaded: waits for the workers, fires the pending per-result callbacks in add-order); returns false if no candidate was successfully evaluated. Calling it twice on the same batch is harmless
	virtual bool getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) =0; //!< completes the batch first, then hands out its best candidate (\p sub is set to the engine's own subset object, its contents are not copied and are overwritten by the following batch - copy them if they must outlive it); earliest added wins ties; false if the batch evaluated nothing or an evaluation failed

	// streaming mode
	// NOTE: crit is expected to be already initialize()d before use here
	/*! \brief Evaluates candidates produced by \p generator until it reports exhaustion, keeping the best one.
	    \param generator called to fill the next candidate; false from it ends the stream
	    \param crit the criterion to evaluate with; must be initialize()d
	    \param tracker result tracker to log every evaluated candidate into, or null for no tracking
	    \param better the search's own "is strictly better" predicate, deciding both improvement and tie-breaking
	    \param on_best fired on each improvement; may be empty
	    \param result receives the best criterion value; meaningful only if the call returns true
	    \param bestsub non-null subset dimensioned for the full feature set, receiving the best candidate
	    \return false if no candidate was evaluated successfully (empty stream or criterion failure) */
	virtual bool evaluate_stream(Generator generator, const PCriterion &crit, const PResultTracker tracker, BetterPredicate better, BestCallback on_best, RETURNTYPE &result, const PSubset bestsub) =0;

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the engine; base default writes nothing

	/*! \brief Drops any cached per-worker criterion clones, so that the next batch or stream
	    re-clones from the criterion it is handed.

	    The threaded implementation clones the criterion once per worker and keeps the clones
	    across batches while the same criterion object is supplied (see the warning above).
	    Call this whenever the ORIGINAL criterion's evaluation context has changed behind the
	    clones' back - its data accessor moved to another data split (the next trial of a
	    repeated-trials protocol), a filter model was re-estimated on new data, a wrapper's
	    classifier or regressor was re-tuned - and the same criterion object is about to be
	    evaluated again. Without it the workers keep evaluating in the old context: a trial
	    protocol would repeat its first trial in every trial. The sequential implementation
	    evaluates on the original and has nothing to drop, hence the empty default. Must not
	    be called while a batch is open. */
	virtual void invalidate_clones() {}
};

//! Streams an engine's print() to \p os.
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
inline std::ostream& operator<<(std::ostream& os, const Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& ce) {
	return ce.print(os);
}

} // namespace
#endif // FSTCANDIDATEEVALUATOR_H ///:~
