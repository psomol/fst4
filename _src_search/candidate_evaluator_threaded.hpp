#ifndef FSTCANDIDATEEVALUATORTHREADED_H
#define FSTCANDIDATEEVALUATORTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    candidate_evaluator_threaded.hpp
   \brief   Implements concurrent evaluation of a criterion on a set of subset candidates
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    March 2011 (last updated July 2026)
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

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <cassert>
#include <iostream>
#include <list>
#include <vector>
#include <cstddef>
#include <atomic>
#include <exception>
#include "error.hpp"
#include "global.hpp"
#include "thread_pool.hpp"
#include "result_tracker.hpp"
#include "candidate_evaluator.hpp"

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
CLASSIFIER   - class implementing interface defined in abstract class Classifier 
EVALUATOR    - class implementing interface defined in abstract class Sequential_Step 
DISTANCE     - class implementing interface defined in abstract class Distance 
DATAACCESSOR - class implementing interface defined in abstract class Data_Accessor 
INTERVALCONTAINER - class of class type Data_Intervaller 
CONTAINER    - STL container of class type Data_Interval  
========================================================================== */

namespace FST {

//! Lifecycle state of a worker's thread-local criterion clone (see Candidate_Evaluator_Threaded::ThreadLocal)
enum CriterionStatus {
  CRIT_NONEXISTENT=0, //!< no clone allocated yet for this worker slot
  CRIT_READY=1        //!< a clone exists and is not currently evaluating
};

/*! \brief Implements concurrent evaluation of a criterion on a set of subset candidates

 Concurrently evaluates subset candidates using clones of the supplied criterion and
 at most max_threads worker threads. It implements the abstract Candidate_Evaluator
 strategy interface, so a search can take either this or Candidate_Evaluator_Sequential
 through the same seam and behave identically apart from speed.

 Two modes are offered, as in the interface:
 \li BATCH mode - initialize(), a sequence of add_candidate() calls, then collect()
     or getBestResult(); getBestResult() returns the candidate that yielded the highest
     criterion value, getFirstResult()/getNextResult() iterate over ALL results of the
     batch in add-order. The optional per-result callback fires for all candidates in
     add-order when the batch completes.
 \li STREAMING mode - evaluate_stream() consumes candidates from a generator until it
     is exhausted and maintains the best-so-far itself.

 \warning The engine CLONES the criterion (once per worker) the first time it sees a
 given criterion object, and reuses those clones for every later batch or stream in
 which the same criterion object is passed again; a clone is created anew only when
 the criterion pointer handed to initialize()/evaluate_stream() differs from the one
 seen before. The clones' internal state is NOT re-synchronized with the original in
 between. If the original's evaluation context changes behind its back - the typical
 case being a wrapper criterion whose data accessor has moved to another data split -
 the clones keep evaluating in the old context and the results are silently wrong.
 When that can happen, call invalidate_clones() before the next batch or stream (or
 re-create the engine, or the search step that owns it) so that a fresh set of clones
 is made from the criterion in its new context.

 \note Batch workers form a PERSISTENT pool: spawned once at the first initialize(),
 parked on a condition variable between batches, joined only in the destructor. Batch
 completion is detected by a completion counter, not by joining, so a batch costs two
 mutexed counter sweeps instead of max_threads thread spawns plus joins. This matters
 for users that submit many tiny batches (Branch & Bound submits one per node
 expansion), which the spawn-per-batch alternative made slower threaded than
 sequential. Scheduling, result placement (cres[] indexed by add-order) and callback
 order do not depend on it, so the determinism guarantees below hold either way.

 \note evaluate_stream(): the MASTER thread generates all candidates into a bounded
 ring (so any randomness in the generator draws from the master's fst_rand stream,
 exactly as sequential execution would), worker threads evaluate them concurrently
 with criterion clones, and the master replays completed slots strictly in generation
 order - tracker adds, best-updates (identical tie-breaking) and on_best callbacks all
 fire on the master. Results, tracker contents and callback sequence are therefore the
 same as with Candidate_Evaluator_Sequential at the same seed, for any thread count.
 Note that letting workers pull from the generator instead would make them draw from
 worker-thread_local RNG streams, which no seed can make reproducible.

 \note Failure semantics match the sequential engine: a criterion evaluate()
 returning false marks the batch/stream failed (collect()/getBestResult()/
 evaluate_stream() then return false, per-result callbacks do not fire), and an
 exception thrown by a criterion on a worker is captured and rethrown on the
 MASTER thread after the workers join - never left to escape the worker, which
 would std::terminate the process with no FST diagnostic.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Candidate_Evaluator_Threaded : public Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> {
public:
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< the abstract Candidate_Evaluator interface this class implements
	typedef std::shared_ptr<CRITERION> PCriterion;               //!< shared pointer to the criterion this engine evaluates candidates against
	typedef std::shared_ptr<SUBSET> PSubset;                     //!< shared pointer to a candidate feature subset
	typedef std::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker; //!< shared pointer to the result tracker fed during evaluate_stream()
	typedef typename parent::Generator Generator;                 //!< see Candidate_Evaluator::Generator
	typedef typename parent::ResultCallback ResultCallback;       //!< see Candidate_Evaluator::ResultCallback
	typedef typename parent::BestCallback BestCallback;           //!< see Candidate_Evaluator::BestCallback
	typedef typename parent::BetterPredicate BetterPredicate;     //!< see Candidate_Evaluator::BetterPredicate

	//! \param max_threads size of the persistent worker pool (lazily spawned at the first initialize() call, parked between batches, joined only in the destructor)
	explicit Candidate_Evaluator_Threaded(const unsigned int max_threads=2) : _max_threads(max_threads), initialized(false), working(false), tlocal(max_threads), _processed_candidates(0), _available_candidates(0), _completed_candidates(0), _pool_started(false), _pshutdown(false), _reading_result(0), _callbacks_fired(false), _batch_failed(false) {
		assert(max_threads>0);
		for(unsigned int i=0;i<_max_threads;i++) tlocal[i].status=CRIT_NONEXISTENT; // not available, not allocated
		notify("Candidate_Evaluator_Threaded constructor."); // deliberately thread-free: the persistent batch pool starts lazily at first initialize()
	}

	virtual unsigned int get_max_threads() const override {return _max_threads;}
	virtual bool is_threaded() const override {return true;}
	virtual ~Candidate_Evaluator_Threaded() {
		if(_pool_started) { // park-protocol shutdown: wake everyone, workers drain any pending work, then exit
			{
				std::unique_lock<std::mutex> lock(mutex_candidate_getter);
				_pshutdown=true;
			}
			condition_available.notify_all();
			for(std::thread &w : _pworkers) if(w.joinable()) w.join();
		}
		notify("Candidate_Evaluator_Threaded destructor.");
	}

	//! \brief Opens a new batch to be evaluated against \p crit, which must already be initialize()d.
	//! \p max_candidates must be greater or equal than the number of candidates that will be added -
	//! the result storage is sized here once, so that workers never race a vector reallocation; it
	//! must be nonzero. See the criterion-clone warning in the class documentation.
	virtual void initialize(const PCriterion &crit, const DIMTYPE max_candidates, ResultCallback on_result=ResultCallback()) override;
	virtual bool add_candidate(const PSubset &sub, const DIMTYPE feature=0) override; //!< adds candidate subset to the evaluation pool; returns false once the result storage reserved by initialize() is full, i.e. when more candidates are added than max_candidates announced

	// the two counters below are read without taking the mutex: they are progress indicators only,
	// and a value read while workers are running may already be out of date when it is returned

	// any of the following readers first waits the threads to finish the complete job. Further calls to add_candidate will be treated as part of a new job
	virtual bool collect() override; //!< completes the batch: waits for all evaluations and fires pending per-result callbacks in add-order
	virtual bool getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) override; //!< completes the batch, then returns (without copying sub contents) the candidate that yielded the highest criterion value; ties go to the candidate added first
	bool getFirstResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature); //!< completes the batch and starts iteration over ALL of its results in ADD-ORDER (not by value); returns a pointer, does not copy sub contents
	bool getNextResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature); //!< advances the getFirstResult() iteration by one result; false once the batch is exhausted

	bool results_to_tracker(const PResultTracker tracker); //!< completes the batch and feeds every candidate's (value,subset) into \p tracker in add-order; false if the batch was empty or failed

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool evaluate_stream(Generator generator, const PCriterion &crit, const PResultTracker tracker, BetterPredicate better, BestCallback on_best, RETURNTYPE &result, const PSubset bestsub) override; //!< streaming mode: the master thread calls \p generator until exhaustion, workers evaluate concurrently, the master replays the results in generation order (see the class notes)
	virtual void invalidate_clones() override; //!< drops the cached per-worker criterion clones (see Candidate_Evaluator::invalidate_clones()); the next initialize()/evaluate_stream() re-clones from the criterion it is handed

	virtual std::ostream& print(std::ostream& os) const override {os << "Candidate_Evaluator_Threaded(max_threads="<<_max_threads<<")"; return os;}
protected:
	const unsigned int _max_threads; //!< size of the persistent worker pool, as given to the constructor
	PCriterion template_crit; //!< to detect change of employed criterion inbetween search() calls
	bool initialized; //!< true after the first initialize() call (guards clone (re)creation logic)
	bool working;     //!< true while a batch is open (between initialize()/add_candidate() and finish_batch())

	//! thread-local storage of current subset candidate, criterion clone, and tracker clone
	typedef struct {
		PCriterion crit;       //!< this worker's private clone of the criterion (created lazily, reused across batches while the criterion pointer is unchanged)
		CriterionStatus status; //!< CRIT_NONEXISTENT or CRIT_READY (see CriterionStatus)
	} ThreadLocal;
	std::vector<ThreadLocal> tlocal; //!< per-worker criterion clone + status, one entry per worker index; sized once in the constructor, never resized (no reallocation races)

	//! persistent batch-worker pool: workers spawn ONCE at first initialize() and PARK on
	//! condition_available between batches instead of being joined and respawned per batch -
	//! the per-batch overhead that makes many-small-batch users (Branch & Bound submits one
	//! batch per node expansion) slower threaded than sequential. Batch completion is
	//! signalled by the _completed_candidates counter (condition_batch_done), never by
	//! joining; workers join only in the destructor. Result placement (cres[] by add-order
	//! index) and callback order do not depend on the pooling.
	std::vector<std::thread> _pworkers;
	void pooled_worker(unsigned int tidx); //!< the persistent worker loop: park, claim, evaluate, count, repeat
	DIMTYPE _processed_candidates; //!< candidates with finished processing, or being currently processed
	DIMTYPE _available_candidates; //!< to sychronize workers with adding new candidates to cres[]
	DIMTYPE _completed_candidates; //!< candidates whose result is stored - batch done when it reaches _available_candidates
	bool _pool_started; //!< the persistent pool exists (lazy spawn at first initialize())
	bool _pshutdown;    //!< destructor flag: workers drain pending work and exit

	std::mutex mutex_candidate_getter; //!< guards the candidate counters (workers claim/complete, master adds/waits)
	std::condition_variable condition_available; //!< wakes parked workers: new candidate available (or shutdown)
	std::condition_variable condition_batch_done; //!< wakes the master: a candidate completed (finish_batch waits on the counter)

	//! temporary storage of subset candidate evaluation results
	typedef struct {
		RETURNTYPE result; //!< the criterion value obtained for sub
		PSubset sub;        //!< the candidate subset (owned copy, not shared with the caller's subset)
		DIMTYPE feature;    //!< the optional feature tag passed to add_candidate()
	} CandidateResult;
	std::vector<CandidateResult> cres; //!< one entry per batch candidate, indexed by add-order (the position add_candidate() assigned it)

	DIMTYPE _reading_result; //!< iterating index for get*Result readers

	ResultCallback _on_result; //!< optional per-candidate callback, fired in add-order when the batch completes
	bool _callbacks_fired; //!< ensures _on_result fires exactly once per batch
	void finish_batch(); //!< stops and joins workers, then fires pending per-result callbacks (once per batch)

	std::atomic<bool> _batch_failed; //!< batch-mode twin of the sequential engine's _failed: a worker saw evaluate()==false
	std::exception_ptr _worker_exception; //!< first exception a criterion threw on a worker; rethrown on the master after join
	std::mutex mutex_worker_exception; //!< guards _worker_exception (shared by batch and streaming workers; the modes never overlap)
	void record_worker_exception() {std::unique_lock<std::mutex> lock(mutex_worker_exception); if(!_worker_exception) _worker_exception=std::current_exception();} //!< captures the currently-handled exception (first one wins) for rethrow on the master by finish_batch()/evaluate_stream(); called from a worker's catch(...) block, never lets an exception escape the worker

	//! shared state of the ordered streaming mode (evaluate_stream): a bounded ring of
	//! master-generated candidate slots, evaluated by workers, replayed in order by the master.
	//! A slot travels SLOT_EMPTY (free for the master to generate into) -> SLOT_PENDING (queued
	//! in _spending, not claimed yet) -> SLOT_RUNNING (a worker is evaluating it) -> SLOT_DONE
	//! or SLOT_FAILED (the criterion returned false or threw), and back to SLOT_EMPTY once the
	//! master has replayed it. All transitions happen under mutex_stream.
	enum StreamSlotState {SLOT_EMPTY=0, SLOT_PENDING, SLOT_RUNNING, SLOT_DONE, SLOT_FAILED};
	void stream_worker(unsigned int tidx); //!< claims pending slots, evaluates with the worker's criterion clone
	std::vector<PSubset> _sslot_sub;                //!< ring: master-generated candidates
	std::vector<RETURNTYPE> _sslot_val;             //!< ring: evaluation results
	std::vector<unsigned char> _sslot_state;        //!< ring: StreamSlotState per slot
	std::list<std::size_t> _spending;               //!< pending slot indices in generation order
	std::mutex mutex_stream;                        //!< guards the ring bookkeeping
	std::condition_variable _scv_work;              //!< wakes workers on new pending slots / shutdown
	std::condition_variable _scv_done;              //!< wakes the master when a slot completes
	bool _sstop;                                    //!< shutdown flag for stream workers
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::initialize(const PCriterion &crit, const DIMTYPE max_candidates, ResultCallback on_result)
{ // max_candidates sizes the result storage up front: it must be at least the number of
  // candidates this batch will receive, so that add_candidate() never reallocates cres
  // while workers hold references into it
	assert(crit);
	assert(max_candidates>0);
	_on_result=on_result;
	_callbacks_fired=false;

	if(working) { // initialize() called while a batch is still open: outstanding evaluations
	              // are allowed to complete (workers drain the queue), their results are
	              // discarded and their callbacks never fire
		std::unique_lock<std::mutex> lock(mutex_candidate_getter);
		condition_batch_done.wait(lock, [this]{return _completed_candidates==_available_candidates;});
		working=false;
	}
	// workers are parked here (nothing pending), so the master may safely reset
	// counters, resize cres and touch the clone array
	_processed_candidates=0;
	_available_candidates=0;
	_completed_candidates=0;
	_reading_result=0;
	working=true;
	_batch_failed=false;
	_worker_exception=nullptr;

	if(!initialized || crit!=template_crit) {// criterion template changed (or first use): invalidate the per-thread clone array
		template_crit=crit;
		for(unsigned int i=0;i<_max_threads;i++) {tlocal[i].crit.reset(); tlocal[i].status=CRIT_NONEXISTENT;} // 0=not available, not allocated
	}
	// master-side clone creation for ALL workers: the persistent pool decouples worker
	// index from candidate index, so there is no way to tell in add_candidate() which
	// workers will run and clone lazily for those only. Clones are cached across batches
	// via status while the criterion pointer is unchanged, so this pays only at first use
	// or on a criterion change (see the staleness warning in the class documentation)
	for(unsigned int i=0;i<_max_threads;i++) if(tlocal[i].status==CRIT_NONEXISTENT) {tlocal[i].crit=clone_unique(*template_crit); tlocal[i].status=CRIT_READY;}
	if((DIMTYPE)(cres.size())<max_candidates) cres.resize(max_candidates);

	if(!_pool_started) { // lazy spawn, ONCE per engine lifetime; workers park between batches
		_pshutdown=false;
		_pworkers.reserve(_max_threads);
		for(unsigned int i=0;i<_max_threads;i++) _pworkers.emplace_back(&Candidate_Evaluator_Threaded::pooled_worker,this,i);
		_pool_started=true;
	}

	initialized=true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::pooled_worker(unsigned int tidx)
{ // the persistent worker loop: parks on condition_available between batches, exits only
  // at destructor shutdown (draining any pending candidates first). Result visibility to
  // the master follows from the mutex acquire/release around the completion counter.
	assert(tidx<_max_threads);
	for(;;)
	{
		DIMTYPE idx;
		{
			std::unique_lock<std::mutex> lock(mutex_candidate_getter);
			condition_available.wait(lock, [this]{return _pshutdown || _processed_candidates<_available_candidates;});
			if(_processed_candidates==_available_candidates) return; // shutdown and nothing pending
			idx=_processed_candidates++;
		}
		if(!_batch_failed) { // after a failure the rest of the batch drains unevaluated
			CandidateResult &cr=cres[idx];
			assert(tlocal[tidx].crit); // clones exist for every worker since initialize()
			bool ok=false;
			try {
				ok=tlocal[tidx].crit->evaluate(cr.result,cr.sub);
			} catch(...) {
				record_worker_exception(); // rethrown on the master in finish_batch(); escaping the worker would std::terminate
			}
			if(!ok) _batch_failed=true; // the sequential engine's _failed semantics
		}
		{
			std::unique_lock<std::mutex> lock(mutex_candidate_getter);
			++_completed_candidates;
		}
		condition_batch_done.notify_one(); // the master is the only waiter
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::add_candidate(const PSubset &sub, const DIMTYPE feature)
{
	assert(initialized);
	assert(sub);
	assert(sub->get_d()>0);

	DIMTYPE local_available;
	{
		std::unique_lock<std::mutex> lock(mutex_candidate_getter);
		if(!working) {//start with new group of candidates (implicit re-arm after collect();
		              //clones and the persistent pool are guaranteed by the last initialize())
			_processed_candidates=0;
			_available_candidates=0;
			_completed_candidates=0;
			working=true;
			_callbacks_fired=false;
			_batch_failed=false;
			_worker_exception=nullptr;
		}

		if(_available_candidates>=cres.size()) return false;
		local_available = _available_candidates;
	}

	if(!cres[local_available].sub || cres[local_available].sub->get_n()!=sub->get_n()) cres[local_available].sub.reset(new SUBSET(sub->get_n()));
	cres[local_available].sub->stateless_copy(*sub);
	cres[local_available].feature=feature;

	{
		std::unique_lock<std::mutex> lock(mutex_candidate_getter);
		++_available_candidates;
	}
	condition_available.notify_one(); // wakes one parked persistent worker (busy workers re-check the counter anyway)
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::finish_batch()
{// waits for the batch to complete (the persistent workers park themselves once the queue
 // drains - no join), then fires pending per-result callbacks (once, in add-order). The
 // mutexed completion-counter wait is what makes the workers' writes to cres[] visible
 // here: every worker releases the mutex after incrementing the counter, the master
 // acquires it before reading
	if(working) {
		{
			std::unique_lock<std::mutex> lock(mutex_candidate_getter);
			condition_batch_done.wait(lock, [this]{return _completed_candidates==_available_candidates;});
		}
		working=false;
	}
	if(_worker_exception) { // rethrow a worker-thread criterion exception on the master (fst4 main catches it like any sequential failure)
		std::exception_ptr e=_worker_exception;
		_worker_exception=nullptr;
		_callbacks_fired=true; // a failed batch must not fire per-result callbacks
		std::rethrow_exception(e);
	}
	if(!_callbacks_fired) {
		_callbacks_fired=true;
		if(_on_result && !_batch_failed) for(DIMTYPE i=0;i<_available_candidates;i++) _on_result(cres[i].result,cres[i].sub,cres[i].feature);
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::collect()
{
	finish_batch();
	return _available_candidates>0 && !_batch_failed;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature)
{// returns whichever candidate yielded the highest criterion value (waits for all threads to
 // finish). The strict > below means that among equal values the EARLIEST added candidate wins
	finish_batch();
	if(_available_candidates==0 || _batch_failed) return false;
	DIMTYPE bestidx=0;
	for(DIMTYPE i=1;i<_available_candidates;i++) if(cres[i].result>cres[bestidx].result) bestidx=i;
	sub=cres[bestidx].sub;
	result=cres[bestidx].result;
	feature=cres[bestidx].feature;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::getFirstResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature)
{// starts iteration over ALL results of the batch in ADD-ORDER, returning the first one
 // (waits for all threads to finish); use getBestResult() to get the best-valued candidate
	if(_available_candidates==0 || cres.size()==0) return false;
	finish_batch();
	if(_batch_failed) return false;
	_reading_result=0;
	sub=cres[_reading_result].sub;
	result=cres[_reading_result].result;
	feature=cres[_reading_result].feature;
	_reading_result=1;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::getNextResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) 
{// returns the next result in ADD-ORDER, false once the batch is exhausted (getFirstResult()
 // has already waited for the threads, so no wait is needed here)
	if(working || _batch_failed || _reading_result>=_available_candidates) return false;
	assert(_reading_result<cres.size());
	sub=cres[_reading_result].sub;
	result=cres[_reading_result].result;
	feature=cres[_reading_result].feature;
	++_reading_result;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::results_to_tracker(const PResultTracker tracker)
{
	if(!tracker || _available_candidates==0) return false;
	finish_batch();
	if(_batch_failed) return false; // sequential semantics: a failed batch tracks nothing
	for(DIMTYPE i=0;i<_available_candidates;i++) tracker->add(cres[i].result,cres[i].sub);
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::stream_worker(unsigned int tidx)
{
	assert(tidx<_max_threads);
	assert(tlocal[tidx].crit);
	PCriterion crit=tlocal[tidx].crit;
	for(;;)
	{
		std::size_t idx;
		{
			std::unique_lock<std::mutex> lock(mutex_stream);
			_scv_work.wait(lock, [this]{return _sstop || !_spending.empty();});
			if(_spending.empty()) return; // _sstop set and no work left
			idx=_spending.front(); _spending.pop_front();
			assert(_sslot_state[idx]==SLOT_PENDING);
			_sslot_state[idx]=SLOT_RUNNING;
		}
		RETURNTYPE val=RETURNTYPE();
		bool ok=false;
		try {
			ok=crit->evaluate(val,_sslot_sub[idx]); // concurrent part: criterion clone on the master-generated candidate
		} catch(...) {
			record_worker_exception(); // slot goes SLOT_FAILED below; the master rethrows after join
		}
		{
			std::unique_lock<std::mutex> lock(mutex_stream);
			_sslot_val[idx]=val;
			_sslot_state[idx]= ok? SLOT_DONE : SLOT_FAILED;
		}
		_scv_done.notify_one();
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::evaluate_stream(Generator generator, const PCriterion &crit, const PResultTracker tracker, BetterPredicate better, BestCallback on_best, RETURNTYPE &result, const PSubset bestsub)
{
	assert(generator);
	assert(crit);
	assert(better);
	assert(bestsub);
	assert(bestsub->get_n()>0);
	assert(!working); // batch and streaming modes must not overlap
	const DIMTYPE n=bestsub->get_n();

	// (re)create criterion clones only if the master criterion object changed - existing
	// clones are reused otherwise (see the criterion-clone warning in the class documentation)
	if(crit!=template_crit) {
		for(unsigned int i=0;i<_max_threads;i++) {tlocal[i].crit.reset(); tlocal[i].status=CRIT_NONEXISTENT;}
		template_crit=crit;
	}
	for(unsigned int i=0;i<_max_threads;i++) if(tlocal[i].status==CRIT_NONEXISTENT) {tlocal[i].crit=clone_unique(*template_crit); tlocal[i].status=CRIT_READY;}

	// ordered streaming (see the class note): the master generates candidates into a
	// bounded ring and replays completed slots strictly in generation order, so the
	// results equal those of the sequential engine at the same seed, for any thread count
	const std::size_t W=4*(std::size_t)_max_threads; // ring capacity: enough to keep workers fed while the master replays
	if(_sslot_sub.size()<W) _sslot_sub.resize(W);
	for(std::size_t i=0;i<W;i++) if(!_sslot_sub[i] || _sslot_sub[i]->get_n()!=n) _sslot_sub[i].reset(new SUBSET(n));
	_sslot_val.assign(W,RETURNTYPE());
	_sslot_state.assign(W,(unsigned char)SLOT_EMPTY);
	_spending.clear();
	_sstop=false;
	_worker_exception=nullptr;

	ThreadPool stp(_max_threads);
	for(unsigned int i=0;i<_max_threads;i++) {
		Candidate_Evaluator_Threaded *self=this; unsigned int tidx=i;
		stp.go([self,tidx]{self->stream_worker(tidx);});
	}

	bool best_available=false;
	bool failed=false;
	bool gen_done=false;
	// 64-bit on every platform ON PURPOSE: Windows is LLP64 (unsigned long is 32 bits there),
	// and head/tail index the ring by head%W — a wrap would desynchronize the slot mapping
	// whenever W is not a power of two, silently corrupting the ordered replay
	unsigned long long head=0, tail=0; // generation-ordered window of in-flight slots: [head,tail)
	unsigned long long ordinal=0;      // 0-based generation ordinal of the slot being replayed
	for(;;) {
		// fill phase: generate on the MASTER thread (master RNG stream) while the ring has room
		while(!gen_done && tail-head<W) {
			const std::size_t t=(std::size_t)(tail%W);
			assert(_sslot_state[t]==SLOT_EMPTY);
			if(!generator(*_sslot_sub[t])) {gen_done=true; break;}
			{
				std::unique_lock<std::mutex> lock(mutex_stream);
				_sslot_state[t]=SLOT_PENDING;
				_spending.push_back(t);
			}
			++tail;
			_scv_work.notify_one();
		}
		if(head==tail) break; // gen_done and everything replayed
		// reap phase: wait for the OLDEST slot, then replay it in generation order
		const std::size_t h=(std::size_t)(head%W);
		{
			std::unique_lock<std::mutex> lock(mutex_stream);
			_scv_done.wait(lock, [this,h]{return _sslot_state[h]==SLOT_DONE || _sslot_state[h]==SLOT_FAILED;});
			if(_sslot_state[h]==SLOT_FAILED) {failed=true;}
		}
		if(failed) break; // sequential semantics: candidates after a failed evaluation are neither tracked nor reported
		const RETURNTYPE val=_sslot_val[h];
		const PSubset &cand=_sslot_sub[h];
		if(tracker) tracker->add(val,cand);
		if(!best_available || better(val,cand,result,bestsub)) {
			result=val;
			bestsub->stateless_copy(*cand);
			best_available=true;
			if(on_best) on_best(-1,ordinal,val,cand);
		}
		++ordinal;
		{
			std::unique_lock<std::mutex> lock(mutex_stream);
			_sslot_state[h]=SLOT_EMPTY;
		}
		++head;
	}

	// shutdown: wake all workers; on failure drop undispatched work first
	{
		std::unique_lock<std::mutex> lock(mutex_stream);
		if(failed) _spending.clear();
		_sstop=true;
	}
	_scv_work.notify_all();
	stp.join_all();

	if(_worker_exception) { // rethrow a worker-thread criterion exception on the master (fst4 main catches it like any sequential failure)
		std::exception_ptr e=_worker_exception;
		_worker_exception=nullptr;
		std::rethrow_exception(e);
	}
	if(failed) return false;
	return best_available;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::invalidate_clones()
{
	if(working) { // a batch is still open: let its outstanding evaluations drain first, as
	              // initialize() does; the workers are parked afterwards
		std::unique_lock<std::mutex> lock(mutex_candidate_getter);
		condition_batch_done.wait(lock, [this]{return _completed_candidates==_available_candidates;});
	}
	// the workers are parked, so the master may touch the clone array; the next initialize()
	// or evaluate_stream() finds every slot CRIT_NONEXISTENT and re-clones from the criterion
	// it is handed - in that criterion's CURRENT context
	for(unsigned int i=0;i<_max_threads;i++) {tlocal[i].crit.reset(); tlocal[i].status=CRIT_NONEXISTENT;}
}

} // namespace
#endif // FSTCANDIDATEEVALUATORTHREADED_H ///:~
