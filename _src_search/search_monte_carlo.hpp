#ifndef FSTSEARCHMONTECARLO_H
#define FSTSEARCHMONTECARLO_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_monte_carlo.hpp
   \brief   Repeatedly samples random subsets to eventually yield the one with highest criterion value
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

#include <memory>
#include <cassert>
#include <iostream>
#include <sstream>
#include <ctime>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search.hpp"
#include "result_tracker.hpp"
#include "candidate_evaluator.hpp"
#include "candidate_evaluator_sequential.hpp"

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

/*! \brief Repeatedly samples random subsets to eventually yield the one with highest criterion value

	The search always runs under a terminating condition - a number of random trials,
	a time limit in seconds, or both; with neither requested it stops after 1000
	trials (see set_stopping_condition()). Candidate evaluation goes through the
	injected engine (a Candidate_Evaluator implementation, sequential in-line by
	default); the random draws themselves always happen on the calling thread, so the
	same seed yields the same sequence of candidates at any thread count.


	\note offers two ways of random subset generation:
	      a) set_cardinality_randomization(const DIMTYPE d_from, const DIMTYPE d_to) invokes
	         the first option, i.e., when generating a random subset, the subset size
	         is first chosen randomly from [d_from,d_to], then features are randomly
	         selected until the required number is reached
	      b) set_cardinality_randomization(const float d_prob) invokes
	         the second option, where each feature is randomly included/excluded
	         with probability d_prob. The expected subset size is thus d_prob*no_of_all_features,
	         but can be actually anything from [0,no_of_all_features].
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Monte_Carlo : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the injected evaluation-strategy interface (sequential in-line or thread-pooled)
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to an EVALUATORENGINE
	//! \p engine defaults to sequential in-line evaluation if not supplied; see set_stopping_condition() for \p trials_limit / \p time_limit / \p time_limit_check_freq
	Search_Monte_Carlo(const unsigned long long trials_limit=1000, const unsigned long time_limit=0, const unsigned int time_limit_check_freq=1, PEvaluatorEngine engine=PEvaluatorEngine()):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine) {
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Search_Monte_Carlo constructor."); set_stopping_condition(trials_limit,time_limit,time_limit_check_freq); reset_cardinality_randomization();
	}
	virtual ~Search_Monte_Carlo() {notify("Search_Monte_Carlo destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	//! samples random subsets of target_d features (of randomized size if target_d==0: from the configured range or inclusion probability, see set_cardinality_randomization(), or uniformly from [1,n] when neither is set) until the stopping condition is met, and returns the best one + its criterion value; false if no candidate was evaluated successfully. \p target_d must be smaller than the full feature count - drawing all n features at random is not a search (unlike Search_Exhaustive, which accepts target_d==n)
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override;

	//! set maximum number of random trials or maximum time in seconds or both; \p trials_limit and \p time_limit both 0 is replaced by a limit of 1000 trials, \p time_limit_check_freq 0 by 100 (the elapsed time is read once per that many trials; the constructor asks for 1, i.e. a check per trial)
	void set_stopping_condition(const unsigned long long trials_limit=1000, const unsigned long time_limit=0, const unsigned int time_limit_check_freq=100) {if(trials_limit==0 && time_limit==0) {_trials_limit=1000; _time_limit=0;} else {_trials_limit=trials_limit; _time_limit=time_limit;} if(time_limit_check_freq==0) _time_limit_check_freq=100; else _time_limit_check_freq=time_limit_check_freq; }

	// set the way of random subset (size) generation
	void set_cardinality_randomization(const DIMTYPE d_from, const DIMTYPE d_to) {if(d_from>0 && d_from<=d_to) {_d_from=d_from; _d_to=d_to;} else {_d_from=_d_to=0;}} //!< generate subset size first from [d_from,d_to], then randomly choose exactly that number of features
	void set_cardinality_randomization(const float d_prob) {if(d_prob>0.0 && d_prob<1.0) _d_prob=d_prob; else _d_prob=0.0;} //!< for each feature roll the dice and select it with probability d_prob
	void reset_cardinality_randomization() {_d_from=_d_to=0; _d_prob=0.0;} //!< clears both ways of randomizing the subset size (the constructor calls this); with neither of them in effect a target_d==0 search draws each probe's size uniformly from [1,n]

	virtual std::ostream& print(std::ostream& os) const override;

protected:
	PEvaluatorEngine _engine; //!< injected evaluation strategy (sequential-inline or thread-pooled); defaulted in the constructor if not supplied
	unsigned long _time_limit; //!< max time allowed to be spent in seconds
	unsigned long long _trials_limit; //!< max no of candidates evaluated (64-bit: counts candidates, see the LLP64 note on Candidate_Evaluator::BestCallback)
	unsigned int _time_limit_check_freq; //!< number of trials before next time check
	DIMTYPE _d_from; //!< lower bound of the random subset size range; see set_cardinality_randomization(const DIMTYPE, const DIMTYPE)
	DIMTYPE _d_to; //!< upper bound of the random subset size range
	float _d_prob; //!< active d_prob must be from (0,1]; generates random subset of random size where the mean size over a series of calls is d_prob*no_of_all_features
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::print(std::ostream& os) const
{
	if(_engine->is_threaded()) {
		os << "Random search  [Search_Monte_Carlo(max_threads="<<_engine->get_max_threads();
		if(_trials_limit>0) os << ", maxtrials="<<_trials_limit;
		if(_time_limit>0) os<<", seconds="<<_time_limit;
		if(_d_prob>0.0) os<<", d_prob="<<_d_prob;
		else os<<", d_from="<<_d_from<<", d_to="<<_d_to;
		os<<")";
	} else {
		os << "Random search  [Search_Monte_Carlo(";
		if(_trials_limit>0) os << "maxtrials="<<_trials_limit<<", ";
		if(_time_limit>0) os<<"seconds="<<_time_limit<<", ";
		if(_d_prob>0.0) os<<"d_prob="<<_d_prob;
		else os<<"d_from="<<_d_from<<", d_to="<<_d_to;
		os<<")";
	}
	if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
	os<<"]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	assert(sub);
	assert(crit);
	assert(_engine);
	const DIMTYPE n=sub->get_n();
	assert(target_d<n);

	bool track=parent::result_tracker_active(); if(track) parent::_tracker->set_output_detail(parent::get_output_detail());

	if(parent::output_normal()) {
		msgbuf sos;
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		this->report(os,sos);
	}
	assert(_trials_limit!=0 || _time_limit!=0);
	assert(_time_limit_check_freq>0);
	unsigned long long _trials=0; // the printed twin of the engine's generation ordinal - same 64-bit domain
	unsigned long _time=0;
	// the random subset size range used when target_d==0 and no inclusion probability is set:
	// the configured [_d_from,_d_to] clamped into [1,n]; with no range configured it is the full [1,n]
	DIMTYPE _d_from_tmp=_d_from, _d_to_tmp=_d_to;
	if(_d_to_tmp<1 || _d_to_tmp>n) _d_to_tmp=n;
	if(_d_from_tmp<1) _d_from_tmp=1; else if(_d_from_tmp>_d_to_tmp) _d_from_tmp=_d_to_tmp;
	assert(_d_from_tmp>=1 && _d_from_tmp<=_d_to_tmp && _d_to_tmp<=n);

	// random candidates are generated until the stopping condition (trials and/or
	// time limit) is met; both engines call the generator on the MASTER thread only,
	// so the random draws all come from one fst_rand stream (same seed = same
	// candidate sequence, threaded or not) and the trial counter needs no synchronization
	StopWatch *psw=&swatch;
	typename EVALUATORENGINE::Generator generator=[this,psw,&_trials,&_time,_d_from_tmp,_d_to_tmp,target_d](SUBSET &cand)->bool {
		if( (_trials_limit!=0 && _trials>=_trials_limit) || (_time_limit!=0 && _time>=_time_limit) ) return false;
		if(target_d==0) {
			if(_d_prob>0.0) do {cand.make_random_subset(_d_prob);} while(cand.get_d()==0);
			else cand.make_random_subset(_d_from_tmp,_d_to_tmp);
		} else cand.make_random_subset(target_d);
		++_trials;
		if(_trials%_time_limit_check_freq==0) _time=(unsigned long)psw->time_elapsed();
		return true;
	};

	typename EVALUATORENGINE::BetterPredicate better=[](const RETURNTYPE val, const PSubset &cand, const RETURNTYPE bestval, const PSubset &bestsub)->bool {
		return val>bestval || (val==bestval && cand->get_d()<bestsub->get_d());
	};

	typename EVALUATORENGINE::BestCallback on_best;
	if(parent::output_normal()) {
		std::ostream *pos=&os;
		// both engines fire this on the master in generation order (tidx==-1) with the
		// candidate's generation ordinal == the trial index, threaded or not
		on_best=[this,psw,pos](const int /*tidx*/, const unsigned long long ordinal, const RETURNTYPE val, const PSubset &bsub) {
			msgbuf sos;
			sos << "new MAXCRIT="<<val<< ", trial=" << ordinal << ", " << *bsub << std::endl << *psw << std::endl << std::endl << std::flush;
			this->report_solution(*pos,val,*bsub,sos);
		};
	}

	if(!_engine->evaluate_stream(generator, crit, track? parent::_tracker : typename parent::PResultTracker(), better, on_best, result, sub)) return false;

	if(parent::output_normal()) {
		msgbuf sos;
		sos << "......................................." << std::endl;
		if(_engine->is_threaded()) sos << "Search_Monte_Carlo_Threaded() search finished after "<<_trials<<" trials. " << swatch << std::endl;
		else sos << "Search_Monte_Carlo() search finished after "<<_trials<<" trials. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}

} // namespace
#endif // FSTSEARCHMONTECARLO_H ///:~
