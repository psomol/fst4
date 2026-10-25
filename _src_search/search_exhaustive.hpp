#ifndef FSTSEARCHEXHAUSTIVE_H
#define FSTSEARCHEXHAUSTIVE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_exhaustive.hpp
   \brief   Implements exhaustive (optimal) search over all feature subsets
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

/*! \brief Implements exhaustive (optimal) search yielding optimal feature subset with respect to chosen criterion

	This single class serves both the sequential and the threaded execution
	model: the evaluation strategy (a Candidate_Evaluator implementation) is
	injected through the constructor and defaults to sequential in-line
	evaluation; construct with a Candidate_Evaluator_Threaded to evaluate
	candidates in a thread pool. Results are the same either way - for a fixed
	seed the search yields the same subset at any thread count.
	Candidates are streamed to the evaluation engine - they are never
	materialized as a whole (their count grows combinatorially with n).

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. In case of result tracking the excessive
	number of possible combinations may consume unacceptable amount of memory - to prevent
	this it is highly recommended to set up result tracker storage limit.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Exhaustive : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class: the generic search interface
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion being optimized
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the candidate-evaluation engine type (sequential or threaded)
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to the evaluation engine
	/*! \brief Constructs the search with the given candidate-evaluation engine; a default-constructed \p engine falls back to Candidate_Evaluator_Sequential (in-line, unthreaded evaluation).
	    \param engine the evaluation strategy to use; pass a Candidate_Evaluator_Threaded to evaluate candidates in a thread pool */
	explicit Search_Exhaustive(PEvaluatorEngine engine=PEvaluatorEngine()):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine) {
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Search_Exhaustive constructor.");
	}
	virtual ~Search_Exhaustive() {notify("Search_Exhaustive destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	//! evaluates every subset of the requested size (every subset of any size from 1 to n if target_d==0) and returns the best one + its criterion value; false if no candidate could be generated (n==0, or the subset object refused to enumerate) or a criterion evaluation failed
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override;

	virtual std::ostream& print(std::ostream& os) const override {
		os << "Exhaustive search  [";
		if(_engine->is_threaded()) os << "Search_Exhaustive(threads="<<_engine->get_max_threads()<<")";
		else os << "Search_Exhaustive()";
		if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
		os<<"]"; return os;
	}
protected:
	PEvaluatorEngine _engine; //!< the injected candidate-evaluation engine (defaults to sequential)
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Exhaustive<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	notify("Search_Exhaustive::search().");
	assert(sub);
	assert(crit);
	assert(_engine);
	const DIMTYPE n=sub->get_n();
	assert(target_d<=n);

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

	// stream all candidate subsets of each considered cardinality to the engine
	DIMTYPE t_to=n;
	DIMTYPE cardinality=1;
	if(target_d>0) {cardinality=target_d; t_to=target_d;}
	bool generate_first=true;
	bool generator_failed=false; // set if the subset object cannot start enumerating candidates of the current cardinality at all (e.g. cardinality>n); such a run yields no result and search() returns false
	PSubset gen_sub(new SUBSET(n));
	typename EVALUATORENGINE::Generator generator=[&gen_sub,&cardinality,&generate_first,&generator_failed,t_to](SUBSET &cand)->bool {
		for(;;) {
			if(generate_first) {
				if(cardinality>t_to) return false;
				gen_sub->deselect_all();
				if(!gen_sub->getFirstCandidateSubset(cardinality,false/*reverse*/)) {generator_failed=true; return false;} // no candidate subsets can be traversed for whatever reason
				generate_first=false;
				cand.stateless_copy(*gen_sub);
				return true;
			}
			if(gen_sub->getNextCandidateSubset()) {cand.stateless_copy(*gen_sub); return true;}
			++cardinality; // all candidates of current cardinality generated - move on
			generate_first=true;
		}
	};

	typename EVALUATORENGINE::BetterPredicate better=[](const RETURNTYPE val, const PSubset&, const RETURNTYPE bestval, const PSubset&)->bool {return val>bestval;};

	typename EVALUATORENGINE::BestCallback on_best;
	if(parent::output_normal()) {
		StopWatch *psw=&swatch;
		std::ostream *pos=&os;
		// both engines fire this on the master thread in generation order (tidx==-1),
		// so threaded output matches sequential output
		on_best=[this,psw,pos](const int /*tidx*/, const unsigned long long /*ordinal*/, const RETURNTYPE val, const PSubset &bsub) {
			msgbuf sos;
			sos << "new MAXCRIT="<<val<<", " << *bsub << std::endl << *psw << std::endl << std::endl << std::flush;
			this->report_solution(*pos,val,*bsub,sos);
		};
	}

	bool found=_engine->evaluate_stream(generator, crit, track? parent::_tracker : typename parent::PResultTracker(), better, on_best, result, sub);
	if(generator_failed || !found) return false;

	if(parent::output_normal()) {
		msgbuf sos;
		sos << "......................................." << std::endl;
		if(_engine->is_threaded()) sos << "Search_Exhaustive_Threaded() search finished. " << swatch << std::endl;
		else sos << "Search_Exhaustive() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}

} // namespace
#endif // FSTSEARCHEXHAUSTIVE_H ///:~
