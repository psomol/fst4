#ifndef FSTSEARCHSEQSTEPSTRAIGHT_H
#define FSTSEARCHSEQSTEPSTRAIGHT_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    seq_step_straight.hpp
   \brief   Implements standard sequential selection step in sequential search type of methods 
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
#include "error.hpp"
#include "global.hpp"
#include "seq_step.hpp"
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

/*! \brief Implements standard sequential selection step in sequential search type of methods

    \note Evaluates each subset candidate using the supplied criterion and returns the
    one subset that yielded highest criterion value; ties go to the candidate
    generated first. This single class serves both the sequential and the threaded
    execution model: the evaluation strategy (a Candidate_Evaluator implementation)
    is injected through the constructor and defaults to sequential in-line
    evaluation. Construct with a Candidate_Evaluator_Threaded to evaluate the
    candidates in a thread pool - the chosen candidate is the same either way.

    \warning With a threaded engine the criterion is cloned per worker on first use
    and the clones are not kept in step with the original afterwards, so the step
    object must be re-created whenever the criterion's context changes (typically
    after every data split change) - see the class documentation of
    Candidate_Evaluator_Threaded. */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Sequential_Step_Straight : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the injected evaluation-strategy interface (sequential in-line or thread-pooled)
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to an EVALUATORENGINE
	//! \p engine defaults to sequential in-line evaluation (Candidate_Evaluator_Sequential) if not supplied; pass a Candidate_Evaluator_Threaded to have the candidates evaluated in a thread pool
	explicit Sequential_Step_Straight(PEvaluatorEngine engine=PEvaluatorEngine()):Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine) {
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Sequential_Step_Straight constructor.");
	}
	virtual ~Sequential_Step_Straight() {notify("Sequential_Step_Straight destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout) override; //!< evaluates every subset offered by sub->get*CandidateSubset() and leaves the highest-valued one in \p sub (see Sequential_Step::evaluate_candidates() for the full contract)

	//! writes "Sequential_Step_Straight(threads=N)" if the injected engine is threaded, else "Sequential_Step_Straight()"
	virtual std::ostream& print(std::ostream& os) const override {
		if(_engine->is_threaded()) os << "Sequential_Step_Straight(threads="<<_engine->get_max_threads()<<")";
		else os << "Sequential_Step_Straight()";
		if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
		return os;
	}
protected:
	PEvaluatorEngine _engine; //!< injected evaluation strategy (sequential-inline or thread-pooled); defaulted in the constructor if not supplied
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(crit);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	assert(_engine);
	// a per-result callback is installed only when a result tracker is attached:
	// without one the candidates are evaluated exactly as before, they are just
	// not recorded anywhere
	typename EVALUATORENGINE::ResultCallback on_result;
	if(parent::result_tracker_active()) {
		typename parent::PResultTracker tracker=parent::_tracker;
		on_result=[tracker](const RETURNTYPE val, const PSubset &csub, const DIMTYPE) {tracker->add(val,csub);};
	}
	// the candidate count is needed only by the threaded engine, to size its result
	// storage; the sequential engine streams candidates and never needs it.
	// getNoOfCandidateSubsets() computes C(n,g) incrementally in 64 bits, so
	// generalized steps (g>=2) are supported threaded as well; it throws only if the
	// count genuinely exceeds DIMTYPE - i.e., when the step loop itself would be
	// infeasible anyway
	const DIMTYPE max_candidates = _engine->is_threaded() ? sub->getNoOfCandidateSubsets(_generalization_level,false/*reverse*/) : 1;
	_engine->initialize(crit, max_candidates, on_result);

	bool b=sub->getFirstCandidateSubset(_generalization_level,false/*reverse*/);
	if(!b) return false; // no candidate subsets can be traversed for whatever reason
	if(parent::output_detailed()) {std::ostringstream sos; sos << "Step...: sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	if(!_engine->add_candidate(sub)) return false;

	b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
	while(b) {
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Step...: sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		if(!_engine->add_candidate(sub)) return false;
		b=sub->getNextCandidateSubset();
	}

	PSubset psub;
	DIMTYPE dummyfeature;
	if(!_engine->getBestResult(result,psub,dummyfeature)) return false;
	assert(psub);
	assert(psub->get_n()==sub->get_n());
	sub->stateless_copy(*psub);
	return true;
}


} // namespace
#endif // FSTSEARCHSEQSTEPSTRAIGHT_H ///:~
