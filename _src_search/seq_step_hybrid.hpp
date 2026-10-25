#ifndef FSTSEARCHSEQSTEPHYBRID_H
#define FSTSEARCHSEQSTEPHYBRID_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    seq_step_hybrid.hpp
   \brief   Implements hybrid selection step in sequential search type of methods 
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
#include <list>
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

/*! \brief Implements hybrid selection step in sequential search type of methods

    A "hybrid" step pre-filters the candidates with a second, presumably much cheaper
    criterion, so that the main (typically wrapper) criterion has to be evaluated on a
    small shortlist only. Every candidate offered by the working subset is first
    evaluated with \c filtercrit; the best keep_perc percent of them (at most keep_max,
    see set_prefiltering_limits()) are then evaluated with the main criterion, and the
    best of those is the step's result.

    The two end points of keep_perc:
    \li keep_perc==100 keeps every candidate, so the main criterion sees the same candidate
        set as Sequential_Step_Straight and the step returns the same best VALUE - but the
        candidates reach the main criterion in filter order rather than in feature order, so
        when several candidates tie on the main criterion a different one may be chosen (the
        first in filter order wins); plus the cost of the extra filter pass.
    \li keep_perc==0 shortens the shortlist to a single candidate, the best one according
        to \c filtercrit. Note that the main criterion is still evaluated - on that one
        candidate - and ITS value is what the step returns; what keep_perc==0 bypasses is
        the SELECTION by the main criterion, not the main criterion itself. The returned
        value therefore stays comparable with the values the search sees in other steps.

    \note Like Sequential_Step_Straight this class serves both the sequential and the threaded
    execution model: the main-criterion phase over the prefiltered candidate list runs through
    the injected Candidate_Evaluator (default: sequential in-line evaluation; inject
    Candidate_Evaluator_Threaded for a thread-pooled main phase). The prefilter stage is a
    deterministic pre-sort and always stays on the calling (master) thread. Results are
    identical for any thread count: the engine fires per-result callbacks (tracker feed) in
    candidate order and breaks best-value ties by earliest candidate. */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
class Sequential_Step_Hybrid : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< the abstract Sequential_Step interface this class implements
	typedef std::shared_ptr<FILTERCRITERION> PFilterCriterion; //!< shared pointer to the (typically cheaper) criterion used to pre-filter candidates
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the main criterion the step optimizes
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the candidate-evaluation interface driving the main-criterion phase
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to an EVALUATORENGINE, as passed to the constructor
	//! \param filtercrit the pre-filtering criterion, evaluated on every offered candidate on the calling (master) thread
	//! \param keep_perc percentage of candidates (by filtercrit ranking) kept for the main-criterion phase; 100 bypasses pre-filtering, 0 shortlists one candidate only (see the class documentation). The shortlist never shrinks below one candidate, so the main criterion is always evaluated at least once
	//! \param keep_max hard cap on the number of candidates kept for the main-criterion phase, regardless of keep_perc (must be >0)
	//! \param engine evaluation-strategy engine for the main-criterion phase; defaults to a fresh Candidate_Evaluator_Sequential (inject Candidate_Evaluator_Threaded for a thread-pooled main phase)
	Sequential_Step_Hybrid(const PFilterCriterion filtercrit, const DIMTYPE keep_perc, const DIMTYPE keep_max=1000, PEvaluatorEngine engine=PEvaluatorEngine()):Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine),track(false) {assert(keep_perc<=100); assert(keep_max>0); assert(filtercrit); _filtercrit=filtercrit; _keep_perc=keep_perc; _keep_max=keep_max; _listsize=0; if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>()); notify("Sequential_Step_Hybrid constructor.");}
	virtual ~Sequential_Step_Hybrid() {notify("Sequential_Step_Hybrid destructor.");}

	void set_prefiltering_limits(const DIMTYPE keep_perc, const DIMTYPE keep_max) {assert(keep_perc<=100); assert(keep_max>0); _keep_perc=keep_perc; _keep_max=keep_max;} //!< reconfigures the pre-filtering percentage/cap (see the constructor parameters)
	DIMTYPE get_prefiltering_perc() const {return _keep_perc;} //!< the currently configured keep_perc
	DIMTYPE get_prefiltering_max() const {return _keep_max;}   //!< the currently configured keep_max

	// NOTE: crit is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout) override; //!< chooses among subsets offered by sub->get*CandidateSubset()

	virtual std::ostream& print(std::ostream& os) const override; //!< writes a human-readable identification of the step, its prefiltering settings, filter criterion and (if enabled) result tracker
protected:
	inline bool filter_candidates(const PSubset sub, const DIMTYPE _generalization_level, std::ostream& os); //!< evaluates every offered candidate with _filtercrit and keeps the top _listsize in _filterlist, sorted descending by filtercrit value (ties broken toward the smaller subset); always runs on the calling (master) thread
	inline bool test_candidate(const PSubset sub); //!< evaluates one candidate with _filtercrit and inserts it into _filterlist in sorted position, trimming the list back to _listsize

	PEvaluatorEngine _engine; //!< candidate-evaluation engine driving the main-criterion phase
	PFilterCriterion _filtercrit; //!< the pre-filtering criterion supplied to the constructor
	DIMTYPE _keep_perc; //!< filter out (100-_keep_perc) % of candidate subsets
	DIMTYPE _keep_max; //!< when filtering keep maximally _keep_max candidate subsets (restriction on CANDIDATELIST size)

	//! Nested class to hold [Subset,criterion value] pair in the course of hybrid feature candidate evaluation in Sequential_Step_Hybrid
	class SubsetCandidate {
	public:
		SubsetCandidate(const RETURNTYPE value, const PSubset sub) : _value(value), _sub(new SUBSET(*sub)) {} //!< stores \p value together with an owned copy of \p sub
		const RETURNTYPE _value; //!< the filter criterion value of _sub
		const PSubset _sub;      //!< an owned copy of the candidate subset (independent of the caller's subset instance)
	};
	typedef std::list<SubsetCandidate> CANDIDATELIST; //!< sorted (by filter criterion value, descending) list of surviving pre-filter candidates

	DIMTYPE _listsize; //!< target size of _filterlist, recomputed from _keep_perc/_keep_max at each filter_candidates() call and floored to 1, so the shortlist is never empty
	CANDIDATELIST _filterlist; //!< the surviving pre-filter candidates, sorted descending by filter criterion value (lazy allocation)
	typename CANDIDATELIST::iterator iter; //!< scratch iterator reused by test_candidate()/filter_candidates()/evaluate_candidates() -- not reentrant, matches the single-threaded pre-filter stage

private:
	bool track; //!< cached result_tracker_active() for the current evaluate_candidates() call
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
std::ostream& Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::print(std::ostream& os) const
{
	if(_engine->is_threaded()) os << "Sequential_Step_Hybrid(keep_perc="<<_keep_perc<<", max="<<_keep_max<<", threads="<<_engine->get_max_threads()<<")";
	else os << "Sequential_Step_Hybrid(keep_perc="<<_keep_perc<<", max="<<_keep_max<<")";
	if(_filtercrit) os << " with filter " << *_filtercrit;
	if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::test_candidate(const PSubset sub)
{
	assert(sub);
	assert(_filtercrit);
	RETURNTYPE val;
	DIMTYPE _idx=0;
	assert(sub->get_d()!=0);
	// the filter criterion's values are deliberately NOT fed to the result tracker: the tracker
	// would then hold values of two different criteria, which are not comparable with each other
	if(!_filtercrit->evaluate(val,sub)) return false;
	for(iter=_filterlist.begin(); _idx<_listsize && iter!=_filterlist.end() && (val<iter->_value || (val==iter->_value && sub->get_d()>=iter->_sub->get_d())); iter++) {++_idx;}
	if(_idx<_listsize) { // candidate good enough to be inserted
		_filterlist.insert(iter,SubsetCandidate(val,sub));
		if(_filterlist.size()>_listsize) _filterlist.pop_back();
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::filter_candidates(const PSubset sub, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	_filterlist.clear();
	assert(_keep_perc<=100);
	assert(_keep_max>0);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	
	// determine maximum _filterlist size: keep_perc percent of the candidates, capped at
	// keep_max, but never less than one - a shortlist of zero would leave the main criterion
	// nothing to evaluate, so keep_perc==0 means "shortlist the single best by filtercrit"
	_listsize=DIMTYPE(((double)_keep_perc/100.0)*sub->getNoOfCandidateSubsets(_generalization_level,false/*reverse*/));
	if(_keep_max<_listsize) _listsize=_keep_max;
	if(_listsize==0) _listsize=1;
	
	// first build candidate list _filterlist (pre-filter using _filtercrit criterion)
	bool b=sub->getFirstCandidateSubset(_generalization_level,false/*reverse*/); if(!b) return false; // no candidate subsets can be traversed for whatever reason
	if(!test_candidate(sub)) return false;
	b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
	while(b) {
		if(!test_candidate(sub)) return false;
		b=sub->getNextCandidateSubset();
	}
	assert(_filterlist.size()>0);
	if(parent::output_detailed()) {
		std::ostringstream sos; 
		sos << "candidate list of "<<_filterlist.size() << ":" << std::endl;
		for(iter=_filterlist.begin(); iter!=_filterlist.end(); iter++) {sos << iter->_value << " ~ " << *(iter->_sub) << std::endl;}
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(crit);
	assert(_filtercrit);
	assert(_keep_perc<=100);
	assert(_keep_max>0);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	track=parent::result_tracker_active();

	if(!filter_candidates(sub,_generalization_level,os)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE filter_candidates() sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	assert(_filterlist.size()>0);

	if(_keep_perc==0) { // shortlist of one: no main-criterion SELECTION, but the main criterion
	                    // is still evaluated on that candidate and its value is what we return
		assert(_filterlist.size()>0);
		sub->stateless_copy( *((_filterlist.begin())->_sub) );
		assert(sub->get_d()!=0);
		if(!crit->evaluate(result,sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
		if(track) parent::_tracker->add(result,sub);
		return true;
	}

	// main-criterion phase through the injected engine (sequential in-line by default,
	// thread-pooled when constructed with Candidate_Evaluator_Threaded); the engine fires the
	// per-result callback in candidate (add) order and getBestResult() breaks ties by earliest
	// candidate, so results and tracker feed are the same for any thread count
	assert(_engine);
	typename EVALUATORENGINE::ResultCallback on_result;
	if(track) {
		typename parent::PResultTracker tracker=parent::_tracker;
		on_result=[tracker](const RETURNTYPE val, const PSubset &csub, const DIMTYPE) {tracker->add(val,csub);};
	}
	_engine->initialize(crit,(DIMTYPE)_filterlist.size(),on_result);
	for(iter=_filterlist.begin();iter!=_filterlist.end();iter++) {
		if(!_engine->add_candidate(iter->_sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(iter->_sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	}
	PSubset psub;
	DIMTYPE dummyfeature;
	if(!_engine->getBestResult(result,psub,dummyfeature)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() (no candidate evaluated)" << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	assert(psub);
	sub->stateless_copy(*psub);
	assert(sub->get_d()!=0);

	return true;
}


} // namespace
#endif // FSTSEARCHSEQSTEPHYBRID_H ///:~
