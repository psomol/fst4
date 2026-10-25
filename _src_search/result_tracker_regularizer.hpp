#ifndef FSTRESULTTRACKERREGULARIZER_H
#define FSTRESULTTRACKERREGULARIZER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    result_tracker_regularizer.hpp
   \brief   Enables eventual selection of a different subset
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker_dupless.hpp"

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

/*! \brief Collects multiple results. Enables eventual selection of alternative solution based on secondary criterion.

    Enables selecting a different subset than the one
    yielded by the search procedure, e.g., with the aim to reduce overfitting
    or to take into account known feature acquisition cost. 
    
    The idea is to consider all known
    subsets with criterion value close enough to the maximum as effectively
    equal - then another criterion is used to choose among those alternative
    solutions. This technique can improve generalization/robustness and reduce
    overfitting, as well as open up other possibilities, like preferring
    a subset being almost equal to the primarily selected, but with considerably 
    lower feature value acquisition cost, etc. For details see paper
    "Somol, Grim, Pudil: The Problem of Fragile Feature Subset Preference in Feature Selection Methods 
    and A Proposal of Algorithmic Workaround. In Proc. ICPR 2010.  IEEE Computer 
    Society, 2010".

*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Result_Tracker_Regularizer : public Clonable_Impl<Result_Tracker_Regularizer<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,CRITERION>,Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> parent; //!< base class alias
	typedef Clonable_Impl<Result_Tracker_Regularizer,parent,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone() via Clonable_Impl; clone()/sharing_clone() defer to Result_Tracker's throwing defaults (see CLONE_INHERIT)
	typedef typename parent::ResultRec ResultRec; //!< one tracked [value,subset] result record, as defined by Result_Tracker
	typedef ResultRec* PResultRec; //!< raw pointer to a ResultRec
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to a criterion (the secondary criterion used to re-rank tracked results)
	Result_Tracker_Regularizer(const IDXTYPE capacity_limit=0):parent_impl(capacity_limit),_searching(false) {notify("Result_Tracker_Regularizer constructor.");} //!< \p capacity_limit forwarded to Result_Tracker_Dupless (0 = unlimited)
	Result_Tracker_Regularizer(const Result_Tracker_Regularizer& rtr):parent_impl(rtr),_searching(false) {notify("Result_Tracker_Regularizer copy-constructor.");} //!< the copy starts with a fresh (non-searching) equivalence-group cursor; _crit/_prevmax/_iter are not copied
	virtual ~Result_Tracker_Regularizer() {notify("Result_Tracker_Regularizer destructor.");}
	
	// NOTE: crit is expected to be already initialize()d before use here
	/*! \brief Re-ranks the tracked results within \p margin of the best tracked (primary) value by a secondary \p crit, returning the one \p crit ranks highest.

	    Assumes the tracked list is sorted with the best (primary) value first, as
	    Result_Tracker_Dupless maintains it, and stops scanning once a tracked
	    value drops below best_primary_value - \p margin. Among the subsets
	    within the margin, picks the one with the highest \p crit value,
	    breaking ties in favor of the smaller cardinality.
	    \param margin non-negative margin (0 restricts the scan to the primary-best value only).
	    \param result1 receives the primary (originally tracked) value of the returned subset.
	    \param result2 receives the secondary \p crit's value of the returned subset.
	    \param sub receives the chosen subset.
	    \param crit the secondary criterion used to re-rank candidates within margin; must be non-null and already initialize()d.
	    \return false if the tracker is empty or a \p crit evaluation fails. */
	virtual bool optimize_within_margin(const RETURNTYPE margin, RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit);

	/*! \brief Starts a group-by-group scan: returns the subset of the first equivalence group (the tracked subsets sharing the highest primary criterion value) that \p crit ranks highest.

	    \p crit must be non-null and already initialize()d, \p sub must be non-null.
	    Ties in \p crit are broken in favor of the smaller cardinality. \p crit is
	    remembered for the subsequent optimize_within_next_equivalence_group() calls.
	    \return false if the tracker holds no result, or if a \p crit evaluation failed;
	            the scan is then not in progress and the next-group call returns false too. */
	virtual bool optimize_within_first_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit);
	/*! \brief Continues the scan started by optimize_within_first_equivalence_group(): returns the best subset of the next equivalence group (the next lower primary criterion value), ranked by the criterion given there.

	    \return false if no scan is in progress (the first-group call was never made,
	            failed, or the previous call consumed the last group), or if a criterion
	            evaluation failed.
	    \warning The scan holds a cursor into the tracked list. Do not add() to the
	    tracker (nor join() into it) between the calls: trimming by the capacity limit
	    or the margin can erase the record the cursor points to. */
	virtual bool optimize_within_next_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub);



	virtual std::ostream& print(std::ostream& os) const override {os << "Result_Tracker_Regularizer(limit=" << parent::_capacity_limit << ", margin=" << parent::_margin << ") size " << parent::results.size(); return os;}

private:
	bool optimize_within_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub); //!< building block for optimize_within_*_equivalence_group
	PCriterion _crit; //!< the secondary criterion set by optimize_within_first_equivalence_group(), reused by subsequent optimize_within_next_equivalence_group() calls
	RETURNTYPE _prevmax; //!< the primary value of the equivalence group currently being scanned; helper of optimize_within_equivalence_group()
	typename parent::CONSTRESULTSITER _iter; //!< cursor into the tracked results list, carried across optimize_within_*_equivalence_group() calls so scanning resumes where the previous call left off; adding to the tracker during a scan may invalidate it
	bool _searching; //!< true while an equivalence-group scan is in progress, i.e. optimize_within_next_equivalence_group() may still find further groups
};



template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_margin(const RETURNTYPE margin, RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit)
{
	assert(margin>=0);
	if(parent::results.empty()) return false;
	bool stop=false;
	
	typename parent::CONSTRESULTSITER iter=parent::results.begin();
	typename parent::CONSTRESULTSITER sel=iter;
	RETURNTYPE bestval, val;
	
	// NOTE: crit is expected to be already initialize()d before use here
	if(!crit->evaluate(bestval,sel->sub)) return false;
	notify("sel: val1=",sel->value,", val2=",bestval,", getd ",sel->sub->get_d(),".");
	
	const RETURNTYPE threshold=sel->value-margin;
	iter++;
	while(!stop && iter!=parent::results.end())
	{
		if(iter->value<threshold) stop=true; else
		{
			// NOTE: crit is expected to be already initialize()d before use here
			if(!crit->evaluate(val,iter->sub)) return false;
			notify("iter: val1=",iter->value,", val2=",val,", getd ",iter->sub->get_d(),".");
			if(val>bestval || (val==bestval && sel->sub->get_d()>iter->sub->get_d())) {bestval=val; sel=iter; notify("best updated.");}
			iter++;
		}
	}
	result1=sel->value; result2=bestval; sub=sel->sub;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub)
{
	// precondition: _iter points to a valid record of the tracked list and _crit is set - both public entry points guarantee this before calling in
	_prevmax=_iter->value;
	RETURNTYPE val, maxval1, maxval2; PSubset maxsub=_iter->sub;
	maxval1=_iter->value;
	if(!_crit->evaluate(maxval2,_iter->sub)) {_searching=false; return false;}
	_iter++;
	while(_iter!=parent::results.end() && _iter->value==_prevmax)
	{
		if(!_crit->evaluate(val,_iter->sub)) {_searching=false; return false;}
		if(val>maxval2 || (val==maxval2 && maxsub->get_d()>_iter->sub->get_d())) {maxval2=val; maxsub=_iter->sub;}
		_iter++;
	}
	if(_iter==parent::results.end()) _searching=false; else _searching=true;
	result1=maxval1; result2=maxval2; sub=maxsub;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_first_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit)
{ // returns the subset from the first equivalence group (=subsets with equal primary criterion value) yielding the highest crit value
	assert(crit);
	assert(sub);
	if(parent::results.empty()) {_searching=false; return false;}
	_crit=crit;
	_iter=parent::results.begin();
	return optimize_within_equivalence_group(result1,result2,sub);
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_next_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub) 
{ // returns the subset from the next equivalence group (=subsets with equal primary criterion value which is the next highest in sequence) yielding the highest crit value
	assert(_crit);
	assert(sub);
	if(!_searching) return false;
	return optimize_within_equivalence_group(result1,result2,sub);
}

} // namespace
#endif // FSTRESULTTRACKERREGULARIZER_H ///:~
