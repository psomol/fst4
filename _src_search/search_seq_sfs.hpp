#ifndef FSTSEARCHSEQSFS_H
#define FSTSEARCHSEQSFS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_seq_sfs.hpp
   \brief   implements Sequential_Forward_Selection and Sequential_Backward_Selection
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
#include "stopwatch.hpp"
#include "search_seq.hpp"

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

/*! \brief Implements Sequential_Forward_Selection and Sequential_Backward_Selection

	\note By default any initial subset contents are ignored; SFS starts from empty set and SBS starts from full set. 
	      Call enable_initial_subset() to let the search start from the initial sub contents.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_SFS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent; //!< base class alias
	typedef typename parent::parent grandparent; //!< the shared Search base (output detail, result tracker)
	typedef std::shared_ptr<EVALUATOR> PEvaluator; //!< shared pointer to the sequential-step evaluator
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	Search_SFS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator) {_direction=FORWARD; notify("Search_SFS constructor.");} //!< \p evaluator must be non-null; direction defaults to FORWARD (SFS); call set_search_direction(BACKWARD) for SBS
	virtual ~Search_SFS() {notify("Search_SFS destructor.");}
	
	// NOTE: crit is expected to be already initialize()d before use here
	/*! \brief Runs SFS (or SBS) and returns the resulting subset + its criterion value.

	    Starts from the empty set (FORWARD) or the full set (BACKWARD), unless
	    enable_initial_subset() was called, in which case the incoming contents of
	    \p sub are the starting point. Each step adds (removes) the best g-tuple of
	    features, g being the generalization level, until target_d features are
	    selected; with target_d==0 the search runs to the end of its range and
	    returns the best-valued subset among all sizes visited.
	    \return false if the criterion failed on some subset, or if the requested
	    combination of starting size, target_d and generalization level makes a
	    solution unreachable (see the reachability test at the start of the body):
	    for instance a forward search asked for fewer features than it starts with,
	    or a target_d not reachable by whole g-sized steps. */
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override;

	void set_search_direction(const SearchDirection direction) {assert(direction==FORWARD || direction==BACKWARD); _direction=direction;} //!< FORWARD implies SFS, BACKWARD implies SBS (do not change concurrently to running search())
	SearchDirection get_search_direction() const {return _direction;} //!< the currently configured direction (see set_search_direction())
	
	//! retrieves the subset of cardinality \p d (1-based: d ranges over 1..n) recorded during the last search(), together with its criterion value; false if that search recorded nothing at this cardinality - which is the case for every size it did not visit, and for all sizes if it returned false. The contents are copied into \p sub, which must be non-null and dimensioned for the full feature set
	bool get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub);

	virtual std::ostream& print(std::ostream& os) const override;
protected:
	SearchDirection _direction; //!< accepted values FORWARD for SFS or BACKWARD for SBS
	//! Structure to hold [subset,criterion value] temporary solutions in the course of search
	struct OneSubset {
		RETURNTYPE critvalue; //!< the subset's criterion value
		std::unique_ptr<SUBSET> sub; //!< the subset itself, or null if no result was recorded at this cardinality
	};
	std::unique_ptr<OneSubset[]> bsubs; //!< per-cardinality result found in the course of the last search() (index d-1); read by get_result()
	DIMTYPE _n; //!< bsubs array size
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	if(_direction==FORWARD) os << "SFS"; else os << "SBS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << "  [Search_SFS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	// NOTE: certain parameter combinations (initial sub->get_d() versus target_d versus parent::_generalization_level) 
	//       make it impossible to yield a valid solution. In such case search() just returns false
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	bool track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d<=sub->get_n());
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	const bool _forward=(_direction==FORWARD);
	sub->set_forward_mode(true); // select_all()/deselect_all() write mode-dependent ids - a Subset left in backward mode by a previous search() would otherwise re-initialize empty
	if(!parent::initial_subset_enabled()) {
		if(_forward) sub->deselect_all(); else sub->select_all();
	}
	DIMTYPE _d=sub->get_d();
	_n=sub->get_n();
	bsubs.reset(new OneSubset[_n]);

	// no solution is reachable under this setting when, with g = generalization level:
	//   forward, but there is no room for another g-tuple (d+g>n),
	//   backward, but removing a g-tuple would not leave a non-empty subset (d<=g),
	//   a target size is requested that the starting subset has already reached or passed
	//     (forward: d>=target_d, backward: d<=target_d),
	//   or the distance between d and target_d is not a whole number of g-sized steps
	if( (_forward && _d+parent::get_generalization_level()>_n) || 
		(!_forward && _d<=parent::get_generalization_level()) || 
		(target_d>0 && _forward && _d>=target_d) || 
		(target_d>0 && !_forward && _d<=target_d) ||
		(target_d>0 && ((target_d>_d)?(target_d-_d):(_d-target_d))%parent::get_generalization_level()>0) ) return false; // no solution reachable under this setting
	
	if(grandparent::output_normal()) {
		msgbuf sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		sos << "with initial subset: " << *sub << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		this->report(os,sos);
	}

	sub->set_forward_mode(_forward); // FORWARD is the prevailing mode

	if(_d>0) { // evaluate initial subset if any
		if(!crit->evaluate(result,sub)) return false;
		if(track) grandparent::_tracker->add(result,sub);
		bsubs[_d-1].sub.reset(new SUBSET(_n));
		bsubs[_d-1].sub->stateless_copy(*sub);
		bsubs[_d-1].critvalue=result;
		if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "subresult=" << result << " " << *sub << std::endl << std::endl);
	}

	DIMTYPE steps;
	if(target_d>0 && target_d<=_n) { // search for subset of size target_d
		steps=((target_d>_d)?(target_d-_d):(_d-target_d))/parent::get_generalization_level();
	} else { // find best dimensionality
		if(_forward) steps=(_n-_d)/parent::get_generalization_level();
		else steps=(_d-1)/parent::get_generalization_level();
	}
	
	// main search loop
	for(DIMTYPE i=0;i<steps;i++) {
		if(!parent::Step(_forward, result, sub, crit, os)) return false;
		if(track) grandparent::_tracker->add(result,sub);
		if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "subresult=" << result << " " << *sub << std::endl << std::endl << std::flush);
		_d=sub->get_d();
		bsubs[_d-1].sub.reset(new SUBSET(_n));
		bsubs[_d-1].sub->stateless_copy(*sub);
		bsubs[_d-1].critvalue=result;
	}
	
	DIMTYPE finalidx;
	if(target_d>0 && target_d<=_n) finalidx=target_d-1;
	else {
		for(finalidx=0;finalidx<_n && !bsubs[finalidx].sub;finalidx++);
		for(DIMTYPE i=finalidx+1;i<_n;i++) if(bsubs[i].sub && bsubs[i].critvalue>bsubs[finalidx].critvalue) finalidx=i;
	}

	if(finalidx>=_n || !bsubs[finalidx].sub) return false;

	result=bsubs[finalidx].critvalue;
	sub->stateless_copy(*(bsubs[finalidx].sub));

	if(grandparent::output_normal()) {
		msgbuf sos; 
		sos << "......................................." << std::endl;
		sos << "Search_SFS() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub) // retrieves temporary solution of cardinality d that had been updated in the course of the last search
{
	assert(sub);
	assert(d>0 && d<=_n);
	if(bsubs && d>0 && d<=_n && bsubs[d-1].sub) {
		result=bsubs[d-1].critvalue;
		sub->stateless_copy(*(bsubs[d-1].sub));
		return true;
	}
	return false;
}


} // namespace
#endif // FSTSEARCHSEQSFS_H ///:~
