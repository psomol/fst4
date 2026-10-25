#ifndef FSTSEARCHSEQDOS_H
#define FSTSEARCHSEQDOS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_seq_dos.hpp
   \brief   Implements Dynamic_Oscillating_Search
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

/*! \brief Implements Dynamic_Oscillating_Search

    \note DOS is a purely d-optimizing procedure not permitting to specify target subset size, target_d is thus ignored.

	\note By default sub is expected to be initialized prior to search() call, i.e., it should contain a valid initial subset of requested size
	      from which the search will start. Empty initial subset is permitted.
	      To ensure that any initial contents of sub are ignored and the search starts always from an empty set, call disable_initial_subset().
	
	\warning Non-empty init can not be used with Sequential_Step_Hybrid() parametrized _keep_perc==0 (due to initial filtercrit and crit values mix-up)
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_DOS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent; //!< base class alias
	typedef typename parent::parent grandparent; //!< the shared Search base (output detail, result tracker)
	typedef std::shared_ptr<EVALUATOR> PEvaluator; //!< shared pointer to the sequential-step evaluator
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	Search_DOS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator), _delta(3) {parent::enable_initial_subset(); notify("Search_DOS constructor.");} //!< \p evaluator must be non-null; initial_subset_enabled() is turned ON by construction (empty initial subset is permitted); target_d is ignored by DOS's search()
	virtual ~Search_DOS() {notify("Search_DOS destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	// NOTE: target_d is ignored
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override; //!< returns found subset + criterion value

	void set_delta(const DIMTYPE delta) {assert(delta>0); _delta=delta;} //!< oscillating cycle depth
	DIMTYPE get_delta() const {return _delta;} //!< the currently configured oscillating cycle depth (see set_delta())

	virtual std::ostream& print(std::ostream& os) const override;
protected:
	/*! \brief Oscillating-cycle phase for the DOS swing state machine driving search()'s inline switch.

	    Structurally the same states as Search_OS's SwingPhase (RESTART/DOWN/
	    RETURN_UP/UP/RETURN_DOWN), but with one behavioral difference (marked
	    "the DOS to OS difference" at each check site in the implementation):
	    every Step() taken in DOWN/RETURN_UP/UP/RETURN_DOWN compares the new
	    criterion value against the pivot immediately and jumps straight back to
	    RESTART as soon as an improvement is found, instead of waiting for the
	    swing to complete like Search_OS does. This eager restart-on-improvement
	    is what makes DOS "dynamic": the oscillation depth effectively adapts
	    as soon as a better solution turns up, and since DOS optimizes the
	    cardinality too, an EQUAL criterion value counts as an improvement when it
	    is reached with fewer features. */
	enum SwingPhase {
		RESTART=0,
		DOWN=1,
		RETURN_UP=2,
		UP=4,
		RETURN_DOWN=8
	};
	DIMTYPE _delta; //!< current oscillating cycle depth (max swing extent); see set_delta()

	std::unique_ptr<SUBSET> pivotsub; //!< the current pivot: the best-known subset (highest criterion value, smallest cardinality among equals), around which DOWN/UP swings are performed
	std::unique_ptr<SUBSET> pivot_top; //!< the subset the last UP swing ended on; when the cycle depth grows, the next UP swing resumes from here instead of repeating the steps already taken (valid only while the corresponding avail flag is set)
	std::unique_ptr<SUBSET> pivot_bottom; //!< the subset the last DOWN swing ended on, used the same way to resume a deeper DOWN swing without repeating steps
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	os << "DOS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << ", delta=" << _delta;
	os << "  [Search_DOS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	bool track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	if(!parent::initial_subset_enabled()) {sub->set_forward_mode(true); sub->deselect_all();} // deselect_all() writes mode-dependent ids: switch to forward mode FIRST, or a Subset left in backward mode by a previous search comes up in the wrong state

	if( (_delta<parent::get_generalization_level()) ) return false; // impossible to run the search

	if(grandparent::output_normal()) {
		msgbuf sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		sos << "with initial subset: " << *sub << std::endl;
		sos << "Subset size to be optimized." << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}

	DIMTYPE _n=sub->get_n();
	if(!pivotsub || pivotsub->get_n()!=_n) pivotsub.reset(new SUBSET(_n));
	if(!pivot_top || pivot_top->get_n()!=_n) pivot_top.reset(new SUBSET(_n));
	if(!pivot_bottom || pivot_bottom->get_n()!=_n) pivot_bottom.reset(new SUBSET(_n));

	RETURNTYPE pivotval; // pivotsub crit value (~max known crit value)
	
	// sub is assumed to contain the initial subset, or to be empty to request standard DOS course of search
	// NOTE: initializing DOS with Sequential_Step_Hybrid() and _keep_perc==0 does not allow proper initialization of 'result'.
	
	sub->set_forward_mode(true);
	if(sub->get_d()==0) {
		if(!parent::Step(true/*forward*/, result, sub, crit, os)) return false; // find first subset
	} else {
		// the initial subset's value seeds the pivot, so this failure MUST be checked: continuing
		// with an unset result would start the oscillation from an indeterminate pivot. NOTE this
		// value comes from crit, while a Sequential_Step_Hybrid with keep_perc==0 selects by its
		// filter criterion - which is why a non-empty init is unsupported there (see the class docs)
		if(!crit->evaluate(result,sub)) return false;
	}
	if(track) grandparent::_tracker->add(result,sub);
	
	SwingPhase swing=RESTART;

	DIMTYPE _d=sub->get_d();
	DIMTYPE _d_piv;
	DIMTYPE _cur_delta=parent::get_generalization_level();
	bool pivot_top_avail=false;
	bool pivot_bottom_avail=false;

	bool swinged=false; // whether the current swing managed to take at least one step

	// The oscillation itself: a state machine over SwingPhase (documented at its declaration),
	// looping until the cycle depth _cur_delta has grown past the configured maximum _delta.
	// `goto label` below re-enters the switch immediately, without re-testing that loop
	// condition; it is only ever used to jump to RESTART, which resets _cur_delta to the
	// minimum anyway, so the condition cannot be violated by the shortcut. On re-entry _d must
	// equal sub->get_d() and `result` must be the value of that very subset - both hold at every
	// jump site, which is right after a successful Step() on `sub`.
	while(_cur_delta<=_delta) {
		label: switch(swing) {
		case RESTART:
			assert(_d==sub->get_d());
			_d_piv=_d;
			pivotsub->stateless_copy(*sub); pivotval=result; 
			pivot_top_avail=pivot_bottom_avail=false;
			assert(pivotsub->get_d()==_d);
			if(grandparent::output_normal()) this->report_solution(os, pivotval, *pivotsub, msgbuf() << "new pivmax="<<pivotval<<", " << *pivotsub << std::endl  << swatch << std::endl << std::endl << std::flush);
			_cur_delta=parent::get_generalization_level();
			swing=DOWN;
			assert(_d==sub->get_d());
			break;
		case DOWN:
			assert(_d==sub->get_d());
			assert(_d <= _d_piv);
			assert(sub->get_d()!=0);
			if(pivot_bottom_avail) {assert(pivot_bottom->get_d()!=0); sub->stateless_copy(*pivot_bottom); _d=sub->get_d();}
			swinged=false;
			while(_d>parent::get_generalization_level() && _d_piv<_cur_delta+_d){
				assert(_d==sub->get_d());
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "DOWN failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
				if(result>pivotval || (result==pivotval && _d<_d_piv)) {swing=RESTART; goto label;} // the DOS to OS difference: restart as soon as the pivot is beaten, equal value with fewer features included
			}
			if(swinged) {
				assert(_d==sub->get_d());
				if(_d>parent::get_generalization_level()) {
					pivot_bottom_avail=true;
					pivot_bottom->stateless_copy(*sub);
				}
				swing=RETURN_UP;
			} else {
				pivot_bottom_avail=false;
				swing=UP;
			}
			assert(_d==sub->get_d());
			break;
		case RETURN_UP:
			assert(_d==sub->get_d());
			assert(_d<_d_piv);
			while(_d<_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "RETURN UP failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
				if(result>pivotval || (result==pivotval && _d<_d_piv)) {swing=RESTART; goto label;} // the DOS to OS difference: restart as soon as the pivot is beaten, equal value with fewer features included
			}
			assert(_d==_d_piv);
			assert(pivotsub->get_d()>0);
			sub->stateless_copy(*pivotsub);
			swing=UP; swinged=false;
			assert(_d==sub->get_d());
			break;
		case UP:
			assert(_d==sub->get_d());
			assert(_d >= _d_piv);
			if(pivot_top_avail) {assert(pivot_top->get_d()!=0); sub->stateless_copy(*pivot_top); _d=sub->get_d();}
			swinged=false;
			while(_d+parent::get_generalization_level()<=_n && _d<_cur_delta+_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "UP failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
				if(result>pivotval || (result==pivotval && _d<_d_piv)) {swing=RESTART; goto label;} // the DOS to OS difference: restart as soon as the pivot is beaten, equal value with fewer features included
			}
			if(swinged) {
				assert(_d==sub->get_d());
				if(_d+parent::get_generalization_level()<=_n) {
					pivot_top_avail=true;
					pivot_top->stateless_copy(*sub);
				}
				swing=RETURN_DOWN;
			} else {
				pivot_top_avail=false;
				swing=DOWN;
				_cur_delta+=parent::get_generalization_level();
			}
			assert(_d==sub->get_d());
			break;
		case RETURN_DOWN:
			assert(_d==sub->get_d());
			assert(_d>_d_piv);
			while(_d>_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "RETURN DOWN failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
				if(result>pivotval || (result==pivotval && _d<_d_piv)) {swing=RESTART; goto label;} // the DOS to OS difference: restart as soon as the pivot is beaten, equal value with fewer features included
			}
			assert(_d==_d_piv);
			assert(pivotsub->get_d()>0);
			sub->stateless_copy(*pivotsub);
			swing=DOWN; swinged=false;
			_cur_delta+=parent::get_generalization_level();
			assert(_d==sub->get_d());
			break;
		}
	}
	// the loop ends once the cycle depth exceeds _delta; the pivot is then the best subset found
	if(pivotsub->get_d()!=0) {
		sub->stateless_copy(*pivotsub);
		result=pivotval;
		if(grandparent::output_normal()) {
			msgbuf sos; 
			sos << "......................................." << std::endl;
			sos << "Search_DOS() search finished. " << swatch << std::endl;
			sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
			this->report(os,sos);
		}
		return true;
	}
	return false; // an empty pivot means no valid subset was ever adopted
}


} // namespace
#endif // FSTSEARCHSEQDOS_H ///:~
