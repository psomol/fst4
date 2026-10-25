#ifndef FSTSEARCHSEQOS_H
#define FSTSEARCHSEQOS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_seq_os.hpp
   \brief   Implements Oscillating_Search
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
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search_ranking_bif.hpp"
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

/*! \brief Implements Oscillating_Search

	\note By default sub is expected to be initialized prior to search() call, i.e., it should contain a valid initial subset of requested size,
	      from which the search will start. If this is not the case and sub does not contain target_d features, sub contents are adjusted 
	      (sub is enlarged by adding features or shrinked by removing features based on individual 
	      feature ranking) before the search actually starts. Setting target_d to 0 leads to repeated search() calls for each possible
	      subset size, to eventually select among them the solution with highest criterion value.
	      To ensure that any initial contents of sub are completely ignored (and initial subset is composed from scratch based purely
	      on individual feature ranking), call disable_initial_subset().

   \warning Search_OS can not be used with Sequential_Step_Hybrid() parametrized _keep_perc==0 (due to initial filtercrit and crit values mix-up)
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_OS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent; //!< base class alias
	typedef typename parent::parent grandparent; //!< the shared Search base (output detail, result tracker)
	typedef Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> SearchBIF; //!< Best Individual Feature ranking search, used to grow/shrink an initial subset to the requested cardinality (see adjust_initial_subset())
	typedef std::shared_ptr<SearchBIF> PSearchBIF; //!< shared pointer to a SearchBIF instance
	typedef std::shared_ptr<EVALUATOR> PEvaluator; //!< shared pointer to the sequential-step evaluator
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	Search_OS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator), _delta(1), track(false) {parent::enable_initial_subset(); notify("Search_OS constructor.");} //!< \p evaluator must be non-null; unlike other sequential searches, initial_subset_enabled() is turned ON by construction since OS always starts from (and if needed adjusts) the supplied subset rather than an empty/full one
	virtual ~Search_OS() {notify("Search_OS destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override; //!< returns found subset + criterion value

	void set_delta(const DIMTYPE delta) {assert(delta>0); _delta=delta;} //!< oscillating cycle depth
	DIMTYPE get_delta() const {return _delta;} //!< the currently configured oscillating cycle depth (see set_delta())

	virtual std::ostream& print(std::ostream& os) const override;
protected:
	/*! \brief Oscillating-cycle phase for the OS swing state machine driving search_one()'s inline switch.

	    RESTART = adopt \c sub as the new pivot (current best) and begin a DOWN
	    swing; DOWN = shrink the pivot by generalization_level features at a
	    time, down to the current cycle depth; RETURN_UP = restore back up to
	    the pivot's cardinality and check whether the DOWN swing improved on the
	    pivot (if so, RESTART; otherwise try UP); UP = grow the pivot; RETURN_DOWN
	    = restore back down to the pivot's cardinality and check whether the UP
	    swing improved (if so, RESTART; otherwise the cycle depth is increased
	    and DOWN retried). Unlike Search_DOS, improvement is checked only once a
	    swing completes (RETURN_UP/RETURN_DOWN), never after an intermediate Step(). */
	enum SwingPhase {
		RESTART=0,
		DOWN=1,
		RETURN_UP=2,
		UP=4,
		RETURN_DOWN=8
	};

	//! standard OS initialized by sub, target dimensionality sub->get_d()
	virtual bool search_one(RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os); //!< returns found subset + criterion value
	/*! \brief Brings \p sub to exactly \p target_d features before the oscillation starts: an
	    undersized subset is supplemented with the best-ranked individual features, an oversized one
	    is reduced by removing the worst-ranked ones. A subset that already has \p target_d features
	    is left untouched.
	    \param target_d the cardinality to reach; must be >0 and at most sub->get_n()
	    \param sub the subset to adjust in place
	    \param bif a ranking search whose evaluate_individuals() has already been run on the same
	           criterion and feature space; it supplies the individual-feature order
	    \param os messages are written here at NORMAL output detail or above
	    \return false if \p bif yields no ranking, or if the adjustment could not reach \p target_d */
	virtual bool adjust_initial_subset(const DIMTYPE target_d, const PSubset sub, const PSearchBIF bif, std::ostream& os);

	DIMTYPE _delta; //!< current oscillating cycle depth (max swing extent); see set_delta()

	std::unique_ptr<SUBSET> pivotsub; //!< the current pivot: the best-known subset for the ongoing search_one() call, around which DOWN/UP swings are performed
	std::unique_ptr<SUBSET> pivot_top; //!< the subset the last UP swing ended on; when the cycle depth grows, the next UP swing resumes from here instead of repeating the steps already taken (valid only while search_one()'s corresponding avail flag is set)
	std::unique_ptr<SUBSET> pivot_bottom; //!< the subset the last DOWN swing ended on, used the same way to resume a deeper DOWN swing without repeating steps

	std::unique_ptr<SUBSET> inisub; //!< target_d==0 only: an untouched copy of the caller's initial subset, re-used as the starting point for every cardinality tried
	std::shared_ptr<SUBSET> tmpsub; //!< target_d==0 only: the working subset one cardinality is searched in, before it is compared with the best result so far
private:
	bool track; //!< whether result tracking is enabled, cached per search() call and read throughout search_one()
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	os << "OS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << ", delta=" << _delta;
	os << "  [Search_OS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::adjust_initial_subset(const DIMTYPE target_d, const PSubset sub, const PSearchBIF bif, std::ostream& os)
{
	assert(sub);
	assert(bif);
	assert(target_d>0 && target_d<=sub->get_n());
	RETURNTYPE value;
	DIMTYPE f;
	
	if(target_d!=0 && sub->get_d()!=target_d) {
		if(grandparent::output_normal()) this->report(os, msgbuf() << "original initial "<< *sub << std::endl << std::flush);
		// collect the BIF ranking (best -> worst) once; an undersized subset grows by
		// selecting best-ranked features, an oversized one shrinks by deselecting
		// WORST-ranked features - hence the reverse iteration in the shrink branch
		std::vector<DIMTYPE> ranked;
		if(!bif->getFirstFeature(value, f)) return false;
		do {
			assert(f<=sub->get_n());
			ranked.push_back(f);
		} while(bif->getNextFeature(value, f));
		if(sub->get_d()<target_d) {
			for(typename std::vector<DIMTYPE>::const_iterator it=ranked.begin();it!=ranked.end() && sub->get_d()!=target_d;++it) sub->select(*it);
		} else {
			for(typename std::vector<DIMTYPE>::const_reverse_iterator it=ranked.rbegin();it!=ranked.rend() && sub->get_d()!=target_d;++it) sub->deselect(*it);
		}
		if(sub->get_d()!=target_d) return false;
		if(grandparent::output_normal()) this->report(os, msgbuf() << "adjusted initial "<< *sub << std::endl << std::endl << std::flush);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	// NOTE: certain parameter combinations (initial sub->get_d() versus _delta versus target_d versus parent::_generalization_level) 
	//       make it impossible to yield a valid solution. In such case search() just returns false
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d<sub->get_n());
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	if(!parent::initial_subset_enabled()) {sub->set_forward_mode(true); sub->deselect_all();} // deselect_all() writes mode-dependent ids: switch to forward mode FIRST, or a Subset left in backward mode by a previous search comes up in the wrong state

	if( (_delta<parent::get_generalization_level()) ) return false; // impossible to run the search

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

	PSearchBIF bif;
	DIMTYPE _n=sub->get_n();

	if(target_d==0) { // optimize the cardinality: run one full oscillating search per subset size
	                  // and keep the best result over all of them (at d==_n there is nothing to
	                  // oscillate, so that one subset is merely evaluated)
		if(!inisub || inisub->get_n()!=_n) inisub.reset(new SUBSET(_n));
		if(!tmpsub || tmpsub->get_n()!=_n) tmpsub.reset(new SUBSET(_n));
		RETURNTYPE value;
		if(!bif) {
			bif.reset(new SearchBIF());
			if(!bif->evaluate_individuals(_n, crit, os)) return false;
		}
		inisub->stateless_copy(*sub);
		for(DIMTYPE d=1;d<=_n;d++) {
			tmpsub->stateless_copy(*inisub);
			if(!adjust_initial_subset(d,tmpsub,bif,os)) return false;
			if(d<_n) {if(!search_one(value,tmpsub,crit,os)) return false;} else {if(!crit->evaluate(value,tmpsub)) return false;}
			if(d==1 || value>result) {
				result=value;
				sub->stateless_copy(*tmpsub);
				if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "NEW MAXVAL (d="<<d<<") ="<<result<<", " << *sub << std::endl << swatch << std::endl << std::endl << std::flush);
			} else
				if(grandparent::output_normal()) this->report_solution(os, value, *tmpsub, msgbuf() << "result (d="<<d<<") ="<<value<<", " << *tmpsub << std::endl << std::endl << std::flush);
		}
	} else { // evaluate subset of target_d features
		if(sub->get_d()!=target_d) {
			if(!bif) {
				bif.reset(new SearchBIF);
				if(!bif->evaluate_individuals(_n,crit,os)) return false;
			}
			if(!adjust_initial_subset(target_d,sub,bif,os)) return false;
		}
		if(!search_one(result,sub,crit,os)) return false;
	}
	if(grandparent::output_normal()) {
		msgbuf sos; 
		sos << "......................................." << std::endl;
		sos << "Search_OS() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search_one(RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	notify("Search_OS::search_one().");
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(sub->get_d()>0);
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	if( (sub->get_d()==0) ||
		(_delta<parent::get_generalization_level()) ) return false; // impossible to run the search

	DIMTYPE _n=sub->get_n();
	if(!pivotsub || pivotsub->get_n()!=_n) pivotsub.reset(new SUBSET(_n));
	if(!pivot_top || pivot_top->get_n()!=_n) pivot_top.reset(new SUBSET(_n));
	if(!pivot_bottom || pivot_bottom->get_n()!=_n) pivot_bottom.reset(new SUBSET(_n));

	RETURNTYPE bestval;
	
	// sub is assumed to contain the initial subset
	sub->set_forward_mode(true);

	// The initial subset's value seeds the pivot, so this failure MUST be checked: continuing
	// with an unset result would start the oscillation from an indeterminate pivot.
	// NOTE this value comes from crit, while a Sequential_Step_Hybrid with keep_perc==0 selects
	// by its filter criterion - which is why that combination is unsupported (see the class docs).
	if(!crit->evaluate(result,sub)) return false; // the initial subset is the initially best pivot
	if(track) grandparent::_tracker->add(result,sub);

	SwingPhase swing=RESTART;

	DIMTYPE _d=sub->get_d();
	DIMTYPE _d_piv=_d;
	DIMTYPE _cur_delta=parent::get_generalization_level();
	bool pivot_top_avail=false;
	bool pivot_bottom_avail=false;

	bool swinged=false; // whether the current swing managed to take at least one step

	// the oscillation itself: a state machine over SwingPhase (documented at its declaration),
	// looping until the cycle depth _cur_delta has grown past the configured maximum _delta
	while(_cur_delta<=_delta)
		switch(swing) {
		case RESTART:
			assert(_d_piv==sub->get_d());
			pivotsub->stateless_copy(*sub); bestval=result;
			pivot_top_avail=pivot_bottom_avail=false;
			_cur_delta=parent::get_generalization_level();
			swing=DOWN;
			if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "new subresult ="<<result<<", " << *sub << std::endl << std::endl << std::flush);
			break;
		case DOWN:
			assert(_d==sub->get_d());
			assert(sub->get_d()!=0);
			if(pivot_bottom_avail) {assert(pivot_bottom->get_d()!=0); sub->stateless_copy(*pivot_bottom); _d=sub->get_d();}
			swinged=false;
			while(_d>parent::get_generalization_level() && _d_piv<_cur_delta+_d){
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "DOWN failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
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
			while(_d<_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "RETURN UP failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
			}
			if(result>bestval) swing=RESTART; // down-swing successful - reset
			else {
				sub->stateless_copy(*pivotsub);
				swing=UP; swinged=false;
			}
			assert(_d==sub->get_d());
			break;
		case UP:
			assert(_d==sub->get_d());
			if(pivot_top_avail) {assert(pivot_top->get_d()!=0); sub->stateless_copy(*pivot_top); _d=sub->get_d();}
			while(_d+parent::get_generalization_level()<=_n && _d<_cur_delta+_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "UP failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
			}
			if(swinged) {
				assert(_d==sub->get_d());
				if(_d+parent::get_generalization_level()<=_n) {
					pivot_top_avail=true;
					pivot_top->stateless_copy(*sub);
				}
				swing=RETURN_DOWN;
			} else { // the UP swing could not move at all: deepen the cycle and try DOWN again
				pivot_top_avail=false;
				swing=DOWN;
				_cur_delta+=parent::get_generalization_level();
			}
			assert(_d==sub->get_d());
			break;
		case RETURN_DOWN:
			assert(_d==sub->get_d());
			while(_d>_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) this->report(os, msgbuf() << "RETURN DOWN failure" << std::endl << std::flush); return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
			}
			if(result>bestval) swing=RESTART; // up-swing successful - reset
			else { // neither swing improved at this depth: go back to the pivot and deepen the cycle
				sub->stateless_copy(*pivotsub);
				swing=DOWN; swinged=false;
				_cur_delta+=parent::get_generalization_level();
			}
			assert(_d==sub->get_d());
			break;
		}

	// the loop ends once the cycle depth exceeds _delta; the pivot is then the best subset found
	if(pivotsub->get_d()!=0) {
		sub->stateless_copy(*pivotsub);
		result=bestval;
		return true;
	}
	return false; // an empty pivot means no valid subset was ever adopted
}


} // namespace
#endif // FSTSEARCHSEQOS_H ///:~
