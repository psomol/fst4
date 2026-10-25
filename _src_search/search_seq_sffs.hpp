#ifndef FSTSEARCHSEQSFFS_H
#define FSTSEARCHSEQSFFS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_seq_sffs.hpp
   \brief   Implements Sequential_Forward_Floating_Search and Sequential_Backward_Floating_Search
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

/*! \brief Implements Sequential_Forward_Floating_Search and Sequential_Backward_Floating_Search

	\note By default any initial subset contents are ignored; SFFS starts from empty set and SBFS starts from full set. 
	      Call enable_initial_subset() to let the search start from the initial sub contents.

    \note This implementation follows the original SFFS definition closely. See Search_SFRS
          for a closely related, slower, but more thorough method.

    \note Among subsets of EQUAL criterion value the search prefers the SMALLER one, both when
          updating the best-so-far used for the target_d==0 (optimize cardinality) answer and,
          implicitly, in that a per-cardinality record is only replaced by a strictly better value.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_SFFS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent; //!< base class alias
	typedef typename parent::parent grandparent; //!< the shared Search base (output detail, result tracker)
	typedef std::shared_ptr<EVALUATOR> PEvaluator; //!< shared pointer to the sequential-step evaluator
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	Search_SFFS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator), _direction(FORWARD), _delta(0), _n(0) {notify("Search_SFFS constructor.");} //!< \p evaluator must be non-null; direction defaults to FORWARD (SFFS); call set_search_direction(BACKWARD) for SBFS
	virtual ~Search_SFFS() {notify("Search_SFFS destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override; //!< returns found subset + criterion value

	void set_search_direction(const SearchDirection direction) {_direction=direction;} //!< FORWARD selects SFFS, BACKWARD selects SBFS (do not change concurrently to a running search())
	SearchDirection get_search_direction() const {return _direction;} //!< the currently configured direction (see set_search_direction())

	void set_delta(const DIMTYPE delta) {_delta=delta;} //!< optional parameter to restrict search extent: bounds the floating frontier to at most target_d+delta features forward and at least target_d-delta backward (0 = unrestricted, the classic full floating frontier); has no effect when target_d==0 (best-dimensionality search)
	DIMTYPE get_delta() const {return _delta;} //!< the currently configured delta bound (see set_delta())

	bool get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub); //!< retrieves temporary solution of cardinality d that had been updated in the course of the last search (if it exists)
	
	virtual std::ostream& print(std::ostream& os) const override;
protected:
	SearchDirection _direction; //!< accepted values FORWARD for SFFS or BACKWARD for SBFS
	//! Structure to hold [subset,criterion value] temporary solutions in the course of search
	struct OneSubset {
		RETURNTYPE critvalue; //!< the subset's criterion value
		std::unique_ptr<SUBSET> sub; //!< the subset itself, or null if no result was recorded at this cardinality
	};
	DIMTYPE _delta; //!< frontier restriction around target_d (0 = unrestricted); see set_delta()
	std::unique_ptr<OneSubset[]> bsubs; //!< per-cardinality best-known result found in the course of the last search() (index d-1); read by get_result()
	DIMTYPE _n; //!< bsubs array size
	std::unique_ptr<SUBSET> pivotsub; //!< the subset the next forward step continues from: the deepest backtracking result that improved its cardinality's record. NOTE the name means something else in Search_DOS, where "pivot" denotes the best-so-far subset - that one is called maxcritsub here
	std::unique_ptr<SUBSET> maxcritsub; //!< stores the subset with maximum known crit value
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	if(_direction==FORWARD) os << "SFFS"; else os << "SBFS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << ", delta=" << _delta;
	os << "  [Search_SFFS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	// NOTE: certain parameter combinations (initial sub->get_d() versus target_d versus parent::_generalization_level) 
	//       make it impossible to yield a valid solution. In such case search() just returns false
	// NOTE: so far this is the most original version of SFFS, i.e., it does 
	//       not "correct misleading paths" ("worse-than-before" forward steps after successful backtracking) etc.
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	bool track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d<sub->get_n());
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	const bool _forward=(_direction==FORWARD);
	sub->set_forward_mode(true); // select_all()/deselect_all() write mode-dependent ids - a Subset left in backward mode by a previous search() would otherwise re-initialize empty
	if(!parent::initial_subset_enabled()) {
		if(_forward) sub->deselect_all(); else sub->select_all();
	}
	DIMTYPE _d=sub->get_d();
	if( (!_forward && _d==0) ||
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

	_n=sub->get_n();
	bsubs.reset(new OneSubset[_n]);
	if(!pivotsub || pivotsub->get_n()!=_n) pivotsub.reset(new SUBSET(_n));
	if(!maxcritsub || maxcritsub->get_n()!=_n) maxcritsub.reset(new SUBSET(_n));
	RETURNTYPE maxcritval;
	bool havemax=false;
	DIMTYPE _d_max;

	sub->set_forward_mode(_forward); // the search's own direction; Step() flips it per step as needed
	if(_d>0) { // evaluate initial subset if any
		if(!crit->evaluate(result,sub)) return false;
		if(track) grandparent::_tracker->add(result,sub);
		bsubs[_d-1].sub.reset(new SUBSET(_n));
		bsubs[_d-1].sub->stateless_copy(*sub);
		bsubs[_d-1].critvalue=result;
		maxcritsub->stateless_copy(*sub); 
		maxcritval=result; 
		_d_max = maxcritsub->get_d();
		havemax=true;
		if(grandparent::output_normal()) this->report_solution(os, maxcritval, *maxcritsub, msgbuf() << "initial MAXCRIT="<<maxcritval<<", " << *maxcritsub << std::endl << swatch << std::endl << std::endl << std::flush);
	}

	// Cardinality window the floating frontier may move in. Without a delta bound it is the
	// full 1.._n range; with one it narrows to target_d-delta .. target_d+delta, clipped to
	// that full range. A delta is meaningless when the cardinality itself is being optimized
	// (target_d==0), so it is ignored then.
	const DIMTYPE _forward_thr=(target_d>0 && _delta>0 && target_d+_delta<_n)?target_d+_delta:_n;
	const DIMTYPE _backward_thr=(target_d>0 && _delta>0 && _delta<target_d)?target_d-_delta:1;
	while( (_forward && _d+parent::get_generalization_level()<=_forward_thr) || (!_forward && _d>=_backward_thr+parent::get_generalization_level()) ) {
		// unconditional step IN THE SEARCH'S OWN DIRECTION (adding a feature in SFFS, removing
		// one in SBFS); the conditional backtracking below always steps the opposite way
		if(!parent::Step(_forward, result, sub, crit, os)) return false;
		if(track) grandparent::_tracker->add(result,sub);
		_d=sub->get_d();

		// tie rule: at EQUAL criterion value the smaller subset wins
		if((!havemax) || (result>maxcritval) || (result==maxcritval && _d<_d_max)) {
			maxcritsub->stateless_copy(*sub); maxcritval=result; _d_max=_d; havemax=true;
			if(grandparent::output_normal()) this->report_solution(os, maxcritval, *maxcritsub, msgbuf() << "new MAXCRIT="<<maxcritval<<", " << *maxcritsub << std::endl << swatch << std::endl << std::endl << std::flush);
		}
		assert(_d-1<_n);
		if(!bsubs[_d-1].sub) { // first result in that dimensionality
			bsubs[_d-1].sub.reset(new SUBSET(_n));
			bsubs[_d-1].sub->stateless_copy(*sub);
			bsubs[_d-1].critvalue=result;
			if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "new subresult["<<_d<<"]="<<result<<", " << *sub << std::endl << std::endl << std::flush);
		} else {
			if(result > bsubs[_d-1].critvalue) { // better than before found for that dimensionality
				bsubs[_d-1].sub->stateless_copy(*sub);
				bsubs[_d-1].critvalue=result;
				if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "new subresult["<<_d<<"]="<<result<<", " << *sub << std::endl << std::endl << std::flush);
			}
		}
		
		pivotsub->stateless_copy(*sub);
		// Conditional BACKTRACKING: keep stepping back as long as each step still IMPROVES the
		// record of the cardinality it lands on, remembering the last improving subset in
		// pivotsub. The loop ends on the first step that does not improve (backtrack=false) or
		// when the cardinality bound is reached; the outer loop then resumes from pivotsub, so a
		// non-improving excursion is discarded. A record is replaced only on a strictly better
		// value, hence an equally good subset does not prolong the backtracking either.
		bool backtrack=true;
		while(backtrack && ((_forward && _d>=1+parent::get_generalization_level()) || (!_forward && _d+parent::get_generalization_level()<=_n))) {
			if(!parent::Step(!_forward, result, sub, crit, os)) return false;
			if(track) grandparent::_tracker->add(result,sub);
			_d=sub->get_d();

			if((!havemax) || (result>maxcritval) || (result==maxcritval && _d<_d_max)) {
				maxcritsub->stateless_copy(*sub); maxcritval=result; _d_max=_d; havemax=true;
				if(grandparent::output_normal()) this->report_solution(os, maxcritval, *maxcritsub, msgbuf() << "new MAXCRIT="<<maxcritval<<", " << *maxcritsub << std::endl << swatch << std::endl << std::endl << std::flush);
			}
			assert(_d-1<_n);
			if(!bsubs[_d-1].sub) { // first result in that dimensionality
				bsubs[_d-1].sub.reset(new SUBSET(_n));
				bsubs[_d-1].sub->stateless_copy(*sub);
				bsubs[_d-1].critvalue=result;
				if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "new subresult["<<_d<<"]="<<result<<", " << *sub << std::endl << std::endl << std::flush);
			} else {
				if(result > bsubs[_d-1].critvalue) { // better than before found for that dimensionality
					bsubs[_d-1].sub->stateless_copy(*sub);
					bsubs[_d-1].critvalue=result;
					pivotsub->stateless_copy(*sub);
					if(grandparent::output_normal()) this->report_solution(os, result, *sub, msgbuf() << "new subresult["<<_d<<"]="<<result<<", " << *sub << std::endl << std::endl << std::flush);
				} else backtrack=false;
			}
		}
		sub->stateless_copy(*pivotsub);
		_d=sub->get_d();
	}
	if(target_d>0) { // return subset of size target_d
		assert(target_d>=1 && target_d<=_n);
		if(!bsubs[target_d-1].sub) return false;
		result=bsubs[target_d-1].critvalue;
		sub->stateless_copy(*bsubs[target_d-1].sub);
	} else { // find best dimensionality
		if(!havemax) return false;
		sub->stateless_copy(*maxcritsub);
		result=maxcritval;
	}
	if(grandparent::output_normal()) {
		msgbuf sos; 
		sos << "......................................." << std::endl;
		sos << "Search_SFFS() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub) // retrieves temporary solution of cardinality d that had been updated in the course of the last search
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
#endif // FSTSEARCHSEQSFFS_H ///:~
