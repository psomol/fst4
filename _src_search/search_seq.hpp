#ifndef FSTSEARCHSEQ_H
#define FSTSEARCHSEQ_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_seq.hpp
   \brief   Defines interface for sequential search method implementations
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
#include "search.hpp"

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

/*! \brief Direction a sequential search proceeds in.

    FORWARD searches grow the subset (SFS, SFFS, SFRS, and the up-swing half of
    OS/DOS); BACKWARD searches shrink it (SBS, SBFS, SBRS, down-swings).
    UNDEFINED is only the pre-construction placeholder and is never a valid
    argument to set_search_direction(). */
enum SearchDirection {
  UNDEFINED=0,
  FORWARD=1,
  BACKWARD=2
};

/*! \brief Abstract class, defines interface for sequential search method implementations, provides common implementation of the "sequential step" operation

	\note To prevent misunderstanding: _generalization_level here does not relate to 
	classification performance on unknown data, but to "generalization" of the course of search,
	i.e., the number of features added/removed while generating candidate subsets in one sequential step
	(higher generalization=more candidates). See book Devijver, Kittler, 1982 for discussion.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_Sequential : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{ // abstract class
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef std::shared_ptr<EVALUATOR> PEvaluator; //!< shared pointer to the sequential-step evaluator that implements Step()'s candidate generation and selection
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	Search_Sequential(const PEvaluator evaluator):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(), _generalization_level(1), _initial_subset_enabled(false) {assert(evaluator); _evaluator=evaluator; notify("Search_Sequential constructor.");} //!< \p evaluator must be non-null
	virtual ~Search_Sequential() {notify("Search_Sequential destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) override =0; //!< returns the found subset of target_d features + its criterion value; target_d==0 asks the method to pick the best subset size itself (it then keeps the best-valued subset over all sizes it visits)
	
	/*! \brief Performs one sequential step: adds (\p forward) or removes (!\p forward) the best g-tuple of features, g being the generalization level.

		The work is done by the injected evaluator, which replaces \p sub's contents
		by the candidate it chose and stores its criterion value in \p result. The
		subset's forward/backward mode is switched for the duration of the call and
		restored afterwards, so a backward step inside a forward search costs nothing
		but the step itself.
		\return the evaluator's success flag: false if it could not generate a
		candidate (nothing left to add or remove) or a criterion evaluation failed.
		\p sub is the traversal state of the candidate enumeration and may have been
		modified even then - it is the caller's business to abandon the search. */
	inline bool Step(const bool forward, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout);

	void set_generalization_level(const DIMTYPE g) {if(g>=1) _generalization_level=g;} //!< optional parameter to enlarge the number of considered candidates (g is the number of features added/removed in one Step)
	DIMTYPE get_generalization_level() const {return _generalization_level;} //!< the currently configured generalization level (see set_generalization_level())

	// to be used only if the search is intended to be started from a pre-specified initial subset, otherwise the standard FORWARD methods start from empty set and BACKWARD methods start from full set
	void enable_initial_subset() {_initial_subset_enabled=true;} //!< makes search() start from the caller-supplied contents of its \c sub argument instead of the concrete method's default (empty set for FORWARD, full set for BACKWARD)
	void disable_initial_subset() {_initial_subset_enabled=false;} //!< reverts to the default: incoming \c sub contents are ignored by search()
	bool initial_subset_enabled() const {return _initial_subset_enabled;} //!< true if enable_initial_subset() was called (and not since disabled by disable_initial_subset())

protected:
	PEvaluator _evaluator; //!< the sequential-step evaluator injected at construction; drives Step()'s candidate generation and best/worst choice
	DIMTYPE _generalization_level; //!< number of features added/removed together in one Step() call (see set_generalization_level())
	bool _initial_subset_enabled; //!< by default all concrete search methods ignore the contents of supplied 'PSubset set' and initialize sub according to their definition; if enabled, then (if possible) the search() starts from supplied 'PSubset sub' contents
};


template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::Step(const bool forward, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // modifies sub, returns the evaluator's success flag
{
	assert(_evaluator);
	assert(sub);
	assert(crit);

	if(parent::output_detailed()) this->report(os, msgbuf() << "Search_Sequential::Step(): sub->get_d()="<<sub->get_d()<<" forward="<<forward <<" sub=" << *sub<<std::endl);

	bool success=true;
	bool _mode_change=(sub->get_forward_mode()!=forward);

	if(_mode_change) sub->set_forward_mode(forward);
	if(!_evaluator->evaluate_candidates(result, sub, crit, _generalization_level,os)) success=false;
	if(_mode_change) sub->set_forward_mode(!forward);

	return success;
}


} // namespace
#endif // FSTSEARCHSEQ_H ///:~
