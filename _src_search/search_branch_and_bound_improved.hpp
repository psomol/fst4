#ifndef FSTSEARCHBRANCHANDBOUNDIMPROVED_H
#define FSTSEARCHBRANCHANDBOUNDIMPROVED_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_branch_and_bound_improved.hpp
   \brief   Implements Improved Branch and Bound (IBB) method, i.e., with fully computed node ordering
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
#include "search_branch_and_bound.hpp"
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"

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

/*! \brief Implements Improved Branch and Bound (IBB) method, i.e., with fully computed node ordering

	IBB is the well-known B&B variant that orders tree nodes in each constructed tree
	level so as to increase the chance that the bound gets higher soon. This effectively
	improves chances that larger sub-trees can be cut and thus more time saved, but
	this is achieved at the cost of additional criterion evaluations in each tree level.
	In time consuming tasks this mechanism proves capable of considerably overperforming the
	Basic Branch & Bound algorithm.

	\warning All Branch & Bound feature selection algorithms require the used
	CRITERION to be monotonic with respect to cardinality. More precisely, it must
	hold that removing a feature from a set MUST NOT increase criterion value.
	Otherwise there is no guarantee as of the optimality of obtained results
	with respect to the used criterion.

	\note All Branch & Bound algorithms by definition yield the solution with the 
	same maximum criterion value, therefore the main concern regarding particular
	Branch & Bound algorithm is only its search speed.

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. 
	
	\note Result tracking in case of Branch & Bound algorithms records only results
	of target cardinality.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Branch_And_Bound_Improved : public Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef typename parent::PCriterion PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef typename parent::PSubset PSubset; //!< shared pointer to a feature subset
	typedef typename parent::PNode PNode; //!< pointer to a search-tree Node, see Search_Branch_And_Bound
	typedef typename parent::Node Node; //!< search-tree node record (feature, value, type, valid), see Search_Branch_And_Bound
	typedef typename parent::NodeType NodeType; //!< UNKNOWN/RANDOM/PREDICTED/COMPUTED node-value provenance, see Search_Branch_And_Bound
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< shared pointer to the injected evaluation-strategy engine
	explicit Search_Branch_And_Bound_Improved(PEvaluatorEngine engine=PEvaluatorEngine()):Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(engine) {notify("Search_Branch_And_Bound_Improved constructor.");} //!< \p engine forwarded to Search_Branch_And_Bound; defaults to sequential in-line evaluation if not supplied
	virtual ~Search_Branch_And_Bound_Improved() {notify("Search_Branch_And_Bound_Improved destructor.");}

	// for search() implementation see parent class

	virtual std::ostream& print(std::ostream& os) const override {
		if(parent::get_engine()->is_threaded()) os << "Improved Branch & Bound (IBB)  [Search_Branch_And_Bound_Improved(threads="<<parent::get_engine()->get_max_threads()<<")]";
		else os << "Improved Branch & Bound (IBB)  [Search_Branch_And_Bound_Improved()]";
		return os;
	}

protected:
	// the following may be overriden in descendant Branch and Bound specialization classes
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit) override {} //!< IBB keeps no structures of its own beyond the base class's tree, so there is nothing to set up
	virtual void pre_evaluate_availables() override; //!< evaluates the criterion for every available feature (one batch through the evaluation engine) so that the next tree level can be ordered by true criterion values - this is what IBB buys its cuts with
	virtual void post_process_tree_level() override{} //!< nothing to do: every node of the new level already carries a COMPUTED value from pre_evaluate_availables()
	virtual bool cut_possible() override; //!< reports whether the current node's (always computed) value fails to exceed the bound, in which case the whole sub-branch below it can be cut
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound_Improved<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::pre_evaluate_availables()
{
	assert(parent::get_criterion());
	assert(parent::get_engine());
	const PSubset &currentset=parent::get_currentset();
	// pre-evaluate available features before next tree level construction;
	// computed values are written back into the corresponding _available node,
	// keyed by the feature tag (Node::feature indexes _available by position)
	typename parent::EVALUATORENGINE::ResultCallback on_result=[this](const RETURNTYPE value, const PSubset&, const DIMTYPE feat) {
		PNode avail;
		if(this->getAvailable(feat,avail)) {
			avail->value=value;
			avail->type=parent::COMPUTED;
		}
	};
	parent::get_engine()->initialize(parent::get_criterion(), parent::get_n(), on_result);
	PNode avail;
	for(bool got=this->getFirstAvailable(avail);got;got=this->getNextAvailable(avail))
	{
		assert(currentset->selected_raw(avail->feature));
		currentset->deselect_raw(avail->feature);
		if(!parent::get_engine()->add_candidate(currentset,avail->feature)) throw FST::fst_error("Criterion evaluation failure.");
		currentset->select_raw(avail->feature);
	}
	// fail as loudly as the sequential path's add_candidate throw above: a criterion
	// exception thrown on a worker thread is rethrown right here on the master, and a
	// failed evaluation marks the whole batch failed - in which case no on_result
	// callback fired, every available stays UNKNOWN with a stale value, and the level
	// sort-in would feed cut_possible() garbage (it expects COMPUTED nodes)
	if(!parent::get_engine()->collect()) throw FST::fst_error("Criterion evaluation failure.");
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Branch_And_Bound_Improved<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::cut_possible()
{
	if(parent::is_bound_valid()) {
		// the node's value is always COMPUTED here: IBB evaluates every available feature
		// in pre_evaluate_availables() before the level is built, so - unlike in FBB - no
		// node can reach this point carrying a merely predicted value
		Node &currentnode=parent::get_current_node(); assert(currentnode.type==parent::COMPUTED);
		if(currentnode.value<=parent::get_bound_value()) {
			return true;
		}
	}
	return false;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDIMPROVED_H ///:~
