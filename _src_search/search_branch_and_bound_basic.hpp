#ifndef FSTSEARCHBRANCHANDBOUNDBASIC_H
#define FSTSEARCHBRANCHANDBOUNDBASIC_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_branch_and_bound_basic.hpp
   \brief   Implements Branch and Bound Basic method, i.e., with randomized node ordering
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
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
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

/*! \brief Implements Branch and Bound Basic method, i.e., with randomized node ordering

	BBB is the simplest and generally slowest of all B&B algorithms.

	\note All Branch & Bound algorithms by definition yield the solution with the 
	same maximum criterion value, therefore the main concern regarding particular
	Branch & Bound algorithm is only its search speed.

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. 
	
	\note Result tracking in case of Branch & Bound algorithms records only results
	of target cardinality.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Branch_And_Bound_Basic : public Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef typename parent::PCriterion PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef typename parent::PSubset PSubset; //!< shared pointer to a feature subset
	typedef typename parent::PNode PNode; //!< pointer to a search-tree Node, see Search_Branch_And_Bound
	typedef typename parent::Node Node; //!< search-tree node record (feature, value, type, valid), see Search_Branch_And_Bound
	typedef typename parent::NodeType NodeType; //!< UNKNOWN/RANDOM/PREDICTED/COMPUTED node-value provenance, see Search_Branch_And_Bound
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< shared pointer to the injected evaluation-strategy engine
	explicit Search_Branch_And_Bound_Basic(PEvaluatorEngine engine=PEvaluatorEngine()):Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(engine) {notify("Search_Branch_And_Bound_Basic constructor.");} //!< \p engine forwarded to Search_Branch_And_Bound; defaults to sequential in-line evaluation if not supplied
	virtual ~Search_Branch_And_Bound_Basic() {notify("Search_Branch_And_Bound_Basic destructor.");}

	// for search() implementation see parent class

	//! writes the method's name and, with a threaded engine, the number of worker threads
	virtual std::ostream& print(std::ostream& os) const override {
		if(parent::get_engine()->is_threaded()) os << "Basic Branch & Bound (BBB)  [Search_Branch_And_Bound_Basic(threads="<<parent::get_engine()->get_max_threads()<<")]";
		else os << "Basic Branch & Bound (BBB)  [Search_Branch_And_Bound_Basic()]";
		return os;
	};

protected:
	// the following may be overriden in descendant Branch and Bound specialization classes
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit) override {} //!< BBB keeps no structures of its own beyond the base class's tree, so there is nothing to set up
	virtual void pre_evaluate_availables() override; //!< BBB does not order the tree by any estimate of feature importance: every available feature gets a random ordering value (type UNKNOWN), which only avoids systematically pathological node orders
	virtual void post_process_tree_level() override; //!< the ordering values above are random, not criterion values, so every node chosen for the new level still needs its true criterion value - they are computed here as one batch through the evaluation engine
	virtual bool cut_possible() override; //!< computes the current node's criterion value if it is not known yet and reports whether the node's value fails to exceed the current bound - in which case the whole sub-branch below it can be cut
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound_Basic<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::pre_evaluate_availables() 
{
	const PSubset &currentset=parent::get_currentset(); assert(currentset);
	// pre-evaluate available features before next tree level construction
	PNode avail;
	for(bool got=this->getFirstAvailable(avail);got;got=this->getNextAvailable(avail)) 
	{
		avail->value=(RETURNTYPE)fst_rand(); // in basic B&B ordering is not defined, randomization here aims at preventing worst-case scenarios
		avail->type=parent::UNKNOWN;
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound_Basic<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::post_process_tree_level()
{
	assert(parent::get_criterion());
	assert(parent::get_engine());
	const PSubset &currentset=parent::get_currentset();	assert(currentset);
	// supply the true criterion value of the level's chosen nodes (BBB pre-orders by
	// random values, so every chosen node needs computing) - batched through the
	// engine. Results map back to their nodes by submission order: the per-result
	// callback fires in add-order under both engine implementations (the submitted
	// vector is fully built before the first add_candidate because the sequential
	// engine fires callbacks synchronously inside add_candidate); list node pointers
	// stay stable (no insert/erase during the batch). A threaded run returns the same
	// tree as a sequential one - candidate generation (and the fst_rand pre-ordering)
	// stays on the master thread
	PNode nod;
	std::vector<PNode> submitted;
	for(bool got=this->getFirstNode(nod);got;got=this->getNextNode(nod))
		if(nod->type!=parent::COMPUTED) submitted.push_back(nod);
	if(submitted.empty()) return;

	DIMTYPE done=0;
	typename parent::EVALUATORENGINE::ResultCallback on_result=[&submitted,&done](const RETURNTYPE value, const PSubset&, const DIMTYPE feat) {
		PNode node=submitted[done++];
		assert(node->feature==feat); (void)feat;
		node->value=value;
		node->type=parent::COMPUTED;
	};
	parent::get_engine()->initialize(parent::get_criterion(),(DIMTYPE)submitted.size(),on_result);
	for(PNode snod : submitted)
	{
		assert(currentset->selected_raw(snod->feature));
		currentset->deselect_raw(snod->feature);
		if(!parent::get_engine()->add_candidate(currentset,snod->feature)) throw FST::fst_error("Criterion evaluation failure.");
		currentset->select_raw(snod->feature);
	}
	// a failed batch must not leave nodes UNKNOWN on the tree path - fail the search instead
	if(!parent::get_engine()->collect()) throw FST::fst_error("Criterion evaluation failure.");
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Branch_And_Bound_Basic<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::cut_possible()
{	// NOTE: the lazy compute below deliberately stays on the master thread -
	// it is a single evaluation on the traversal's critical path (nothing to batch)
	if(parent::is_bound_valid()) {
		Node &currentnode=parent::get_current_node();
		if(currentnode.type!=parent::COMPUTED) {
			const PSubset &currentset=parent::get_currentset();	assert(currentset);
			assert(parent::get_criterion());
			assert(currentnode.feature<parent::get_n());
			assert(currentset->selected_raw(currentnode.feature));
			currentset->deselect_raw(currentnode.feature);
			if(!parent::get_criterion()->evaluate(currentnode.value,currentset)) throw FST::fst_error("Criterion evaluation failure."); 
			currentset->select_raw(currentnode.feature);
			currentnode.type=parent::COMPUTED;
		}
		if(currentnode.value<=parent::get_bound_value()) {
			return true;
		}
	}
	return false;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDBASIC_H ///:~
