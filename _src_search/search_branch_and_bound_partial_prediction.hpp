#ifndef FSTSEARCHBRANCHANDBOUNDPARTIALPREDICTION_H
#define FSTSEARCHBRANCHANDBOUNDPARTIALPREDICTION_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_branch_and_bound_partial_prediction.hpp
   \brief   Implements Branch and Bound with Partial Prediction (BBPP) method, i.e., with predicted node ordering
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
#include "branch_and_bound_predictor.hpp"
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

/*! \brief Implements Branch and Bound with Partial Prediction (BBPP) method, i.e., with predicted node ordering

	BBPP is generally slower than FBB but faster than IBB. Its principle is
	identical to that of Improved Branch & Bound (IBB), but it replaces
	large amount of criterion evaluations needed for ordering nodes in tree levels
	by quickly predicted values. The optimality of final result is not jeopardized
	but considerable amount of time is saved. See FBB for a more radical but still
	optimality preserving use of value prediction.

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

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
class Search_Branch_And_Bound_Partial_Prediction : public Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef typename parent::PCriterion PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef typename parent::PSubset PSubset; //!< shared pointer to a feature subset
	typedef typename parent::PNode PNode; //!< pointer to a search-tree Node, see Search_Branch_And_Bound
	typedef typename parent::Node Node; //!< search-tree node record (feature, value, type, valid), see Search_Branch_And_Bound
	typedef typename parent::NodeType NodeType; //!< UNKNOWN/RANDOM/PREDICTED/COMPUTED node-value provenance, see Search_Branch_And_Bound
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< shared pointer to the injected evaluation-strategy engine
	explicit Search_Branch_And_Bound_Partial_Prediction(PEvaluatorEngine engine=PEvaluatorEngine()):Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(engine) {notify("Search_Branch_And_Bound_Partial_Prediction constructor.");} //!< \p engine forwarded to Search_Branch_And_Bound; defaults to sequential in-line evaluation if not supplied
	virtual ~Search_Branch_And_Bound_Partial_Prediction() {notify("Search_Branch_And_Bound_Partial_Prediction destructor.");}

	// for search() implementation see parent class

	//! Writes the algorithm's name together with the prediction mechanism in use, and the thread count when a threaded evaluation engine is injected.
	virtual std::ostream& print(std::ostream& os) const override {
		if(parent::get_engine()->is_threaded()) os << "Branch & Bound with Partial Prediction (BBPP)  [Search_Branch_And_Bound_Partial_Prediction(threads="<<parent::get_engine()->get_max_threads()<<") with " << _predictor << "]";
		else os << "Branch & Bound with Partial Prediction (BBPP)  [Search_Branch_And_Bound_Partial_Prediction() with " << _predictor << "]";
		return os;
	};
protected:
	PREDICTOR _predictor; //!< the prediction mechanism instance (PREDICTOR type parameter, e.g. Branch_And_Bound_Predictor_Averaging); used only to ORDER each tree level's nodes (unlike FBB, BBPP always computes the true value along the tree path before deciding to cut, so optimality never relies on the prediction being accurate)

	// the following may be overriden in descendant Branch and Bound specialization classes
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit) override {_predictor.initialize(n);} //!< resets the prediction mechanism for a feature space of \p n features; nothing else needs setting up for BBPP
	virtual void leaf_result(const DIMTYPE feature, const RETURNTYPE value) override; //!< feeds one leaf's contrast (parent value minus leaf value) to the predictor, so the leaf level - the cheapest and most numerous evidence in the tree - also trains the ordering predictions
	//! \brief Assigns an ordering value to every available feature: predicted (parent value minus the
	//! predicted contrast) where the predictor already has evidence for that feature, otherwise the exact
	//! criterion value, obtained in one batch through the evaluation engine and fed back to the predictor.
	virtual void pre_evaluate_availables() override;
	//! \brief Replaces the merely PREDICTED values of the nodes that made it into the new tree level by
	//! exact criterion values (again in one engine batch). This is what makes the method PARTIAL
	//! prediction: predictions decide the ORDER of the level only, while every node on the tree path
	//! carries a true value, so a cut is never based on an estimate and optimality is preserved.
	virtual void post_process_tree_level() override;
	//! \brief Returns true when the current node's value does not exceed the current bound, so the whole
	//! sub-branch below it can be cut. The value is always exact here (post_process_tree_level() has made
	//! it COMPUTED), so no lazy evaluation is needed; returns false while no bound is valid yet.
	virtual bool cut_possible() override;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
void Search_Branch_And_Bound_Partial_Prediction<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::leaf_result(const DIMTYPE feature, const RETURNTYPE value)
{
	// conditionally update prediction info: the base class fires this hook after each leaf's
	// bound update, in leaf order, so the parent node's value is the right reference point
	Node &parentnode=parent::get_parent_node();
	if(parentnode.type==parent::COMPUTED) _predictor.learn(feature,parentnode.value-value);
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
void Search_Branch_And_Bound_Partial_Prediction<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::pre_evaluate_availables()
{
	assert(parent::get_criterion());
	assert(parent::get_engine());
	const PSubset &currentset=parent::get_currentset(); assert(currentset);
	Node &parentnode=parent::get_parent_node(); assert(parentnode.type==parent::COMPUTED);

	// pass 1 (master thread): decide predict-vs-compute for every available feature. Deciding
	// for all of them upfront gives the same decisions as deciding one at a time while
	// evaluating, because predictor state is per-feature (the PREDICTOR contract) and each
	// feature is decided once - so the outcome does not depend on the evaluation engine
	PNode avail;
	RETURNTYPE val;
	DIMTYPE to_compute=0;
	for(bool got=this->getFirstAvailable(avail);got;got=this->getNextAvailable(avail))
	{
		if(_predictor.predict(avail->feature,val)) { //value could be predicted
			avail->value=parentnode.value-val;
			avail->type=parent::PREDICTED;
		} else { //value could not be predicted
			avail->type=parent::UNKNOWN; // computed by the engine batch below
			++to_compute;
		}
	}
	if(to_compute==0) return;

	// pass 2: batch-evaluate the non-predictable features through the engine; the per-result
	// callback fires in add-order under both engine implementations, so node values are
	// assigned and prediction contrasts learned in the same order either way
	typename parent::EVALUATORENGINE::ResultCallback on_result=[this,&parentnode](const RETURNTYPE value, const PSubset&, const DIMTYPE feat) {
		PNode node;
		if(this->getAvailable(feat,node)) {
			node->value=value;
			node->type=parent::COMPUTED;
			// conditionally update prediction info
			if(parentnode.type==parent::COMPUTED) _predictor.learn(feat,parentnode.value-value);
		}
	};
	parent::get_engine()->initialize(parent::get_criterion(),parent::get_n(),on_result);
	for(bool got=this->getFirstAvailable(avail);got;got=this->getNextAvailable(avail))
	{
		if(avail->type!=parent::PREDICTED) {
			assert(currentset->selected_raw(avail->feature));
			currentset->deselect_raw(avail->feature);
			if(!parent::get_engine()->add_candidate(currentset,avail->feature)) throw FST::fst_error("Criterion evaluation failure.");
			currentset->select_raw(avail->feature);
		}
	}
	// collect() is where a worker's exception is rethrown; a false return means some
	// evaluate() failed and no callbacks fired at all, which would leave nodes at UNKNOWN and
	// poison the level sort-in - so it must be an error, not a silent continuation
	if(!parent::get_engine()->collect()) throw FST::fst_error("Criterion evaluation failure.");
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
void Search_Branch_And_Bound_Partial_Prediction<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::post_process_tree_level()
{
	assert(parent::get_criterion());
	assert(parent::get_engine());
	const PSubset &currentset=parent::get_currentset();	assert(currentset);
	Node &parentnode=parent::get_parent_node();	assert(parentnode.type==parent::PREDICTED || parentnode.type==parent::COMPUTED);

	// supply the true criterion value of the level's chosen PREDICTED nodes (partial
	// prediction: predictions only order the level, the tree path is exact) - batched
	// through the engine. Results map back to their nodes by submission order: the
	// per-result callback fires in add-order under both engine implementations, and the
	// submitted vector is fully built before the first add_candidate, which matters because
	// the sequential engine fires callbacks synchronously inside add_candidate. The list
	// node pointers stay valid throughout (nothing is inserted into or erased from the level
	// during the batch), and prediction contrasts are learned in the same order either way
	PNode nod;
	std::vector<PNode> submitted;
	for(bool got=this->getFirstNode(nod);got;got=this->getNextNode(nod))
		if(nod->type!=parent::COMPUTED) submitted.push_back(nod);
	if(submitted.empty()) return;

	DIMTYPE done=0;
	typename parent::EVALUATORENGINE::ResultCallback on_result=[this,&parentnode,&submitted,&done](const RETURNTYPE value, const PSubset&, const DIMTYPE feat) {
		PNode node=submitted[done++];
		assert(node->feature==feat);
		node->value=value;
		node->type=parent::COMPUTED;
		// conditionally update prediction info
		if(parentnode.type==parent::COMPUTED) _predictor.learn(feat,parentnode.value-value);
	};
	parent::get_engine()->initialize(parent::get_criterion(),(DIMTYPE)submitted.size(),on_result);
	for(PNode snod : submitted)
	{
		assert(currentset->selected_raw(snod->feature));
		currentset->deselect_raw(snod->feature);
		if(!parent::get_engine()->add_candidate(currentset,snod->feature)) throw FST::fst_error("Criterion evaluation failure.");
		currentset->select_raw(snod->feature);
	}
	// a failed batch must not be allowed to leave nodes without a true value on the tree path
	if(!parent::get_engine()->collect()) throw FST::fst_error("Criterion evaluation failure.");
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
bool Search_Branch_And_Bound_Partial_Prediction<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::cut_possible()
{
	if(parent::is_bound_valid()) {
		Node &currentnode=parent::get_current_node(); assert(currentnode.type==parent::COMPUTED);
		if(currentnode.value<=parent::get_bound_value()) { // BBPP ensures that the value must have been COMPUTED by now
			return true;
		}
	}
	return false;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDPARTIALPREDICTION_H ///:~
