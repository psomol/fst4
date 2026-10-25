#ifndef FSTSEARCHBRANCHANDBOUNDFAST_H
#define FSTSEARCHBRANCHANDBOUNDFAST_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_branch_and_bound_fast.hpp
   \brief   Implements Fast Branch and Bound, i.e., B&B with full utilization of prediction mechanism
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

/*! \brief Implements Fast Branch and Bound, i.e., B&B with full utilization of prediction mechanism

	FBB is in most feature selection tasks the fastest of all Branch & Bound algorithms
	and as such should be the method of first choice whenever optimal feature
	selection is required and possible (see the warning below). Nevertheless,
	the FBB's prediction mechanism can theoretically fail and slow the search
	down (an analogy is perhaps the Quick Sort which is known as the best sorting 
	algorithm for the general case but no guarantee is given about its actual speed).
	If you prefer more conservative option, try BBPP or the even slower but more
	predictable IBB or BBB.
	
	\note All Branch & Bound algorithms by definition yield the solution with the 
	same maximum criterion value, therefore the main concern regarding particular
	Branch & Bound algorithm is only its search speed.

	\warning All Branch & Bound feature selection algorithms require the used
	CRITERION to be monotonic with respect to cardinality. More precisely, it must
	hold that removing a feature from a set MUST NOT increase criterion value.
	Otherwise there is no guarantee as of the optimality of obtained results
	with respect to the used criterion.

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. 
	
	\note Result tracking in case of Branch & Bound algorithms records only results
	of target cardinality.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
class Search_Branch_And_Bound_Fast : public Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef typename parent::PCriterion PCriterion; //!< shared pointer to the criterion the search optimizes
	typedef typename parent::PSubset PSubset; //!< shared pointer to a feature subset
	typedef typename parent::PNode PNode; //!< pointer to a search-tree Node, see Search_Branch_And_Bound
	typedef typename parent::Node Node; //!< search-tree node record (feature, value, type, valid), see Search_Branch_And_Bound
	typedef typename parent::NodeType NodeType; //!< UNKNOWN/RANDOM/PREDICTED/COMPUTED node-value provenance, see Search_Branch_And_Bound
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< shared pointer to the injected evaluation-strategy engine
	explicit Search_Branch_And_Bound_Fast(PEvaluatorEngine engine=PEvaluatorEngine()):Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(engine) {notify("Search_Branch_And_Bound_Fast constructor.");} //!< \p engine forwarded to Search_Branch_And_Bound; defaults to sequential in-line evaluation if not supplied
	virtual ~Search_Branch_And_Bound_Fast() {notify("Search_Branch_And_Bound_Fast destructor.");}

	// for search() implementation see parent class

	void set_gamma(const RETURNTYPE gamma=1.0) {_predictor.set_gamma(gamma);} //!< forwards to the PREDICTOR's set_gamma() (in Branch_And_Bound_Predictor_Averaging the scaling factor applied to a prediction); the PREDICTOR type must provide that method
	void set_delta(const unsigned int delta=1) {_predictor.set_delta(delta);} //!< forwards to the PREDICTOR's set_delta() (in Branch_And_Bound_Predictor_Averaging the number of samples a feature needs before it is predicted at all); the PREDICTOR type must provide that method

	//! writes the method's name, the predictor's own print() and, with a threaded engine, the number of worker threads
	virtual std::ostream& print(std::ostream& os) const override {
		if(parent::get_engine()->is_threaded()) os << "Fast Branch & Bound (FBB)  [Search_Branch_And_Bound_Fast(threads="<<parent::get_engine()->get_max_threads()<<") with " << _predictor << "]";
		else os << "Fast Branch & Bound (FBB)  [Search_Branch_And_Bound_Fast() with " << _predictor << "]";
		return os;
	};
protected:
	PREDICTOR _predictor; //!< the prediction mechanism instance (PREDICTOR type parameter, e.g. Branch_And_Bound_Predictor_Averaging); learns per-feature criterion-value drops from computed leaf/cut evaluations and predicts them for unevaluated nodes, letting FBB skip most criterion evaluations while still guaranteeing the optimal result

	// the following may be overriden in descendant Branch and Bound specialization classes
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit) override {_predictor.initialize(n);} //!< resets the predictor's per-feature statistics before each search - nothing is carried over from a previous one
	virtual void leaf_result(const DIMTYPE feature, const RETURNTYPE value) override; //!< feeds one more observed criterion-value drop to the predictor; fired per leaf, in candidate order, by the base class
	virtual void pre_evaluate_availables() override; //!< orders the available features for the next tree level: each feature's value is predicted where the predictor has enough samples for it, and computed by the criterion otherwise
	virtual void post_process_tree_level() override {} //!< nothing to do: a PREDICTED node value is what makes FBB fast, and the values are turned into computed ones only where a cut decision needs them (see cut_possible())
	virtual bool cut_possible() override; //!< decides whether the sub-branch below the current node can be cut: a value above the bound rejects the cut right away (a predicted value is enough for that), while an actual cut is always decided on a computed value, which is evaluated here if missing
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
void Search_Branch_And_Bound_Fast<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::leaf_result(const DIMTYPE feature, const RETURNTYPE value)
{
	// update the prediction info - but only from a parent whose value was actually
	// computed: the drop learned here is parent value - leaf value, and against a
	// merely PREDICTED parent value the predictor would be learning from its own
	// guesses. The base class fires this after each leaf's bound update, in leaf order
	Node &parentnode=parent::get_parent_node();
	if(parentnode.type==parent::COMPUTED) _predictor.learn(feature,parentnode.value-value);
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
void Search_Branch_And_Bound_Fast<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::pre_evaluate_availables()
{
	assert(parent::get_criterion());
	assert(parent::get_engine());
	const PSubset &currentset=parent::get_currentset(); assert(currentset);
	// no assert on the parent node's type here: FBB legitimately descends into
	// PREDICTED nodes whenever cut_possible() declines to cut, so the parent node
	// need not be COMPUTED
	Node &parentnode=parent::get_parent_node();

	// pass 1 (master thread): decide predict-vs-compute for every available feature.
	// Deciding the whole level upfront gives the same decisions as deciding one
	// feature at a time would, because predictor state is per-feature (the PREDICTOR
	// contract): predict(f) reads only feature f's own statistics, each available is
	// visited exactly once per level, and learn(f) fires only at f's own turn - so no
	// decision can observe another feature's this-level learn
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

	// pass 2: batch-evaluate the non-predictable features through the engine; the
	// per-result callback fires in add-order under both engine implementations, so
	// values are assigned and prediction contrasts learned in one fixed order -
	// master-thread decisions plus in-order result replay make a threaded run
	// reproduce the sequential one exactly
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
	// a criterion exception thrown on a worker is rethrown here; a failed evaluation
	// marks the batch failed with no callbacks fired - the nodes would stay UNKNOWN
	// and poison the level sort-in, so the search must fail instead (same rationale
	// as in the IBB body)
	if(!parent::get_engine()->collect()) throw FST::fst_error("Criterion evaluation failure.");
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class PREDICTOR>
bool Search_Branch_And_Bound_Fast<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,PREDICTOR>::cut_possible()
{	// NOTE: the lazy compute-before-cut below deliberately stays on the master thread -
	// it is a single evaluation on the traversal's critical path (nothing to batch)
	if(parent::is_bound_valid()) {
		Node &currentnode=parent::get_current_node();
		// in FBB the PREDICTED value is enough to evaluate cut as not possible
		if(currentnode.value>parent::get_bound_value()) return false;
		// but decision to cut must be always based on COMPUTED value...
		if(currentnode.type!=parent::COMPUTED) {
			const PSubset &currentset=parent::get_currentset();	assert(currentset);
			assert(parent::get_criterion());
			assert(currentnode.feature<parent::get_n());
			assert(currentset->selected_raw(currentnode.feature));
			currentset->deselect_raw(currentnode.feature);
			if(!parent::get_criterion()->evaluate(currentnode.value,currentset)) throw FST::fst_error("Criterion evaluation failure."); 
			currentset->select_raw(currentnode.feature);
			currentnode.type=parent::COMPUTED;
			// conditionally update prediction info
			// (no assert on the parent's type - the if below already handles the legitimate PREDICTED-parent case)
			Node &parentnode=parent::get_parent_node();
			if(parentnode.type==parent::COMPUTED) _predictor.learn(currentnode.feature,parentnode.value-currentnode.value);
		}
		if(currentnode.value<=parent::get_bound_value()) {
			return true;
		}
	}
	return false;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDFAST_H ///:~
