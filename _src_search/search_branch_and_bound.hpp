#ifndef FSTSEARCHBRANCHANDBOUND_H
#define FSTSEARCHBRANCHANDBOUND_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_branch_and_bound.hpp
   \brief   Implements Branch and Bound template method as basis for more advanced B&B implementations
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search.hpp"
#include "result_tracker.hpp"
#include "candidate_evaluator.hpp"
#include "candidate_evaluator_sequential.hpp"

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

/*! \brief Implements Branch and Bound template method as basis for more advanced B&B implementations

	This is the abstract basis of concrete Branch & Bound implementations.

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

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Branch_And_Bound : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class: the abstract Search interface (search(), progress reporting, result tracker)
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion under optimization
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the injected candidate-evaluation strategy type (sequential-inline or thread-pooled); process_leafs(), and descendants' pre_evaluate_availables()/post_process_tree_level() overrides, batch candidate evaluations through it
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to an EVALUATORENGINE, as accepted by the constructor and returned by get_engine()
	/*! \brief Constructs the search, optionally injecting a candidate-evaluation engine.
	    \param engine the evaluation strategy candidate batches are routed through
	    (e.g. Candidate_Evaluator_Sequential or a thread-pooled implementation); a
	    default-constructed (null) shared_ptr installs a Candidate_Evaluator_Sequential,
	    so plain sequential B&B needs no explicit engine argument. */
	explicit Search_Branch_And_Bound(PEvaluatorEngine engine=PEvaluatorEngine()):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine) {
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Search_Branch_And_Bound constructor.");
	}
	virtual ~Search_Branch_And_Bound() {notify("Search_Branch_And_Bound destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override; //!< returns the found subset of exactly target_d features + its criterion value; target_d must be nonzero (Branch & Bound searches one fixed cardinality, it does not optimize it)

	virtual std::ostream& print(std::ostream& os) const override {os << "Search_Branch_And_Bound() abstract class"; return os;};

protected:
	//! Marks how a Node's value was obtained: UNKNOWN = not yet assigned; RANDOM = reserved but currently unused by any B&B variant in this file; PREDICTED = filled from a PREDICTOR estimate pending confirmation (FBB/BBPP node ordering); COMPUTED = an exact criterion value
	typedef enum {UNKNOWN, RANDOM, PREDICTED, COMPUTED} NodeType;
	//! Structure representing search tree node
	typedef struct {
		DIMTYPE    feature;  //!< the feature index this node represents removing from its ancestor set (meaningless in the single root node)
		RETURNTYPE value;    //!< the node's criterion value: exact once type==COMPUTED, an ordering estimate while type==PREDICTED, not meaningful otherwise
		NodeType   type;     //!< how value was obtained, see NodeType
		bool       valid;    //!< for use in _available only: true while the feature is still an undecided candidate (see available_add()/available_remove()); unused for nodes stored in _tree
	} Node;
	typedef Node* PNode; //!< pointer to a Node, as used by the getFirst.../getNext... accessors below
	typedef std::list<Node> ListNode; //!< one tree level's not-yet-decided candidate nodes, kept sorted ascending by value; the main search loop visits/pops from the back (highest value = most promising) via get_current_node()
	typedef std::vector<ListNode> VectorListNode; //!< the search tree: one ListNode per level, indexed by tree depth k (see _tree)
	typedef typename ListNode::iterator ListNodeIter; //!< iterator into a single tree level's ListNode, as used by _nodeiter/getFirstNode()/getNextNode()
	typedef typename std::vector<ListNode>::iterator TreeLevelIter; //!< iterator over tree levels (used only to clear _tree in setup_default_structures())
	typedef std::vector<Node> VectorNode; //!< type of _available: one Node per original feature, indexed by feature id

private:
	// the following structures need to be valid throughout the course of one search
	PCriterion _criterion; //!< the criterion supplied to search(), assigned once initialize() has run (see search())

	PEvaluatorEngine _engine; //!< injected evaluation strategy (sequential-inline or thread-pooled)

	VectorListNode _tree; //!< the search tree; _tree[k] holds the not-yet-decided candidate nodes at depth k (see make_tree_level())

	ListNodeIter   _nodeiter; //!< used in get*Node()
	DIMTYPE        _availiter; //!< iteration cursor used by getFirstAvailable()/getNextAvailable() over _available

	VectorNode     _available; //!< the pool of still-undecided features, one entry per original feature and indexed by feature id: Node::feature repeats that id, and Node::valid is the in-pool indicator (true = still undecided, see available_add()/available_remove())
	DIMTYPE         get_available_size() {DIMTYPE c=0; for(DIMTYPE i=0;i<_available.size();i++) if(_available[i].valid) c++; return c;} //!< number of features still undecided (valid==true) in _available
	void            available_add(const DIMTYPE feature) {assert(_available.size()>feature && !_available[feature].valid); _available[feature].valid=true; _available[feature].type=UNKNOWN;} //!< re-admits feature to the undecided candidate pool (valid=true, type reset to UNKNOWN); used on backtrack and when a cut releases it
	void            available_remove(const DIMTYPE feature) {assert(_available.size()>feature && _available[feature].valid); _available[feature].valid=false;} //!< withdraws feature from the undecided candidate pool (valid=false), e.g. once it has been sorted into a tree level

	PSubset        _currentset; //!< the subset defined by the branch currently being followed: all n features minus the ones removed on the path from the root down to level k
	void            currentset_add(const DIMTYPE feature) {assert(!_currentset->selected_raw(feature)); _currentset->select_raw(feature);} //!< re-selects feature in _currentset, undoing a prior currentset_remove() (used when backtracking up the tree)
	void            currentset_remove(const DIMTYPE feature) {assert(_currentset->selected_raw(feature)); _currentset->deselect_raw(feature);} //!< deselects feature from _currentset, committing its removal as the branch descends past it

	PSubset        _tempset; //!< temporary copy of _currentset subset to be modified and evaluated

	PSubset        _boundset; //!< the best subset of target cardinality d found so far; meaningful only once _boundvalid is true
	RETURNTYPE     _boundvalue; //!< the criterion value of _boundset; meaningful only once _boundvalid is true
	bool           _boundvalid; //!< true once at least one candidate of target cardinality d has been evaluated, so _boundset/_boundvalue hold a valid bound
	
	DIMTYPE         k; //!< current tree level
	DIMTYPE         n; //!< no. of all features
	DIMTYPE         d; //!< no. of features to be selected
	
protected:
	PCriterion const &get_criterion() const {assert(_criterion); return _criterion;} //!< accessor for _criterion, for use by descendant B&B variants' hooks
	PEvaluatorEngine const &get_engine() const {assert(_engine); return _engine;} //!< accessor for the injected evaluation strategy (sequential-inline or thread-pooled), for use by descendant B&B variants' hooks

	RETURNTYPE get_bound_value() const {return _boundvalue;} //!< the current best (bound) criterion value; meaningful only once is_bound_valid()
	bool       is_bound_valid() const {return _boundvalid;} //!< true once a valid bound has been established, see _boundvalid

	int     get_k() const {return k;} //!< the tree depth/level currently being processed (root is 0; the deepest stored level is n-d-1, and level n-d is the leaf level, evaluated but never stored). The return type is int, while the member k is DIMTYPE
	DIMTYPE get_n() const {return n;} //!< total number of features, see n
	DIMTYPE get_d() const {return d;} //!< target subset cardinality, see d

	bool getFirstNode(PNode &nod) {assert(k<n-d+1 && _tree.size()==n-d); if(_tree[k].empty()) return false; else {_nodeiter=_tree[k].begin(); nod=&(*_nodeiter); return true;}} //!< begins forward iteration (ascending by value, from _tree[k].begin()) over the current level's not-yet-decided nodes into nod; returns false if the level is empty
	bool getNextNode(PNode &nod) {assert(k<n-d+1 && _tree.size()==n-d); if(++_nodeiter==_tree[k].end()) return false; else {nod=&(*_nodeiter); return true;}} //!< advances the getFirstNode() iteration; returns false once the level's end is reached

	bool getFirstAvailable(PNode &avail) {assert(_available.size()>0); for(_availiter=0; _availiter<_available.size() && !_available[_availiter].valid; _availiter++) {} if(_availiter<_available.size()) {avail=&(_available[_availiter]); return true;} else return false;} //!< begins iteration over _available in feature-index order, skipping withdrawn (invalid) entries, into avail; returns false if none are available
	bool getNextAvailable(PNode &avail) {assert(_available.size()>0 && _availiter<_available.size()); _availiter++; while(_availiter<_available.size() && !_available[_availiter].valid) {_availiter++;} if(_availiter<_available.size()) {avail=&(_available[_availiter]); return true;} else return false;} //!< advances the getFirstAvailable() iteration, again skipping invalid entries; returns false once exhausted
	bool getAvailable(DIMTYPE idx, PNode &avail) {assert(idx<_available.size()); if(idx<_available.size()) {avail=&(_available[idx]); return true;} else return false;} //!< direct lookup of _available[idx] by feature index, regardless of its valid flag; returns false if idx is out of range

protected:
	PSubset& get_currentset(){return _currentset;} //!< mutable accessor for _currentset, for use by descendant B&B variants' hooks
	Node&    get_parent_node(){assert(k>0 && _tree.size()>=k && !_tree[k-1].empty()); return _tree[k-1].back();} //!< the node one level up (_tree[k-1].back()) whose feature removal defines the branch currently being followed; valid at the leaf level too (k==n-d==_tree.size(), the unstored leaf level whose parent is the last stored one), which is where FBB/BBPP's leaf hooks call it
	Node&    get_current_node(){assert(_tree.size()>k && !_tree[k].empty()); return _tree[k].back();} //!< the node at the current level k presently being visited (_tree[k].back()), i.e. about to be tested for cutting or descended into
	
	std::ostream* shared_os; //!< to enable sharing the same stream across the methods below
	
	/*! \brief Allocates and resets everything one search() run needs: the n-d stored tree levels (all
	    emptied), the _available pool (all n features valid), _currentset (all features selected, i.e.
	    the root), the _tempset scratch subset and the _boundset holder (bound marked invalid).
	    \param d target subset cardinality; must satisfy 0 < d < n
	    \param n total number of features */
	void setup_default_structures(const DIMTYPE d, const DIMTYPE n);
	/*! \brief Registers one fully-evaluated candidate subset of target cardinality d against the running bound.
	    Updates _boundset/_boundvalue/_boundvalid whenever val improves on (or is the first) bound
	    and, at NORMAL output or above, reports the improvement as a new solution; unconditionally
	    feeds (val,sub) to the result tracker if one is active. Called from process_rightmost_path()
	    and from process_leafs()'s per-leaf engine callback. */
	void update_bound(const RETURNTYPE val, const PSubset &sub);
	/*! \brief Builds the k-th tree level (_tree[k]), keeping exactly n-d-k+1 candidates in reserve for the levels still to come.
	    Calls pre_evaluate_availables() to obtain each still-available feature's ordering value,
	    sorts the best (levelsize = get_available_size()-(n-d-k+1)) of them ascending by value into
	    _tree[k] (withdrawing the chosen features from _available via available_remove()), then
	    calls post_process_tree_level() so a descendant can substitute exact values for any node
	    whose ordering value was only an estimate. */
	void make_tree_level();
	/*! \brief Evaluates and registers the "rightmost" leaf reachable from the current branch by removing every feature still in _available at once.
	    Copies _currentset into _tempset, deselects every remaining available feature, evaluates the
	    criterion on the result and passes it to update_bound() - this seeds/tightens the bound as
	    soon as each new tree level is built, before further descent, so later cuts can trigger sooner. */
	void process_rightmost_path();

	std::shared_ptr<StopWatch> swatch; //!< stopwatch for the running search, (re)created at the start of search() and reported alongside progress/solution messages
protected:
	// the following may/must be overriden in descendant Branch and Bound specialization classes (to implement specific functionality or threading)
	virtual void process_leafs(); //!< engine-routed leaf evaluation; descendants hook per-leaf work via leaf_result() instead of overriding this
	virtual void leaf_result(const DIMTYPE feature, const RETURNTYPE value) {} //!< fired once per leaf in candidate (enumeration) order from process_leafs' engine callback, after the bound update - FBB/BBPP learn prediction contrasts here
	//! \brief Called once per search() run, after setup_default_structures() and the root evaluation but
	//! before the traversal starts, so descendants can size and reset their own structures. The
	//! arguments repeat the run's parameters; the criterion is the one search() was called with and is
	//! already initialize()d.
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit)=0;
	//! \brief Called by make_tree_level() before the k-th level is built. It must assign Node::value (and
	//! Node::type) to EVERY currently valid entry of _available, since those values are what the level is
	//! sorted by; features left at type UNKNOWN sort to the end of the level. When it is called,
	//! _currentset is the subset of the branch above level k and _tree[k] is still empty.
	virtual void pre_evaluate_availables()=0;
	//! \brief Called by make_tree_level() once the k-th level has been filled and its features withdrawn
	//! from _available. It is the place to replace values that are only estimates (type PREDICTED) by
	//! exact ones (type COMPUTED) where the variant requires it; it may leave the level untouched.
	virtual void post_process_tree_level()=0;
	//! \brief Tests the current node (get_current_node(), i.e. _tree[k].back()) for the possibility to cut
	//! the whole sub-branch below it. It is called only with a non-empty _tree[k], and it MUST return
	//! false while is_bound_valid() is false - there is nothing to compare against yet. A variant whose
	//! node value is not COMPUTED yet is expected to compute it here (on the calling thread: this is a
	//! single evaluation on the traversal's critical path).
	virtual bool cut_possible()=0;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::setup_default_structures(const DIMTYPE d, const DIMTYPE n)
{
	assert(d>0);
	assert(d<n);
	_tree.resize(n-d); // logically n-d+1 levels, but the last (leaf) level is not stored - its nodes are evaluated directly by process_leafs()
	for(TreeLevelIter i=_tree.begin();i!=_tree.end();i++) i->clear();
	_available.resize(n); for(DIMTYPE f=0;f<n;f++) {_available[f].feature=f; _available[f].valid=true;}
	_currentset.reset(new SUBSET(n)); _currentset->select_all_raw();
	_tempset.reset(new SUBSET(n));
	_boundset.reset(new SUBSET(n));	_boundvalid=false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::update_bound(const RETURNTYPE val, const PSubset &sub)
{
	assert(sub);
	assert(_boundset);
	assert(sub->get_n()==_boundset->get_n());
	assert(swatch);
	if(!_boundvalid || val>_boundvalue) {
		_boundvalue=val;
		_boundset->stateless_copy(*sub);
		_boundvalid=true;
		if(parent::output_normal()) this->report_solution(*shared_os, val, *sub, msgbuf() << "NEW BOUND (val="<<val<<"): " << *sub << std::endl << *swatch << std::endl << std::endl << std::flush);
	}
	if(parent::result_tracker_active()) parent::_tracker->add(val,sub);
}	

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::make_tree_level() // make k-th tree level
{
	// based on k and no. of _available features
	assert(k>0);
	assert(k<n-d+1);
	assert(_tree.size()==n-d);
	assert(_tree[k].empty());
	assert(_available.size()==n);
	
	pre_evaluate_availables(); // assign values to candidates to enable sorting into the tree level to be constructed here
	
	const DIMTYPE levelsize=get_available_size()-(n-d-k+1);
	assert(levelsize>0);
	assert(levelsize<=get_available_size());

	// sort-in levelsize features according to value (in case of type UNKNOWN feature put at the end of list)
	PNode avail;
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail))
	{
		DIMTYPE pos=0;
		ListNodeIter it=_tree[k].begin();
		while(it!=_tree[k].end() && it->value<avail->value) {++it; ++pos;}
		if(pos<levelsize) {
			_tree[k].insert(it,*avail);
			if(_tree[k].size()>levelsize) _tree[k].pop_back();
		}
	}
	assert(_tree[k].size()==levelsize);
	for(ListNodeIter it=_tree[k].begin();it!=_tree[k].end();it++) {
		available_remove(it->feature);
	}
	
	post_process_tree_level(); // enable to re-evaluate chosen candidate values (e.g., supply missing true _criterion values, etc.)
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::process_rightmost_path()
{
	assert(_tempset);
	assert(_currentset);
	assert(_available.size()==n);
	assert(get_available_size()>0);
	assert(_criterion);
	_tempset->stateless_copy(*_currentset);
	
	RETURNTYPE val;
	PNode avail;
	
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail)) _tempset->deselect_raw(avail->feature);
	if(!_criterion->evaluate(val,_tempset)) throw FST::fst_error("Criterion evaluation failure."); 
	update_bound(val,_tempset); //adds to tracker
	if(parent::output_detailed()) this->report(*shared_os, msgbuf() << "rightmost leaf (val="<<val<<"): " << *_tempset << std::endl << std::endl << std::flush);
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::process_leafs()
{
	assert(_currentset);
	assert(get_criterion());
	assert(_engine);
	PNode avail;
	// bound updates and progress output fire once per leaf in enumeration order: the
	// sequential engine fires them interleaved with the evaluations, the threaded engine
	// once the batch has completed - either way the order, and hence the result, is the same
	typename EVALUATORENGINE::ResultCallback on_result=[this](const RETURNTYPE value, const PSubset &csub, const DIMTYPE feat) {
		update_bound(value,csub); //adds to tracker
		if(parent::output_detailed()) this->report(*shared_os, msgbuf() << "leaf (val="<<value<<"): " << *csub << std::endl << std::endl << std::flush);
		leaf_result(feat,value); // descendant hook (FBB/BBPP prediction learning) - fires in candidate order under both engine implementations
	};
	_engine->initialize(get_criterion(), get_n(), on_result);
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail))
	{
		currentset_remove(avail->feature);
		if(!_engine->add_candidate(_currentset,avail->feature)) throw FST::fst_error("Criterion evaluation failure.");
		currentset_add(avail->feature);
	}
	_engine->collect(); // (return value carries no error here: sequential evaluation failures throw at add_candidate above, threaded ones throw in the worker)
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + _criterion value
{
	swatch.reset(new StopWatch);
	notify("Search_Branch_And_Bound::search().");
	assert(sub);
	assert(crit);
	n=sub->get_n();
	assert(target_d>0 && target_d<n);
	d=target_d;
	assert(d>0 && d<=n);
	shared_os=&os;

	if(parent::result_tracker_active()) parent::_tracker->set_output_detail(parent::get_output_detail());
		
	if(parent::output_normal()) {
		msgbuf sos;
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		this->report(os,sos);
	}
	// user-supplied criteria default to is_monotonic()==false and get this warning
	// automatically; the criteria shipped with FST4 that are monotonic declare the
	// property, so they stay silent
	if(parent::output_normal() && !crit->is_monotonic()) {
		this->report(os, msgbuf() << "WARNING: the supplied criterion does not declare monotonicity w.r.t. subset cardinality" << std::endl
		   << "(Criterion::is_monotonic()==false). Branch & Bound optimality guarantees do NOT apply -" << std::endl
		   << "the result may be suboptimal. Override is_monotonic() to true only if removing a feature" << std::endl
		   << "provably never increases the criterion value; otherwise prefer a non-optimal search." << std::endl << std::endl << std::flush);
	}

	if(n==d) {
		sub->select_all_raw();
		if(!crit->evaluate(result,sub)) throw FST::fst_error("Criterion evaluation failure."); 
		return true;
	}
	
	setup_default_structures(d,n);
	
	// initialize root
	Node root; // Node::feature and Node::valid are not used in root
	_tempset->select_all_raw(); 
	if(!crit->evaluate(root.value,_tempset)) throw FST::fst_error("Criterion evaluation failure."); 
	root.type=COMPUTED;
	_tree[0].push_back(root);
		
	initialize(d,n,crit); // to enable overloaded functionality
	_criterion=crit;
		
	// Deepest level at which a CUT message was last printed. Cuts near the leaves are both
	// frequent and uninteresting, so a CUT is reported only when it happens at that level or
	// higher up the tree; this throttles the message stream without affecting the search.
	DIMTYPE lastcut_k=n-d;

	DIMTYPE feature;
	// TRAVERSAL INVARIANT, maintained by every block below. At any point between iterations:
	//  - the features removed on the path from the root to level k are exactly the features
	//    deselected in _currentset, one per level 1..k-1, namely _tree[j].back().feature;
	//  - a feature is in exactly one of three places: selected in _currentset, sorted into some
	//    level of _tree, or valid in _available. Descending one level moves _tree[k].back().feature
	//    out of _currentset; backtracking one level moves it back into _currentset AND back into
	//    _available, and pops it off the level, so the level's next node becomes the current one.
	//  - a level that becomes empty means its subtree is fully explored, hence the "pop and go up"
	//    loops keep climbing while _tree[k] is empty. Reaching k==0 ends the search.
	k=1; // start with the first level below root
	do{
		// traverse tree branch down
		while(k>0 && k<n-d) {

			make_tree_level(); // make k-th level, choose |_available|-(n-d-k+1) nodes from _available and use them to make next level
			process_rightmost_path();
			
			// recursively cut all sub-branches that can be cut
			while(k>0 && cut_possible())
			{
				assert(!_tree[k].empty());
				if(k<=lastcut_k) { // report only cuts at or above the last reported level (see lastcut_k)
					if(parent::output_normal()) this->report(os, msgbuf() << "CUT (k="<<k<<", val="<<_tree[k].back().value<<", bound="<<get_bound_value()<<", feature=" << _tree[k].back().feature<<")  " << std::endl << std::endl);
					lastcut_k=k;
				}
				feature=_tree[k].back().feature; assert(feature<n);
				available_add(feature);
				_tree[k].pop_back();
				// the cut emptied this level: climb until a level with an untried node is found,
				// restoring each passed level's feature into _currentset and _available on the way
				while(k>0 && _tree[k].empty()) {
					--k;
					if(k>0) {
						assert(!_tree[k].empty());
						feature=_tree[k].back().feature; assert(feature<n);
						currentset_add(feature);
						available_add(feature);
						_tree[k].pop_back();
					}
				} 
			}
			// no more cutting possible, proceed down to next level (or finish if k==0)
			if(k>0) {
				feature=_tree[k].back().feature; assert(feature<n);
				currentset_remove(feature);
				++k;
			}
		}
		// evaluate all leafs that are descendants of the just visited node
		if(k==n-d) {
			process_leafs(); //depends on _currentset, _available, not on k
			// the leaf level has no stored nodes, so there is nothing left to try here:
			// backtrack up to the closest ancestor that still has an untried node, undoing
			// each level's feature removal as we climb
			do {
				--k;
				if(k>0) {
					assert(!_tree[k].empty());
					feature=_tree[k].back().feature; assert(feature<n);
					currentset_add(feature);
					available_add(feature);
					_tree[k].pop_back();
				}
			} while(k>0 && _tree[k].empty());
		}
		// proceed down through the next rightmost subbranch to the left from the last one
		if(k>0) {
			assert(!_tree[k].empty());
			feature=_tree[k].back().feature; assert(feature<n);
			currentset_remove(feature);
			++k;
		}
		
	} while(k>0);

	assert(_boundset);
	assert(sub->get_n()==_boundset->get_n());
	sub->stateless_copy(*_boundset);
	result=_boundvalue;
	if(parent::output_normal()) {
		msgbuf sos; 
		sos << "......................................." << std::endl;
		sos << "Branch and Bound search finished. " << *swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	swatch.reset();
	return true;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUND_H ///:~
