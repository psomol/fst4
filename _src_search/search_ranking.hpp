#ifndef FSTSEARCHRANKING_H
#define FSTSEARCHRANKING_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_ranking.hpp
   \brief   Defines interface for individual feature ranking method implementations
   \author  Stanislav Mikes (xaos@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    ? 2012 (last updated July 2026)
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
#ifdef FST_THREADED
#include <mutex>
#endif
#include <iostream>
#include <sstream>
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search.hpp"
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

/*! \brief Abstract base of the feature-ranking methods: scores features one by one, then turns the ranking into a feature subset.

	A concrete ranking method supplies evaluate_individuals() only, which fills the
	protected \c bifs list with one (feature, criterion value) record per feature,
	ordered by descending value; the order of features of equal value is decided by
	that method. Everything else - the two search() modes, the feature-list
	bookkeeping and the iteration over the ranking - lives here.

	search() has two modes:
	\li target_d>0: the target_d highest-ranked features form the result, which is
	    evaluated once by the criterion; a result tracker receives that single record.
	\li target_d==0: every prefix of the ranking is evaluated - the best feature
	    alone, the best two features, and so on - and the best-valued prefix is
	    returned; equal values leave the shorter prefix in place. Here every
	    evaluated prefix is reported to the result tracker.

	target_d must be smaller than the full feature count. Note that a ranking of
	individually evaluated features ignores feature interactions by construction;
	its subsets are generally weaker than those found by the subset search methods,
	which is the price for the linear number of criterion evaluations. */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Ranking : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< the abstract Search interface this class implements
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATORENGINE; //!< the injected candidate-evaluation engine interface driving individual-feature evaluation
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to an EVALUATORENGINE, as passed to the constructor
	//! \param engine evaluation-strategy engine used to evaluate candidate features; defaults to a fresh Candidate_Evaluator_Sequential (inject Candidate_Evaluator_Threaded for a thread-pooled search() dimensionality-optimization phase)
	explicit Search_Ranking(PEvaluatorEngine engine=PEvaluatorEngine()):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine) {
		if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>());
		notify("Search_Ranking constructor.");
	}
	virtual ~Search_Ranking() {notify("Search_Ranking destructor.");}

        // creates FEATURELIST, must be called before get*Feature() and search()
	virtual bool evaluate_individuals(const DIMTYPE n, const PCriterion crit, std::ostream& os=std::cout) = 0; //!< ranks all \p n features individually by \p crit, filling \c bifs descending by criterion value (the order within a group of equal values is the concrete method's own convention); called by search() itself, and required before getFirstFeature()/getNextFeature() are used on their own -- implemented by the concrete ranking method (e.g. Search_Ranking_BIF)

        // NOTE: crit is expected to be already initialize()d before use here
	//! ranks the features (evaluate_individuals()) and returns the target_d best-ranked ones, or - for target_d==0 - the best-valued prefix of the ranking; false if the ranking or one of the criterion evaluations failed. \p target_d must be smaller than the full feature count
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout) override;

	bool getFirstFeature(RETURNTYPE &value, DIMTYPE &feature) const; //!< iterator over \c bifs (the ranking built by evaluate_individuals()), highest-ranked feature first; returns false if \c bifs is empty
	bool getNextFeature(RETURNTYPE &value, DIMTYPE &feature) const; //!< continues the getFirstFeature() iteration; returns false once the ranking is exhausted. There is one cursor per object: it is protected against concurrent access in threaded builds, but two interleaved iterations of the same object still consume one another's positions

  virtual std::ostream& print(std::ostream& os) const override {os << "Search_Ranking() abstract class"; return os;};
protected:
	PEvaluatorEngine _engine; //!< injected evaluation strategy (sequential-inline or thread-pooled)
	//! Structure to hold [feature,criterion value] pair while ranking features
	typedef struct {
		DIMTYPE feature;    //!< the feature's index
		RETURNTYPE critval; //!< the feature's individual criterion value, as evaluated by evaluate_individuals()
	} OneFeature;
	typedef std::list<OneFeature> FEATURELIST; //!< the individual-feature ranking built by evaluate_individuals(), sorted descending by critval
	FEATURELIST bifs; //!< the individual-feature ranking; empty until evaluate_individuals() is called
private:
	mutable typename FEATURELIST::const_iterator get_iter; //!< iteration cursor for getFirstFeature()/getNextFeature() (single shared cursor; see get_iter_mutex under FST_THREADED)
#ifdef FST_THREADED
	mutable std::mutex get_iter_mutex; //!< protects get_iter for thread-safe iteration
#endif
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::getFirstFeature(RETURNTYPE &value, DIMTYPE &feature) const
{
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(get_iter_mutex);
#endif
	get_iter = bifs.begin();
	if(get_iter!=bifs.end()) {
		value=(*get_iter).critval;
		feature=(*get_iter).feature;
		get_iter++;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::getNextFeature(RETURNTYPE &value, DIMTYPE &feature) const
{
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(get_iter_mutex);
#endif
	if(get_iter!=bifs.end()) {
		value=(*get_iter).critval;
		feature=(*get_iter).feature;
		get_iter++;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	notify("Search_Ranking::search().");
	bool track=parent::result_tracker_active(); if(track) parent::_tracker->set_output_detail(parent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d<sub->get_n());
	DIMTYPE _n=sub->get_n();

	if(parent::output_normal()) {
		msgbuf sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		this->report(os,sos);
	}

	if(!evaluate_individuals(_n,crit,os)) return false;

	std::shared_ptr<SUBSET> tempsub(new SUBSET(_n));
	typename FEATURELIST::const_iterator iter;
	DIMTYPE i;
	if(target_d>0) { // return subset of size target_d
		sub->deselect_all();
		for(iter=bifs.begin(),i=0; iter!=bifs.end() && i<target_d; iter++,i++) sub->select((*iter).feature);
		if(!crit->evaluate(result,sub)) return false;
		if(track) parent::_tracker->add(result,sub);
	}
	else { // find best dimensionality
		tempsub->deselect_all();
		sub->deselect_all();
		assert(bifs.begin()!=bifs.end());
		assert(_engine);

		// per-candidate work: track, keep best, report progress - fired in
		// candidate (i.e., growing-subset-size) order by either engine; the
		// sequential one fires it interleaved with the evaluations themselves
		bool first=true;
		RETURNTYPE best=RETURNTYPE();
		typename EVALUATORENGINE::ResultCallback on_result=[&](const RETURNTYPE cval, const PSubset &csub, const DIMTYPE) {
			if(track) parent::_tracker->add(cval,csub);
			if(first || cval>best) {
				best=cval;
				sub->stateless_copy(*csub);
				first=false;
				if(parent::output_normal()) this->report_solution(os, cval, *csub, msgbuf() << "new best result="<<cval<<", " << *csub << std::endl << std::endl << std::flush);
			} else {
				if(parent::output_normal()) this->report_solution(os, cval, *csub, msgbuf() << "result="<<cval<<", " << *csub << std::endl << std::endl << std::flush);
			}
		};
		_engine->initialize(crit, _n, on_result);
		for(iter=bifs.begin(); iter!=bifs.end(); iter++) {
			tempsub->select((*iter).feature);
			if(!_engine->add_candidate(tempsub)) return false;
		}
		if(!_engine->collect()) return false;
		result=best;
	}
	if(parent::output_normal()) {
		msgbuf sos; 
		sos << "......................................." << std::endl;
		sos << *this << " search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}


} // namespace
#endif // FSTSEARCHRANKING_H ///:~
