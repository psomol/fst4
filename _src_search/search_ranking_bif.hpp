#ifndef FSTSEARCHRANKINGBIF_H
#define FSTSEARCHRANKINGBIF_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search_ranking_bif.hpp
   \brief   Implements Best Individual Features, i.e., individual feature ranking
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search_ranking.hpp"

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

/*! \brief Implements Best Individual Features, i.e., individual feature ranking

    Each feature is evaluated alone, as a subset of size one, and the features are
    ordered by descending criterion value; the inherited Search_Ranking::search()
    then turns that ranking into the resulting subset.

    \note This single class serves both the sequential and the threaded execution
    model: the evaluation strategy (a Candidate_Evaluator implementation) is injected
    through the constructor and defaults to sequential in-line evaluation. The
    best-dimensionality scan in the inherited Search_Ranking::search() runs through
    the same engine. The ranking does not depend on the engine - the same features
    are ranked in the same order at any thread count. */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Ranking_BIF : public Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class: the individual-ranking search skeleton (final best-dimensionality scan, feature-list bookkeeping)
	typedef std::shared_ptr<CRITERION> PCriterion; //!< shared pointer to the criterion being optimized
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< the candidate-evaluation engine type, inherited from parent
	/*! \brief Constructs the search with the given candidate-evaluation engine; a default-constructed \p engine falls back to sequential in-line evaluation.
	    \param engine the evaluation strategy to use; pass a Candidate_Evaluator_Threaded to evaluate the individual features in a thread pool */
	explicit Search_Ranking_BIF(PEvaluatorEngine engine=PEvaluatorEngine()):Search_Ranking<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(engine) {notify("Search_Ranking_BIF constructor.");}
	virtual ~Search_Ranking_BIF() {notify("Search_Ranking_BIF destructor.");}

	/*! \brief Scores every single feature individually under \p crit and ranks them into parent::bifs (descending), the BIF core step consumed by Search_Ranking::search().
	    \param n number of features to evaluate (features 0..n-1)
	    \param crit the criterion each singleton feature subset is scored with
	    \param os stream progress messages are reported to at NORMAL/DETAILED output detail
	    \return false if any individual evaluation fails (the candidate evaluator's add_candidate()/collect() returned false)
	    \note Features of equal criterion value end up in descending feature-index
	    order, because each newly ranked feature is inserted ahead of the equally
	    valued ones already in the list. The order matters: it decides which of the
	    tied features enter a subset of a given size. */
      	virtual bool evaluate_individuals(const DIMTYPE n, const PCriterion crit, std::ostream& os=std::cout) override;

	virtual std::ostream& print(std::ostream& os) const override {
		os << "Individual feature ranking  [";
		if(parent::_engine->is_threaded()) os << "Search_Ranking_BIF(threads="<<parent::_engine->get_max_threads()<<")";
		else os << "Search_Ranking_BIF()";
		if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
		os<<"]"; return os;
	}
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::evaluate_individuals(const DIMTYPE n, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	notify("Search_Ranking_BIF::evaluate_individuals().");
	assert(crit);
	assert(n>0);
	assert(parent::_engine);
	std::shared_ptr<SUBSET> tempsub(new SUBSET(n)); // tempsub forward_mode assumed
	tempsub->deselect_all();
	parent::bifs.clear();

	// sorted insertion + progress output, fired once per feature in feature
	// order by either engine (the sequential one fires it interleaved with the
	// evaluations themselves); an equally valued feature is inserted in front of
	// those already ranked
	typename parent::EVALUATORENGINE::ResultCallback on_result=[this,&os](const RETURNTYPE val, const PSubset&, const DIMTYPE feat) {
		typename parent::OneFeature ofbuf;
		typename parent::FEATURELIST::iterator iter;
		ofbuf.feature=feat; ofbuf.critval=val;
		for(iter=this->bifs.begin(); iter!=this->bifs.end() && ofbuf.critval<(*iter).critval; iter++) {}
		this->bifs.insert(iter,ofbuf);
		if(parent::output_normal()) this->report(os, msgbuf() << "f:"<<feat<<", critval="<<val<<std::endl<<std::flush);
	};
	parent::_engine->initialize(crit, n, on_result);
	for(DIMTYPE i=0;i<n;i++) { // compute individual values, store in list
		if(i>0) tempsub->deselect(i-1);
		tempsub->select(i);
		if(!parent::_engine->add_candidate(tempsub,i)) return false;
	}
	if(!parent::_engine->collect()) return false;

	if(parent::output_normal()) this->report(os, msgbuf() << std::endl<<std::flush);
	if(parent::output_detailed()) {
		msgbuf sos; sos << "parent::bifs: ";
		for(typename parent::FEATURELIST::iterator iter=this->bifs.begin(); iter!=this->bifs.end(); iter++) sos << (*iter).feature << " ";
		sos << std::endl << std::flush;
		this->report(os,sos);
	}
	return true;
}


} // namespace
#endif // FSTSEARCHRANKINGBIF_H ///:~
