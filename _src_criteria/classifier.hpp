#ifndef FSTCLASSIFIER_H
#define FSTCLASSIFIER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    classifier.hpp
   \brief   Defines interface for classifier implementations
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
#include <vector>
#include <utility>
#include <cassert>
#include "clonable.hpp"
#include "error.hpp"
#include "global.hpp"
#include "wrapper_metrics.hpp"

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

/*! \brief abstract class, defines interface for classifier implementations (mainly to be used in wrappers)

    \note Lifecycle contract. A classifier must be trained before any of the test*()
          methods or classify() is called: train() learns the model restricted to the
          features selected in \c sub, from the TRAIN blocks of \c da. The test*()
          methods then read the TEST blocks of the accessor passed to them, which must
          expose the same features in the same order as the accessor train() was given
          (in practice the same accessor instance, or a sharing clone of it). Passing a
          subset of different content requires a new train() call. Instances are not
          thread-safe; threaded searches use clone() to give each thread its own copy. */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier : public Clonable { // abstract class
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset;            //!< shared pointer to the (fixed, non-reassignable) feature subset being evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier() {}
	virtual ~Classifier() {}

	/*! Classifies one pattern; on success \c cls receives the predicted class index
	    (0-based, in [0,classes)). Returns false when no decision can be made at all -
	    the model has not been trained, or the implementation hit an unrecoverable
	    condition; false is not a "reject" option and aborts the enclosing test loop. */
	virtual bool classify(DIMTYPE &cls, const PPattern &pattern)  =0;
	virtual bool train(const PDataAccessor da, const PSubset sub) =0; //!< learns from designated training part of data
	/*! Estimates accuracy (fraction of correctly classified test patterns) using the
	    designated test data. The default implementation is the canonical accuracy loop
	    over the virtual classify(); classifiers needing extra preconditions or a batched
	    evaluation override it (Classifier_Normal_Bayes, Classifier_Multinomial_NaiveBayes,
	    Classifier_SVM, Classifier_Extern). Returns false if the test part is empty or
	    classify() failed. */
	virtual bool test(RETURNTYPE &result, const PDataAccessor da);

	/*! per-class real scores for one pattern (any monotone transform of the class
	    posteriors); enables ranking-based metrics (AUC). Default: not supported. */
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) {(void)scores; (void)pattern; return false;}

	/*! assembles the classes*classes confusion matrix (row-major, row = true class)
	    over the designated test data. Default implementation replicates the canonical
	    test() loop via the virtual classify(); concrete classifiers override only to
	    preserve special conventions (see Classifier_Multinomial_NaiveBayes). */
	virtual bool test_confusion(std::vector<RETURNTYPE> &confusion, DIMTYPE &classes, const PDataAccessor da);

	/*! collects per-sample (positive-class score, true class) pairs over the designated
	    test data (two-class data only; positive class = class index 1); the enumeration
	    order (class-major, block order) is deterministic. Default implementation uses
	    the virtual classify_scores(). */
	virtual bool test_scores(std::vector<std::pair<RETURNTYPE,DIMTYPE>> &scores, const PDataAccessor da);

	/*! collects the row-major samples x classes score matrix and true labels over the
	    designated test data (any class count; same deterministic class-major, block-order
	    enumeration as test_scores). Default implementation uses the virtual
	    classify_scores(). Feeds the one-vs-rest macro AUC. */
	virtual bool test_scores_matrix(std::vector<RETURNTYPE> &scores, std::vector<DIMTYPE> &labels, DIMTYPE &classes, const PDataAccessor da);

	/*! evaluates the designated test data under the given metric. The default spec
	    (plain accuracy, no class weights) is served directly by test(); other metrics
	    dispatch through test_confusion()/test_scores(). Returns false when the metric is
	    undefined on this test part (callers skip the split). */
	virtual bool test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec);

	Classifier* clone() const override =0;
	Classifier* sharing_clone() const override {throw fst_error("Classifier::sharing_clone() not supported, use Classifier::clone() instead.");}
	Classifier* stateless_clone() const override {throw fst_error("Classifier::stateless_clone() not supported, use Classifier::clone() instead.");}
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// the canonical accuracy loop (see test_confusion below for the setClass re-issue note)
	notify("Classifier::test().");
	assert(da);
	typename DATAACCESSOR::PPattern p;
	unsigned long long count=0, correct=0;
	DIMTYPE clstmp;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c_test=0;c_test<da->getNoOfClasses();c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p=blk.data; const auto s=blk.size;
			for(auto i=decltype(s){0};i<s;i++)
			{
				if(!classify(clstmp,&p[(std::size_t)i*stride])) return false;
				if(clstmp==c_test) correct++;
				count++;
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as the classifier's training accessor
		}
	}
	if(count==0) return false;
	result=(RETURNTYPE)correct/(RETURNTYPE)count;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_confusion(std::vector<RETURNTYPE> &confusion, DIMTYPE &classes, const PDataAccessor da)
{
	// the canonical test() loop, tallying a confusion matrix instead of correct/count.
	// NOTE: the re-issued setClass() after each block is necessary for classifiers whose
	//       classify() iterates a data accessor that may be the same instance (kNN)
	assert(da);
	classes=da->getNoOfClasses();
	if(classes<2) return false;
	confusion.assign((size_t)classes*(size_t)classes,(RETURNTYPE)0.0);
	typename DATAACCESSOR::PPattern p;
	unsigned long long count=0;
	DIMTYPE clstmp;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c_test=0;c_test<classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p=blk.data; const auto s=blk.size;
			for(auto i=decltype(s){0};i<s;i++)
			{
				if(!classify(clstmp,&p[(std::size_t)i*stride])) return false;
				if(clstmp>=classes) clstmp=(c_test+1)%classes; // "wrong class" convention (kNN, no neighbours): deterministic error column, keeps the true-class row total intact
				confusion[(size_t)c_test*(size_t)classes+(size_t)clstmp]+=1.0;
				count++;
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as the classifier's training accessor
		}
	}
	return count>0;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_scores(std::vector<std::pair<RETURNTYPE,DIMTYPE>> &scores, const PDataAccessor da)
{
	// same canonical loop, collecting (positive-class score, true class) pairs (two-class data)
	assert(da);
	const DIMTYPE classes=da->getNoOfClasses();
	if(classes!=2) return false;
	scores.clear();
	std::vector<RETURNTYPE> sc;
	typename DATAACCESSOR::PPattern p;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c_test=0;c_test<classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p=blk.data; const auto s=blk.size;
			for(auto i=decltype(s){0};i<s;i++)
			{
				if(!classify_scores(sc,&p[(std::size_t)i*stride]) || sc.size()<2) return false;
				scores.push_back(std::make_pair(sc[1]-sc[0],c_test));
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as the classifier's training accessor
		}
	}
	return !scores.empty();
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_scores_matrix(std::vector<RETURNTYPE> &scores, std::vector<DIMTYPE> &labels, DIMTYPE &classes, const PDataAccessor da)
{
	// same canonical loop as test_scores, keeping every class's score column
	assert(da);
	classes=da->getNoOfClasses();
	if(classes<2) return false;
	scores.clear(); labels.clear();
	std::vector<RETURNTYPE> sc;
	typename DATAACCESSOR::PPattern p;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c_test=0;c_test<classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p=blk.data; const auto s=blk.size;
			for(auto i=decltype(s){0};i<s;i++)
			{
				if(!classify_scores(sc,&p[(std::size_t)i*stride]) || sc.size()<(size_t)classes) return false;
				for(DIMTYPE c=0;c<classes;c++) scores.push_back(sc[c]);
				labels.push_back(c_test);
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as the classifier's training accessor
		}
	}
	return !labels.empty();
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec)
{
	if(spec.is_default()) return test(result,da); // plain accuracy, served by test() itself
	if(spec.is_regression_metric()) return false; // regression metrics apply to Regressor stacks only
	if(spec.metric==Wrapper_Metric::auc) {
		if(da->getNoOfClasses()==2) { // two classes: the exact binary AUC
			std::vector<std::pair<RETURNTYPE,DIMTYPE>> scores;
			if(!test_scores(scores,da)) return false;
			return auc_from_scores<RETURNTYPE,DIMTYPE>(result,scores);
		}
		std::vector<RETURNTYPE> scores; // K>2: one-vs-rest macro AUC
		std::vector<DIMTYPE> labels;
		DIMTYPE classes;
		if(!test_scores_matrix(scores,labels,classes,da)) return false;
		return auc_macro_from_scores<RETURNTYPE,DIMTYPE>(result,scores,labels,classes);
	}
	std::vector<RETURNTYPE> confusion;
	DIMTYPE classes;
	if(!test_confusion(confusion,classes,da)) return false;
	return metric_from_confusion<RETURNTYPE,DIMTYPE>(result,spec.metric,confusion,classes,spec.class_weights,(DIMTYPE)spec.f1_positive);
}

} // namespace
#endif // FSTCLASSIFIER_H ///:~
