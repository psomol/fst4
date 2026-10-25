#ifndef FSTCRITERIONWRAPPER_H
#define FSTCRITERIONWRAPPER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_wrapper.hpp
   \brief   Criterion_Wrapper adapts Classifier objects to serve as feature selection criterion
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
#include <cmath>
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"
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

/*! \brief Wraps Classifier objects to serve as feature selection criterion

    Each evaluate() call runs the full split protocol of the data accessor supplied to
    initialize(): for every split the wrapped classifier is trained on the TRAIN part and
    evaluated on the TEST part, and the criterion value is the MEAN of the per-split
    values. What is measured per split is the configured wrapper metric (see
    set_metric()); by default that is plain classification accuracy. All supported
    metrics are oriented so that higher is better, as FST4 searches maximize. */
template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
class Criterion_Wrapper : public Clonable_Impl<Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>,Criterion<RETURNTYPE,SUBSET> > { // adapter class
public:
	typedef Clonable_Impl<Criterion_Wrapper,Criterion<RETURNTYPE,SUBSET> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<CLASSIFIER> PClassifier;      //!< shared pointer to the wrapped classifier
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor supplying training/test splits
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Wrapper() {notify("Criterion_Wrapper constructor.");}
	virtual ~Criterion_Wrapper() {notify("Criterion_Wrapper destructor.");}
	
	//! \brief Stores the classifier and the data accessor to be used by every later evaluate().
	//! Must be called before the first evaluate(). Both are kept by shared_ptr for the whole
	//! life cycle of *this; the accessor must already be initialized and must carry the split
	//! setting the evaluation is to run over. Clones take a clone of the classifier and a
	//! sharing clone of the accessor, so a threaded search never shares either instance.
	virtual void initialize(const PClassifier classifier, const PDataAccessor da);

	//! \brief Trains and tests the wrapped classifier on every split, returning the mean of the
	//! per-split values of the configured metric (plain accuracy by default; see set_metric()).
	//! Returns false when training fails on any split, or when the metric is undefined on all of them.
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;

	/*! selects the wrapper metric (default: plain accuracy).
	    The spec is immutable configuration, copied by clones - thread-safe, RNG-free. */
	void set_metric(const Metric_Spec<RETURNTYPE> &spec) {_metric_spec=spec;}
	const Metric_Spec<RETURNTYPE>& get_metric() const {return _metric_spec;} //!< returns the currently configured wrapper metric (see set_metric())


	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Wrapper()"; if(_classifier) os << " with " << *_classifier; return os;}
protected:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; clones _classifier and sharing-clones _da, copies _metric_spec.
	Criterion_Wrapper(const Criterion_Wrapper& cw);

	std::shared_ptr<CLASSIFIER> _classifier;    //!< the wrapped classifier trained then tested by evaluate()
	std::shared_ptr<DATAACCESSOR> _da;          //!< data accessor supplying the train/test splits
	Metric_Spec<RETURNTYPE> _metric_spec;       //!< configured wrapper metric; default = plain accuracy (see set_metric())
};

template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::Criterion_Wrapper(const Criterion_Wrapper& cw)
{
	notify("Criterion_Wrapper copy-constructor.");
	_classifier=clone_unique(*cw._classifier);
	_da=sharing_clone_unique(*cw._da);
	_metric_spec=cw._metric_spec;
}



template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
void Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::initialize(const PClassifier classifier, const PDataAccessor da) // passing by std::shared_ptr to enable keeping the reference throughout *this life cycle
{
	assert(da);
	assert(classifier);
	_da=da;
	_classifier=classifier;
}

template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
bool Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	assert(_da);
	assert(_classifier);
	RETURNTYPE val;
	result=0.0;
	RETURNTYPE cnt=0.0;
	if(!_metric_spec.is_default()) {
		// non-default metric: per-split metric, then mean over splits (macro over folds,
		// same shape as the accuracy path below). Splits on which the metric is undefined
		// (e.g. AUC with a single-class fold) are skipped; all-splits-undefined = failure.
		for(bool b=_da->getFirstSplit();b==true;b=_da->getNextSplit()) {
			if(!_classifier->train(_da,sub)) return false;
			if(_classifier->test_metric(val,_da,_metric_spec)) {
				result+=val;
				cnt+=1.0;
			}
		}
		if(cnt==0.0) return false;
		result/=(RETURNTYPE)cnt;
		return true;
	}
	for(bool b=_da->getFirstSplit();b==true;b=_da->getNextSplit()) {
		if(!_classifier->train(_da,sub)) return false;
		if(!_classifier->test(val,_da)) return false;
		result+=val;
		cnt+=1.0;
	}
	if(cnt==0.0) return false; // no split produced (an unprepared or exhausted splitter)
	result/=(RETURNTYPE)cnt;
	return true;
}

} // namespace
#endif // FSTCRITERIONWRAPPER_H ///:~
