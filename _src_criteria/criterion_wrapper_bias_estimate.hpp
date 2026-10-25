#ifndef FSTCRITERIONWRAPPERBIASESTIMATE_H
#define FSTCRITERIONWRAPPERBIASESTIMATE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_wrapper_bias_estimate.hpp
   \brief   Wrapper Bias Evaluator - returns classifier bias estimate (difference between accuracy on training and validation data parts)
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
#include "criterion_wrapper.hpp"

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

/*! \brief Wrapper Bias Evaluator - returns bias estimate (difference of classifier accuracy on training and validation data parts)

   \details Wraps Classifier objects to serve as secondary feature selection criterion.
   Trains classifier on training data, then evaluates the difference between estimated classification 
   accuracy on training data and test data. Especially in small-sample scenarios a considerable difference
   (i.e., bias) can be expected. Large bias can be viewed as a sign of overfitting and can thus
   be used as indicator of feature selection process failure.
      
   \note Expect at least 2x slower operation than that of Criterion_Wrapper.
*/
template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
class Criterion_Wrapper_Bias_Estimate : public Clonable_Impl<Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>,Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> > { // adapter class
public:
	typedef Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> parent; //!< base class: plain accuracy wrapper criterion
	typedef Clonable_Impl<Criterion_Wrapper_Bias_Estimate,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<CLASSIFIER> PClassifier;      //!< shared pointer to the wrapped classifier
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor supplying training/test splits
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Wrapper_Bias_Estimate():_collected_bias_estimate(0.0),_collected_bias_count(0.0) {notify("Criterion_Wrapper_Bias_Estimate constructor.");}
	virtual ~Criterion_Wrapper_Bias_Estimate() {notify("Criterion_Wrapper_Bias_Estimate destructor.");}

	//! \note The inherited Criterion_Wrapper::initialize() must be called before the first evaluate().
	//! \note set_metric() is inherited but has NO effect here: the bias is always computed from
	//!       plain classification accuracy, never from the configured wrapper metric.

	//! \brief Trains on the TRAIN part of each split, tests on both parts, and returns the mean
	//! per-split bias: accuracy on the (dependent) train part minus accuracy on the test part.
	//! Higher values indicate more overfitting: unlike the other criteria this one is a
	//! diagnostic whose LOW values are the good ones, so a maximizing search must not be
	//! pointed at it directly (Criterion_Negative flips it if needed). Side effect: the
	//! per-split biases are also accumulated for getBiasEstimate().
	//! Returns false when training or testing fails on any split.
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;

	RETURNTYPE getBiasEstimate() const {if(_collected_bias_count>0.0) return _collected_bias_estimate/_collected_bias_count; else return 0.0;} //!< returns the mean per-split bias over all splits evaluated since the last reset (each evaluate() call contributes one value per split), or 0 when nothing has been accumulated
	void resetBiasEstimate() {_collected_bias_estimate=0.0; _collected_bias_count=0.0;} //!< clears the accumulated bias estimate and the split count


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Wrapper_Bias_Estimate()"; if(parent::_classifier) os << " with " << *parent::_classifier; return os;}
protected:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; delegates to parent_impl (clones _classifier, sharing-clones _da) and copies the collected bias accumulators.
	Criterion_Wrapper_Bias_Estimate(const Criterion_Wrapper_Bias_Estimate& cwbe) : parent_impl(cwbe), _collected_bias_estimate(cwbe._collected_bias_estimate), _collected_bias_count(cwbe._collected_bias_count) {}

	RETURNTYPE _collected_bias_estimate;  //!< running sum of per-split bias (train-part minus test-part accuracy) since the last resetBiasEstimate()
	RETURNTYPE _collected_bias_count;     //!< number of SPLITS accumulated into _collected_bias_estimate since the last reset (one evaluate() call adds as many as the accessor has splits)
};



template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
bool Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	assert(parent::_da);
	assert(parent::_classifier);
	RETURNTYPE val1, val2, bias=0.0;
	result=0.0;
	RETURNTYPE cnt=0.0;
	// bias estimation
	parent::_da->resubstitute();
	for(bool b=parent::_da->getFirstSplit();b==true;b=parent::_da->getNextSplit()) {
		// train on TRAIN data part
		if(!parent::_classifier->train(parent::_da,sub)) return false;
		// test on TRAIN (dependent) data part
		parent::_da->substitute(TEST,TRAIN);
		if(!parent::_classifier->test(val2,parent::_da)) return false;
		// test on TEST (independent) data part
		parent::_da->substitute(TEST,TEST);
		if(!parent::_classifier->test(val1,parent::_da)) return false;

		bias+=val2-val1;
		cnt+=1.0;
	}
	if(cnt==0.0) return false; // no split produced (an unprepared or exhausted splitter)
	_collected_bias_estimate+=bias;
	_collected_bias_count+=(RETURNTYPE)cnt;
	result=bias/(RETURNTYPE)cnt;
	return true;
}

} // namespace
#endif // FSTCRITERIONWRAPPERBIASESTIMATE_H ///:~
