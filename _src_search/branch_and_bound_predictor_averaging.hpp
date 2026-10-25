#ifndef FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H
#define FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    branch_and_bound_predictor_averaging.hpp
   \brief   Averaging Prediction Mechanism for use in Fast Branch & Bound and Branch & Bound with Partial Prediction
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
#include "error.hpp"
#include "global.hpp"
#include "branch_and_bound_predictor.hpp"

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

/*! \brief Averaging Prediction Mechanism in Fast Branch and Bound & Branch and Bound with Partial Prediction

    Caller convention (see Search_Branch_And_Bound_Fast/_Partial_Prediction): the
    "value" learned for a feature is the nonnegative criterion-value drop caused
    by removing that feature from a node's parent (parent value - child value),
    and the "value" predicted for a feature is used the same way to estimate an
    unevaluated child's value as parent value - predicted drop. This class itself
    is agnostic to that interpretation; it just accumulates and averages whatever
    per-feature values it is given.
*/
template<class RETURNTYPE, typename DIMTYPE>
class Branch_And_Bound_Predictor_Averaging : public Branch_And_Bound_Predictor<RETURNTYPE,DIMTYPE> {
public:
	Branch_And_Bound_Predictor_Averaging(): _gamma(1.0), _delta(1) {notify("Branch_And_Bound_Predictor_Averaging constructor.");}
	virtual ~Branch_And_Bound_Predictor_Averaging() {notify("Branch_And_Bound_Predictor_Averaging destructor.");}

	virtual void initialize(const DIMTYPE features) override; //!< (re)initializes the per-feature sum/count accumulators to zero for \p features features; call before use for each new search
	virtual bool learn(const DIMTYPE feature, const RETURNTYPE value) override; //!< adds one observed \p value (non-negative, asserted) to \p feature 's running sum and increments its sample count; always returns true
	virtual bool predict(const DIMTYPE feature, RETURNTYPE &value) const override; //!< sets \p value to the gamma-scaled running average learned for \p feature and returns true, but only once at least \c _delta samples have been learned for it; returns false otherwise
			unsigned long get_count(const DIMTYPE feature) const {assert(feature<_count.size()); return _count[feature];} //!< the number of learn() calls recorded so far for \p feature - a diagnostic accessor, telling how well founded the feature's prediction is (predict() succeeds once this reaches the delta set by set_delta())

			void set_gamma(const RETURNTYPE gamma=1.0) {assert(gamma>=0.0); _gamma=gamma;} //!< sets the scaling factor applied to the averaged value in predict() (asserted >=0.0); default 1.0 = plain average
			void set_delta(const unsigned int delta=1) {assert(delta>=1); _delta=delta;} //!< sets the minimum sample count a feature must reach before predict() returns a value (asserted >=1); default 1 = predict as soon as one sample is learned

	virtual std::ostream& print(std::ostream& os) const override; //!< writes gamma/delta plus, for every feature with at least one learned sample, its current predicted value and sample count, and the total sample count across all features
protected:
	std::vector<RETURNTYPE>    _value; //!< sums up 'learned' values
	std::vector<unsigned long> _count; //!< records the number of summed up values

	RETURNTYPE    _gamma; //!< scaling factor applied to the averaged prediction; see set_gamma()
	unsigned int  _delta; //!< minimum sample count before predict() returns a value; see set_delta()
};

template<class RETURNTYPE, typename DIMTYPE>
void Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::initialize(const DIMTYPE features)
{
	assert(features>=1);
	_value.clear(); _value.resize(features);
	_count.clear(); _count.resize(features);
	for(DIMTYPE i=0;i<features;i++) {_value[i]=0.0; _count[i]=0;}
}

template<class RETURNTYPE, typename DIMTYPE>
bool Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::learn(const DIMTYPE feature, const RETURNTYPE value)
{
	assert(value>=0.0);
	assert(feature<_value.size());
	_value[feature]+=value;
	_count[feature]++;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE>
bool Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::predict(const DIMTYPE feature, RETURNTYPE &value) const
{
	assert(feature<_value.size());
	if(_count[feature]>=_delta) {	
		value=_gamma*(_value[feature]/(RETURNTYPE)_count[feature]);
		return true;
	}
	return false;
}

template<class RETURNTYPE, typename DIMTYPE>
std::ostream& Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::print(std::ostream& os) const
{
	os << "Branch_And_Bound_Predictor_Averaging(gamma="<<_gamma<<", delta="<<_delta<<")"; 
	RETURNTYPE val;
	unsigned long cnt=0;
	for(DIMTYPE i=0;i<_value.size();i++) if(_count[i]>0) {
		if(predict(i,val)) os << std::endl << "feature "<<i<<": predictor value="<<val<< ", count="<<_count[i];
		cnt+=_count[i];
	}
	if(cnt>0) os << std::endl << "Total count: "<<cnt;
	return os;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H ///:~
