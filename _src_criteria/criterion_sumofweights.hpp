#ifndef FSTCRITERIONSUMOFWEIGHTS_H
#define FSTCRITERIONSUMOFWEIGHTS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_sumofweights.hpp
   \brief   Criterion_Sum_Of_Weights returns sum of pre-specified feature weights for features in the evaluated subset
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    October 2010 (last updated July 2026)
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
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"

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

/*! \brief Returns sum of pre-specified feature weights for features in the evaluated subset.

    This trivial criterion is intended as secondary criterion to be used
    in conjunction with Result_Tracker_Regularizer, allowing to find subset
    among those close to the known best such that the sum of known feature weights 
    (e.g., feature acquisition cost) is minimized. This is usable, e.g., in medicine
    where different measurements have different costs (e.g., measuring body temperature 
    may be cheaper than laboratory tests, etc.). The technique is described in paper 
    "Somol, Grim, Pudil: The Problem of Fragile Feature Subset Preference in Feature Selection Methods 
    and A Proposal of Algorithmic Workaround. In Proc. ICPR 2010.  IEEE Computer 
    Society, 2010".
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
class Criterion_Sum_Of_Weights : public Clonable_Impl<Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>,Criterion<RETURNTYPE,SUBSET> > { // adapter class
public:
	typedef Clonable_Impl<Criterion_Sum_Of_Weights,Criterion<RETURNTYPE,SUBSET> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	typedef std::shared_ptr<RETURNTYPE> PWeights;          //!< shared pointer to a raw array of \p noofweights per-feature weights
	Criterion_Sum_Of_Weights() : _noofweights(0) {notify("Criterion_Sum_Of_Weights constructor.");}
	virtual ~Criterion_Sum_Of_Weights() {notify("Criterion_Sum_Of_Weights destructor.");}

	//! \brief Adopts an existing weight array: SHARES ownership with the caller, nothing is copied.
	//! \warning The two initialize() overloads differ in ownership. This one keeps the caller's
	//! buffer alive through the shared_ptr and lets the caller keep modifying it; the array[]
	//! overload below takes a private copy instead. Note that clones always deep-copy, so a
	//! cloned criterion stops sharing the array whichever overload was used.
	virtual void initialize(const DIMTYPE noofweights, const PWeights weights) {_noofweights=noofweights; _weights=weights;}
	//! \brief Copies \p noofweights weights from a raw array into a freshly owned buffer
	//! (the caller's array is not referenced afterwards); see the ownership warning above.
	virtual void initialize(const DIMTYPE noofweights, const RETURNTYPE weights[]);

	//! \brief Sums up the weights of the features selected in \p sub; always returns true.
	//! Weights are indexed by RAW feature index, so the array must hold exactly one entry per
	//! feature of the full feature set (asserted: noofweights == sub->get_n()).
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Sum_Of_Weights()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; deep-copies the weight array into a freshly owned buffer.
	Criterion_Sum_Of_Weights(const Criterion_Sum_Of_Weights& css);
protected:
	PWeights _weights;             //!< per-feature weights, indexed by raw feature index (size _noofweights)
	DIMTYPE _noofweights;          //!< number of entries in _weights; must equal the evaluated subset's get_n()
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::Criterion_Sum_Of_Weights(const Criterion_Sum_Of_Weights& css) :
	_noofweights(css._noofweights)
{
	notify("Criterion_Sum_Of_Weights copy-constructor.");
	if(css._weights)
	{
		assert(css._noofweights>0);
		_weights.reset(new RETURNTYPE[css._noofweights], std::default_delete<RETURNTYPE[]>());
		for(DIMTYPE i=0;i<css._noofweights;i++)_weights.get()[i]=css._weights.get()[i];
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
void Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::initialize(const DIMTYPE noofweights, const RETURNTYPE weights[])
{
	assert(noofweights>0);
	_noofweights=noofweights;
	_weights.reset(new RETURNTYPE[_noofweights], std::default_delete<RETURNTYPE[]>());
	for(DIMTYPE i=0;i<_noofweights;i++) _weights.get()[i]=weights[i];
}



template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
bool Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::evaluate(RETURNTYPE &result, PSubset sub)
{
	assert(_weights);
	assert(sub);
	assert(_noofweights==sub->get_n());

	result=0;
	for(DIMTYPE fidx : sub->selected_features()) result+=_weights.get()[fidx];
	return true;
}

} // namespace
#endif // FSTCRITERIONSUMOFWEIGHTS_H ///:~
