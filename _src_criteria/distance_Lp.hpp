#ifndef FSTDISTANCELP_H
#define FSTDISTANCELP_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    distance_Lp.hpp
   \brief   Implements L_{p} distance, p equals numerator/denominator (template parameters)
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
#include "distance.hpp"

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

/*! \brief Implements L_{p} distance, p equals numerator/denominator (template parameters)

    ( sum over the currently active features of |pattern1[f] - pattern2[f]|^p )^(1/p),
    with the exponent p fixed at compile time as the ratio of the two int template
    parameters (so that a fractional p can be expressed without a floating-point template
    argument). The defaults numerator=denominator=1 give p=1, i.e. plain L_1 - use
    Distance_L1 for that case, it is cheaper. Typical instantiations are p=2 (Euclidean,
    <...,2,1>) and fractional exponents such as p=1/2 (<...,1,2>).

    \note p >= 1 yields a true metric; p < 1 does NOT - the triangle inequality fails, so
    the result is a dissimilarity only. That is legitimate here (Classifier_kNN needs only
    the neighbour ORDER), and fractional exponents are sometimes preferred in very high
    dimensions, but do not rely on metric properties.
    \note p must not be 0 - the final pow(result,1.0/p) would divide by zero. */
template<typename DATATYPE, typename REALTYPE, typename DIMTYPE, class SUBSET, const int numerator=1, const int denominator=1>
class Distance_Lp : public Clonable_Impl<Distance_Lp<DATATYPE,REALTYPE,DIMTYPE,SUBSET,numerator,denominator>,Distance<DATATYPE,DIMTYPE,SUBSET> > {
public:
	typedef Distance<DATATYPE,DIMTYPE,SUBSET> parent; //!< base class: the abstract distance interface
	typedef Clonable_Impl<Distance_Lp,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Distance
	Distance_Lp() : _p(REALTYPE(numerator)/REALTYPE(denominator)) {notify("Distance_Lp (empty) constructor.");} // base Distance() zeroes _n and _d
	//! Constructs and immediately reset()s for \p n raw features (see Distance::reset()).
	Distance_Lp(const DIMTYPE n) : _p(REALTYPE(numerator)/REALTYPE(denominator)) {parent::reset(n); notify("Distance_Lp constructor.");}
	//! Copy-constructor; delegates to parent_impl (deep-copies the current _index[] mapping) and copies _p.
	Distance_Lp(const Distance_Lp& dlp) : parent_impl(dlp), _p(dlp._p) {notify("Distance_Lp copy-constructor.");}
	virtual ~Distance_Lp() {notify("Distance_Lp destructor.");}
	
	virtual DATATYPE distance(const DATATYPE *pattern1, const DATATYPE *pattern2) override;
	

	
	virtual std::ostream& print(std::ostream& os) const override {os << "Distance_Lp(p=" << _p << ")"; return os;}
protected:
	const REALTYPE _p; //!< the L_p exponent, fixed at construction as numerator/denominator (both compile-time template parameters)
};

template<typename DATATYPE, typename REALTYPE, typename DIMTYPE, class SUBSET, const int numerator, const int denominator>
DATATYPE Distance_Lp<DATATYPE,REALTYPE,DIMTYPE,SUBSET,numerator,denominator>::distance(const DATATYPE *pattern1, const DATATYPE *pattern2)
{
	assert(parent::_index);
	assert(parent::_d>0);
	assert(pattern1!=nullptr);
	assert(pattern2!=nullptr);
	DATATYPE result=0;
	DIMTYPE index_i;
	for(DIMTYPE i=0;i<parent::_d;i++) {
		index_i=parent::_index[i];
		result+=std::pow(std::abs(pattern1[index_i]-pattern2[index_i]),_p); // qualified: bare abs(double) risks the int overload on some toolchains
	}
	return std::pow(result,1.0/_p);
}

} // namespace
#endif // FSTDISTANCELP_H ///:~
