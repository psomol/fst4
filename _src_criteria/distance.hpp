#ifndef FSTDISTANCE_H
#define FSTDISTANCE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    distance.hpp
   \brief   Defines interface for distance evaluators (to be used primarily in Classifier_kNN)
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
#include <cmath>
#include <list>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"

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

//! Abstract class, defines interface for distance evaluators to be used in Classifier_kNN
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Distance : public Clonable {
public:
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to a feature subset, as consumed by narrow_to()
	Distance() : _n(0), _d(0) {}
	//! Copy-constructor; deep-copies the current _index[] mapping (same narrowed/denarrowed state as \p d).
	Distance(const Distance &d);
	virtual ~Distance() {}
	//! \brief Allocates for \p n raw features and makes _index[] the identity map 0..n-1, with all \p n features active (get_d()==get_n()); narrow_to() restricts that.
	void reset(const DIMTYPE n);

	//! \brief Restricts distance() to \p sub's selected features only.
	//! Rewrites _index[] to the subset's selected feature indices (raw-feature order) and sets
	//! get_d() to the subset size; distance() then reads only those _d entries of a pattern,
	//! through _index[], instead of all _n. Precondition: reset() must have been called with
	//! n >= the subset size.
	void narrow_to(const PSubset sub); // fill _index[] -> access to data is then mapped to sub-matrix
	//! Undoes narrow_to(): _index[] reverts to the identity map and get_d() returns to get_n().
	void denarrow(); // reset _index[] -> access is reset to the full matrix

	DIMTYPE get_n() const {return _n;} //!< the full (raw, unnarrowed) feature count fixed by reset()
	DIMTYPE get_d() const {return _d;} //!< the currently active feature count: the narrowed subset size after narrow_to(), get_n() after reset()/denarrow()

	//! \brief Distance between two raw patterns over the currently active features only.
	//! Reads exactly get_d() entries of each pattern, at the raw indices in _index[] (identity
	//! unless narrow_to() is in effect) -- concrete distances must index through _index[], never
	//! assume a contiguous 0..get_d()-1 layout.
	//! \param pattern1,pattern2 raw feature vectors of at least get_n() entries each
	//! \return a non-negative dissimilarity: 0 for identical patterns, larger values mean
	//!         farther apart. Implementations are free to return any monotone transform of a
	//!         true distance (e.g. a squared or fractionally powered one), so the value need
	//!         NOT satisfy the triangle inequality and need not be a metric. Only the induced
	//!         ORDER of neighbours matters to the callers (Classifier_kNN).
	virtual DATATYPE distance(const DATATYPE *pattern1, const DATATYPE *pattern2) =0;

	Distance* clone() const override =0;
	Distance* sharing_clone() const override {throw fst_error("Distance::sharing_clone() not supported, use Distance::clone() instead.");}
	Distance* stateless_clone() const override {throw fst_error("Distance::stateless_clone() not supported, use Distance::clone() instead.");}
protected:
	DIMTYPE _n;                          //!< full raw feature count, fixed by reset()
	DIMTYPE _d;                          //!< currently active feature count (see get_d())
	std::unique_ptr<DIMTYPE[]> _index;   //!< maps active-feature position [0,_d) to a raw feature index in [0,_n)
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Distance<DATATYPE,DIMTYPE,SUBSET>::Distance(const Distance& d) :
	_n(d._n),
	_d(d._d)
{
	notify("Distance copy-constructor.");
	if(_n>0) {_index.reset(new DIMTYPE[_n]); memcpy((void *)_index.get(),(void *)(d._index).get(),sizeof(DIMTYPE)*_n);}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n)
{
	assert(n>0);
	_n=n;
	_index.reset(new DIMTYPE[_n]);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub)
{
	assert(_n>=sub->get_d_raw()); // precondition: enough space to store the subset
	assert(_index);
	_d=0;
	for(DIMTYPE f : sub->selected_features()) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::denarrow()
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}


} // namespace
#endif // FSTDISTANCE_H ///:~
