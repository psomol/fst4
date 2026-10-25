#ifndef FSTCRITERIONSUBSETSIZE_H
#define FSTCRITERIONSUBSETSIZE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_subsetsize.hpp
   \brief   Criterion_Subset_Size returns the (positive) subset size - rates larger subsets higher; wrap in Criterion_Negative to favor smaller subsets
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

/*! \brief Returns the (positive) subset size sub->get_d() - since FST4 searches
    maximize, this rates LARGER subsets higher. Wrap in Criterion_Negative to
    favor smaller subsets instead.

    This trivial criterion is intended as secondary criterion only to be used
    in conjunction with Result_Tracker_Regularizer, implementing feature
    selection regularization (improvement of feature selection generalization
    ability and result robustness) as described in paper "Somol, Grim, Pudil:
    The Problem of Fragile Feature Subset Preference in Feature Selection Methods
    and A Proposal of Algorithmic Workaround. In Proc. ICPR 2010.  IEEE Computer
    Society, 2010". In the fst4 command-line tool this is what
    `--regularize-by subsetsize` selects, together with `--regularize-margins`
    and `--tracker regularizer`.

    \note Unlike the model-based criteria, which score the raw marker set they are
    handed and therefore read Subset::get_d_raw(), this one reads Subset::get_d():
    the count of currently selected features EXCLUDING frozen ones, independent of
    the forward/backward search mode. That is the subset size the search itself
    manipulates, which is the quantity the regularization trade-off is about.
*/
template<class RETURNTYPE, class SUBSET>
class Criterion_Subset_Size : public Clonable_Impl<Criterion_Subset_Size<RETURNTYPE,SUBSET>,Criterion<RETURNTYPE,SUBSET> > { // adapter class
public:
	typedef Clonable_Impl<Criterion_Subset_Size,Criterion<RETURNTYPE,SUBSET> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Subset_Size() {notify("Criterion_Subset_Size constructor.");}
	virtual ~Criterion_Subset_Size() {notify("Criterion_Subset_Size destructor.");}
	
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override {assert(sub); result=RETURNTYPE(sub->get_d()); notify("CSS result=",result,", getd ",sub->get_d(),"."); return true;}


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Subset_Size()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; the class carries no state of its own beyond what parent_impl copies.
	Criterion_Subset_Size(const Criterion_Subset_Size& css);
protected:
};

template<class RETURNTYPE, class SUBSET>
Criterion_Subset_Size<RETURNTYPE,SUBSET>::Criterion_Subset_Size(const Criterion_Subset_Size& css)
{
	notify("Criterion_Subset_Size copy-constructor.");
}



} // namespace
#endif // FSTCRITERIONSUBSETSIZE_H ///:~
