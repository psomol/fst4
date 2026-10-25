#ifndef FSTCRITERIONNEGATIVE_H
#define FSTCRITERIONNEGATIVE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_negative.hpp
   \brief   Criterion_Negative returns the negative of another Criterion's result
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

/*! \brief Returns the negative of another Criterion's result.

    This trivial criterion can be used to turn maximization to minimization.
    Useful, e.g., in subset size minimizing scenarios (in conjunction with
    Criterion_Subset_Size) or in feature acquisition weight minimization
    scenarios (in conjunction with Criterion_Sum_Of_Weights).

    \note is_monotonic() is left at its false default and must stay there: negating
    a criterion destroys cardinality monotonicity by construction. If the wrapped
    criterion cannot decrease when a feature is added, the negated one cannot
    increase - which is the opposite property, and not the one Branch & Bound prunes
    on. A negated criterion is therefore never a valid Branch & Bound criterion,
    however monotonic its operand.
*/
template<class CRITERION, class RETURNTYPE, class SUBSET>
class Criterion_Negative : public Clonable_Impl<Criterion_Negative<CRITERION,RETURNTYPE,SUBSET>,Criterion<RETURNTYPE,SUBSET> > { // adapter class
public:
	typedef Clonable_Impl<Criterion_Negative,Criterion<RETURNTYPE,SUBSET> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<CRITERION> PCriterion;    //!< shared pointer to the wrapped criterion whose result is negated
	typedef std::shared_ptr<SUBSET> PSubset;          //!< shared pointer to the feature subset being evaluated
	//! Constructs the adapter around \p crit, the criterion whose evaluate() result is negated.
	Criterion_Negative(const PCriterion crit): _crit(crit) {notify("Criterion_Negative constructor.");}
	virtual ~Criterion_Negative() {notify("Criterion_Negative destructor.");}
	
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) 
 override	{
		assert(sub); 
		assert(_crit); 
		RETURNTYPE _result;
		if(!_crit->evaluate(_result,sub)) return false;
		result=-_result;
		notify("Criterion_Negative result=",result,"."); 
		return true;
	}


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Negative()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; clones _crit.
	Criterion_Negative(const Criterion_Negative& cn);
	PCriterion _crit;      //!< the wrapped criterion whose evaluate() result is negated
protected:
};

template<class CRITERION, class RETURNTYPE, class SUBSET>
Criterion_Negative<CRITERION,RETURNTYPE,SUBSET>::Criterion_Negative(const Criterion_Negative& cn)
{
	notify("Criterion_Negative copy-constructor.");
	// deep-copies the wrapped criterion, so a clone shares nothing with its origin;
	// the unique_ptr returned by clone_unique() converts into the shared_ptr member
	if(cn._crit) _crit=clone_unique(*cn._crit);
}



} // namespace
#endif // FSTCRITERIONNEGATIVE_H ///:~
