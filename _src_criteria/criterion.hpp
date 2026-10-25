#ifndef FSTCRITERION_H
#define FSTCRITERION_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion.hpp
   \brief   Defines interface for feature selection criterion classes
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

//! Abstract class, defines interface for feature selection criterion classes
template<class RETURNTYPE, class SUBSET>
class Criterion : public Clonable { // abstract class
public:
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being evaluated
	Criterion() {}
	virtual ~Criterion() {}

	//! \brief Scores the given feature subset; higher is better (FST4 searches maximize).
	//! \param result set to the criterion value on success; left unspecified on failure
	//! \param sub the (fixed) subset to evaluate -- only its selected features are read
	//! \return false if the subset cannot be evaluated at all (e.g. too small for the
	//!         model, or singular statistics). This is NOT a per-candidate reject: callers
	//!         treat it as fatal for the current search step or run -- the sequential-step
	//!         candidate evaluator marks its whole batch failed, streaming searches abort,
	//!         and Branch & Bound throws. A criterion that can legitimately be undefined on
	//!         some subsets must guard against reaching those subsets another way (e.g. a
	//!         minimum-subset-size constraint), not by relying on evaluate() to skip them.
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) =0;

	/*! Declares monotonicity with respect to subset cardinality: true means removing
	    a feature can NEVER increase the criterion value. Branch & Bound prunes on this
	    property, so ALL of its optimality guarantees rest on this declaration - and the
	    library cannot verify it, it is an assertion made by the criterion's author. A
	    false declaration silently turns Branch & Bound into a heuristic that may skip
	    the optimum. Search_Branch_And_Bound therefore prints a warning at search start
	    when the supplied criterion does not declare the property, and the fst4
	    command-line tool refuses the `--search bb-*` algorithms outright for such a
	    criterion. Defaults to false: user-supplied criteria should override to true
	    only if monotonicity provably holds. The built-in normal-model and multinomial
	    filters declare true (see Criterion_Normal and Criterion_Multinomial); wrapper
	    criteria and the adapter criteria (negative/subset-size/sum-of-weights)
	    deliberately keep the false default. */
	virtual bool is_monotonic() const {return false;}

	Criterion* clone() const override =0;
	Criterion* sharing_clone() const override {throw fst_error("Criterion::sharing_clone() not supported, use Criterion::clone() instead.");}
	Criterion* stateless_clone() const override {throw fst_error("Criterion::stateless_clone() not supported, use Criterion::clone() instead.");}
};

/*! \brief Abstract class, defines interface for classes implementing "independent" feature selection criteria
     \note Counterpart to Criterion_Wrapper, except that it serves only as basis for further specializations
*/
template<class RETURNTYPE, class SUBSET>
class Criterion_Filter : public Criterion<RETURNTYPE,SUBSET> { // abstract class
public:
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being evaluated
	Criterion_Filter() {}
	virtual ~Criterion_Filter() {}
};


} // namespace
#endif // FSTCRITERION_H ///:~
