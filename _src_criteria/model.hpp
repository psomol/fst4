#ifndef FSTMODEL_H
#define FSTMODEL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    model.hpp
   \brief   Defines interface for data model implementations
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
#include <iostream>
#include "error.hpp"
#include "global.hpp"

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

/*! \brief abstract class, defines interface for data model implementations

    \note Model is deliberately NOT derived from Clonable: a model is a plain value-like
    object owned by the criterion or classifier that learned it, never handed around
    through a Clonable interface. Owners therefore deep-copy their model by hand in their
    own copy-constructor (typically `new Model_X(*other._model)`), which is what makes
    clones of those owners independent. */
template<class SUBSET, class DATAACCESSOR>
class Model {
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying training data
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to a feature subset, as consumed by narrow_to()
	Model() {notify("Model constructor");}
	virtual ~Model() {notify("Model destructor");}
	//! \brief Fits the model's parameters from \p da's training data, over all its features.
	//! Concrete models may additionally offer a (da, subset) convenience overload to learn at
	//! a lower dimensionality directly - that is not part of this interface, so generic code
	//! must not assume it exists. narrow_to()/denarrow() do NOT relearn -- they only change
	//! which features of an ALREADY-learned model are exposed to subsequent evaluation.
	virtual void learn(PDataAccessor da) =0;
	//! Restricts subsequent model access to \p sub's selected features, without relearning
	//! (analogous to Distance::narrow_to() -- see there for the _index[] mechanism concrete
	//! models typically reuse).
	virtual void narrow_to(const PSubset sub) =0; // access to data will be mapped to sub-matrix
	//! Undoes narrow_to(): model access reverts to the full dimensionality it was learned at.
	virtual void denarrow() =0; // access is reset to the full matrix
};

} // namespace
#endif // FSTMODEL_H ///:~
