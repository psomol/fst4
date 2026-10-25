#ifndef FSTREGRESSOR_H
#define FSTREGRESSOR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    regressor.hpp
   \brief   Defines interface for regressor implementations
   \author  Jiri Grim (grim@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    2012 (last updated July 2026)
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
#include "clonable.hpp"
#include "wrapper_metrics.hpp"
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

/*! \brief abstract class, defines interface for Regressor implementations (mainly to be used in wrappers)

    The counterpart of Classifier for the regression pipeline; the same lifecycle applies -
    train() first, then test*()/predict(), with a data accessor exposing the same features
    in the same order. Two regressors are shipped: Regressor_Weighted_Neighbours
    (distance-weighted nearest neighbours) and Regressor_Ridge (closed-form L2-regularized
    least squares).

    \note DEPENDENT-VARIABLE CONVENTION. The dependent variable predicted here is by
    default the CLASS LABEL of the sample, not a data column - "regression simulating
    classification", which is what makes a regressor usable as a feature-selection
    criterion on ordinary classification data. Concrete regressors express this with the
    sentinel index -1 (set_dependent_index(-1)), their default; a non-negative index
    selects an actual data column instead.
    \note Every value produced for a criterion is oriented HIGHER IS BETTER, as FST4
    searches maximize: test() returns a hit rate under the class-label convention and a
    NEGATED error otherwise, and the regression metrics of the metric palette are r2,
    neg-rmse and neg-mae (see Metric_Spec). */
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Regressor : public Clonable { // abstract class
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset;            //!< shared pointer to the (fixed, non-reassignable) feature subset being evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Regressor() {}
	virtual ~Regressor() {}

	//! \brief Predicts the dependent variable of one pattern into \p depvar (see the
	//! dependent-variable convention in the class documentation). Returns false when no
	//! prediction can be made at all - the regressor has not been trained, or the
	//! implementation hit an unrecoverable condition; false aborts the enclosing test loop.
	virtual bool predict(RETURNTYPE &depvar, const PPattern &pattern)  =0;
	virtual bool train(const PDataAccessor da, const PSubset sub) =0; //!< learns from designated training part of data
	//! \brief Scores the designated test data with the regressor's own default measure,
	//! oriented higher-is-better (a hit rate under the class-label convention, a negated
	//! prediction error otherwise). Returns false if the test part is empty or predict() failed.
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) =0;

	/*! evaluates the designated test data under the given regression metric (r2/neg-rmse/neg-mae);
	    the default spec is served by test() itself. Classification metrics do not apply to a
	    regression stack and make this return false. The dependent-variable convention is the
	    concrete regressor's - see the class documentation. */
	virtual bool test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec)
		{if(spec.is_default()) return test(result,da); (void)result; (void)da; return false;}

	Regressor* clone() const override =0;
	Regressor* sharing_clone() const override {throw fst_error("Regressor::sharing_clone() not supported, use Regressor::clone() instead.");}
	Regressor* stateless_clone() const override {throw fst_error("Regressor::stateless_clone() not supported, use Regressor::clone() instead.");}
};

} // namespace
#endif // FSTREGRESSOR_H ///:~
