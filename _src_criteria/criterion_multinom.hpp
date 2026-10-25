#ifndef FSTCRITERIONMULTINOM_H
#define FSTCRITERIONMULTINOM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_multinom.hpp
   \brief   Defines interface for feature selection criterion implementations based on multinomial model
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
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"
#include "model_multinom.hpp"

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

/*! \brief Abstract class, defines interface for feature selection criterion implementations based on multinomial model

    Intended for term-count data (text categorization and the like). The protocol is the
    same for both concrete criteria, Criterion_Multinomial_Bhattacharyya and
    Criterion_Multinomial_MI: initialize() learns ONE model over the FULL feature set from
    the training part of the data accessor, and each evaluate() then narrows that model to
    the subset being scored (or reads a cached per-feature statistic computed on the full
    model). The model is therefore learned once per data-split setting, not once per
    evaluated subset - call initialize() again whenever the accessor's split changes. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Multinomial : public Criterion_Filter<RETURNTYPE,SUBSET> { // abstract class
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Multinomial();
	virtual ~Criterion_Multinomial() {notify("Criterion_Multinomial destructor.");}
	//! \brief learns the multinomial model from \p da; subclasses must call this base before their own initialize() work.
	//! \return always true - model-learning problems are reported by throwing, not by the return value
	virtual bool initialize(PDataAccessor da);

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override =0;

	/*! both multinomial filters are monotonic w.r.t. subset cardinality, hence
	    B&B-compatible: the multinomial Bhattacharyya distance cannot decrease when
	    a feature is added, and the MI criterion is a SUM of nonnegative per-feature
	    terms (removing a feature cannot increase it) */
	virtual bool is_monotonic() const override {return true;}

	DIMTYPE get_n() const {assert(_model); return _model->get_n();} //!< the full (raw, unnarrowed) feature count the model was learned for
	DIMTYPE get_d() const {assert(_model); return _model->get_d();} //!< the currently active (narrowed) feature count

protected:
	//! Copy-constructor; deep-copies _model.
	Criterion_Multinomial(const Criterion_Multinomial& cm);
protected:
	std::unique_ptr<Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model; //!< the learned per-class multinomial (term-probability) model
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Multinomial(const Criterion_Multinomial& cm)
{
	notify("Criterion_Multinomial copy-constructor.");
	if(cm._model) _model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cm._model));
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Multinomial()
{
	notify("Criterion_Multinomial constructor.");
	_model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Multinomial::initialize().");
	assert(_model);
	_model->learn(da);
	return true; 
}


} // namespace
#endif // FSTCRITERIONMULTINOM_H ///:~
