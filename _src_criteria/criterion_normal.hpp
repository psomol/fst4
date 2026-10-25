#ifndef FSTCRITERIONNORMAL_H
#define FSTCRITERIONNORMAL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_normal.hpp
   \brief   Defines interface for feature selection criterion implementations based on normal model
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
#include <string>
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"
#include "model_normal.hpp"
#include "indexed_vector.hpp"
#include "indexed_matrix.hpp"

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

/*! \brief Abstract class, defines interface for feature selection criterion implementations based on normal model

    \note This class also provides the template-method skeleton shared by the concrete
    two-class normal criteria (Bhattacharyya, Divergence, GMahalanobis): initialize()
    learns the model, allocates buffers and computes the shared mean-difference vector
    before delegating criterion-specific pre-computation to hooks; evaluate() narrows the
    model and delegates the math to evaluate_narrowed(). Subclasses keep only their
    buffers, their pre-computation and their math.
    \note All these criteria are defined for TWO-CLASS data only - initialize() throws
    fst_error when the data accessor reports a different number of classes. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal : public Criterion_Filter<RETURNTYPE,SUBSET> { // abstract class
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Normal();
	virtual ~Criterion_Normal() {notify("Criterion_Normal destructor.");}
	//! \brief template method: learn model, compute _meandif, then allocate_buffers() + update_derived_model().
	//! \throw fst_error if \p da does not describe exactly 2 classes.
	//! \return true (failures are reported by throwing)
	virtual bool initialize(PDataAccessor da);

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override; //!< template method: narrow model and _meandif to sub, then evaluate_narrowed()

	/*! all three normal-model filters (Bhattacharyya, Generalized Mahalanobis,
	    Divergence) are monotonic w.r.t. subset cardinality, hence B&B-compatible;
	    Divergence holds the property in theory only - see the numerical-instability
	    known issue (prefer Bhattacharyya for B&B) */
	virtual bool is_monotonic() const override {return true;}

	DIMTYPE get_n() const {assert(_model); return _model->get_n();} //!< the full (raw, unnarrowed) feature count the model was learned for
	DIMTYPE get_d() const {assert(_model); return _model->get_d();} //!< the currently active (narrowed) feature count, see evaluate()

	virtual std::ostream& print(std::ostream& os) const override {os << criterion_name() << "()"; return os;}
protected:
	//! Copy-constructor; deep-copies _model and _meandif (subclasses copy their own buffers separately).
	Criterion_Normal(const Criterion_Normal& cn);
protected:
	// hooks for the concrete two-class normal criteria, called by the template methods above
	virtual const char* criterion_name() const =0; //!< class name, used in output and error messages
	virtual void allocate_buffers() =0; //!< (re)allocate criterion-specific work buffers for get_n() features (use the alloc_* helpers)
	virtual void update_derived_model() =0; //!< precompute criterion-specific derivations of the freshly learned model (_meandif is already computed)
	//! \brief narrow remaining buffers to sub, redim work buffers to get_d(), compute the criterion value.
	//! On entry _model and _meandif have ALREADY been narrowed to \p sub by evaluate(), so the
	//! implementation sees only the d selected features; get_d() is that d, and _meandif->at(i)
	//! addresses the narrowed positions. \p sub is non-empty (evaluate() rejects d==0 before the call).
	virtual bool evaluate_narrowed(RETURNTYPE &result, const PSubset sub) =0;

	//! lazy (re)allocation of a work vector for get_n() features, for use from allocate_buffers()
	void alloc_vector(std::unique_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > &v) {
		if(!v) v.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>);
		if(get_n()>v->get_n()) v->reset(get_n());
	}
	//! lazy (re)allocation of a work matrix for get_n() features; LUstorage as required by Indexed_Matrix::LUdecompose() temp storage
	template<typename MATRIXVALUETYPE>
	void alloc_matrix(std::unique_ptr<Indexed_Matrix<MATRIXVALUETYPE,DIMTYPE,SUBSET> > &m, const bool LUstorage=false) {
		if(!m) m.reset(new Indexed_Matrix<MATRIXVALUETYPE,DIMTYPE,SUBSET>(get_n(),LUstorage));
		if(get_n()>m->get_n()) m->reset(get_n(),LUstorage);
	}
protected:
	std::unique_ptr<Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model; //!< the learned per-class normal (mean/covariance) model, narrowed in evaluate()
	std::unique_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _meandif; //!< mean[0]-mean[1], shared by all two-class normal criteria
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal(const Criterion_Normal& cn)
{
	notify("Criterion_Normal copy-constructor.");
	if(cn._model) _model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cn._model));
	if(cn._meandif) _meandif.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*cn._meandif));
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal()
{
	notify("Criterion_Normal constructor.");
	_model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	if(da->getNoOfClasses()!=2) throw fst_error(std::string(criterion_name())+"() defined for two-class problems only.");
	notify("Criterion_Normal::initialize().");
	assert(_model);
	_model->learn(da);

	alloc_vector(_meandif);
	_meandif->copy_raw(_model->get_mean()[0]);
	_meandif->subtract_raw(_model->get_mean()[1]);

	allocate_buffers();
	update_derived_model();
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	notify("Criterion_Normal::evaluate().");
	assert(_model);
	assert(get_n()>0);
	assert(_meandif);
	assert(get_n()==_meandif->get_n());
	assert(sub);

	if(sub->get_d_raw()==0) return false;

	_model->narrow_to(sub);
	_meandif->narrow_to(sub);
	if(!evaluate_narrowed(result,sub)) return false;
	// A non-finite value (NaN or an infinity) means the normal-model arithmetic broke down
	// on this subset -- an inverse or determinant of a numerically singular covariance. It
	// must never reach a search: NaN compares false against everything, so a search would
	// silently accept or reject candidates on garbage and still report success.
	if(!std::isfinite(result))
		throw FST::fst_error(std::string(criterion_name())+"::evaluate: non-finite criterion value on the evaluated subset (numerically singular class covariance; reduce the subset size, drop constant or collinear features, or use more training data)");
	return true;
}


} // namespace
#endif // FSTCRITERIONNORMAL_H ///:~
