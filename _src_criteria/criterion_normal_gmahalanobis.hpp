#ifndef FSTCRITERIONNORMALGMAHALANOBIS_H
#define FSTCRITERIONNORMALGMAHALANOBIS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_normal_gmahalanobis.hpp
   \brief   Implements Generalized Mahalanobis distance based on normal (gaussian) model to serve as feature selection criterion
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
#include "criterion_normal.hpp"
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

/*! \brief Implements Generalized Mahalanobis distance based on normal (gaussian) model to serve as feature selection criterion

    For the two classes modelled as Gaussians (mu0,S0) and (mu1,S1) restricted to the
    d selected features, with m = mu0-mu1 and P0, P1 the class priors, the value
    returned is the squared Mahalanobis distance of the class means in the metric of
    the prior-weighted pooled covariance:

        M = m^T (P0*S0 + P1*S1)^-1 m

    It is the mean-separation term alone - unlike Bhattacharyya and Divergence it
    ignores any difference between the two covariances, which makes it the cheapest of
    the three (one inversion, no determinants) and blind to classes that differ in
    shape but not in location. Higher is better, the value is non-negative and
    cardinality-monotonic, hence usable with Branch & Bound.

    \note Defined for two-class problems only
    \note Requires a non-singular pooled covariance on the evaluated subset - constant,
    duplicate or linearly dependent features, or a subset larger than the training
    sample count, make the criterion undefined. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal_GMahalanobis : public Clonable_Impl<Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< base class: the normal-model template-method skeleton
	typedef Clonable_Impl<Criterion_Normal_GMahalanobis,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Normal_GMahalanobis() {notify("Criterion_Normal_GMahalanobis constructor.");}
	virtual ~Criterion_Normal_GMahalanobis() {notify("Criterion_Normal_GMahalanobis destructor.");}


protected:
	// Template-method hooks: initialize()/evaluate() skeletons live in Criterion_Normal
	virtual const char* criterion_name() const override {return "Criterion_Normal_GMahalanobis";}
	virtual void allocate_buffers() override {
		parent::alloc_matrix(_cov12);
		parent::alloc_matrix(_invbuf);
		parent::alloc_matrix(_LUtemp,true);
	}
	virtual void update_derived_model() override {
		_cov12->set_all_raw_to(0.0);
		_cov12->add_constmul_raw(parent::_model->get_cov()[0],parent::_model->get_Pc(0));
		_cov12->add_constmul_raw(parent::_model->get_cov()[1],parent::_model->get_Pc(1));
	}
	virtual bool evaluate_narrowed(RETURNTYPE &result, const PSubset sub) override;
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; deep-copies the three criterion-specific work buffers (base _model/_meandif copied by parent_impl).
	Criterion_Normal_GMahalanobis(const Criterion_Normal_GMahalanobis& cngm);
private:
	std::unique_ptr<Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> > _cov12; //!< prior-weighted pooled covariance Pc(0)*cov[0]+Pc(1)*cov[1], computed by update_derived_model()
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf; //!< inverse of the narrowed _cov12, used in the quadratic-form term
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; //!< scratch LU-decomposition storage, reused for each matrix inversion
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal_GMahalanobis(const Criterion_Normal_GMahalanobis& cngm) :
	parent_impl(cngm)
{
	notify("Criterion_Normal_GMahalanobis copy-constructor.");
	if(cngm._cov12) _cov12.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cngm._cov12));
	if(cngm._invbuf) _invbuf.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cngm._invbuf));
	if(cngm._LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cngm._LUtemp));
}



template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate_narrowed(RETURNTYPE &result, const PSubset sub)
{ // (model and _meandif have been narrowed to sub by Criterion_Normal::evaluate)
	notify("Criterion_Normal_GMahalanobis::evaluate_narrowed().");
	assert(_cov12);
	assert(_invbuf);
	assert(_LUtemp);
	assert(parent::get_n()==_cov12->get_n());
	assert(parent::get_n()<=_invbuf->get_n_max());
	assert(parent::get_n()<=_LUtemp->get_n_max());

	_cov12->narrow_to(sub);

	_invbuf->redim(parent::get_d());
	_LUtemp->redim(parent::get_d());

	_cov12->LUdecompose(*_LUtemp);
	_cov12->invert(*_invbuf,*_LUtemp);

	RETURNTYPE left=0.0;
	RETURNTYPE pom;

	for(DIMTYPE j=0;j<parent::get_d();j++)
	{
		pom=0.0;
		for(DIMTYPE i=0;i<parent::get_d();i++)
		{
			pom+=parent::_meandif->at(i)*_invbuf->at_raw(i,j);
		}
		left+=pom*parent::_meandif->at(j);
	}
	result=left;
	return true;
}


} // namespace
#endif // FSTCRITERIONNORMALGMAHALANOBIS_H ///:~
