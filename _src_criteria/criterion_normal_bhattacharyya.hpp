#ifndef FSTCRITERIONNORMALBHATTACHARYYA_H
#define FSTCRITERIONNORMALBHATTACHARYYA_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_normal_bhattacharyya.hpp
   \brief   Implements Bhattacharyya distance based on normal (gaussian) model to serve as feature selection criterion
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

/*! \brief Implements Bhattacharyya distance based on normal (gaussian) model to serve as feature selection criterion

    For the two classes modelled as Gaussians (mu0,S0) and (mu1,S1) restricted to the
    d selected features, with m = mu0-mu1 and S = (S0+S1)/2, the value returned is

        B = 1/8 * m^T S^-1 m  +  1/2 * ln( det(S) / sqrt(det(S0)*det(S1)) )

    i.e. the standard Bhattacharyya distance between two normal distributions: a
    mean-separation term plus a covariance-dissimilarity term. Higher is better, the
    value is non-negative and grows with subset size (monotonic w.r.t. cardinality,
    hence usable with Branch & Bound). evaluate_narrowed() computes the first term
    equivalently as 0.25 * m^T (S0+S1)^-1 m, which saves halving the summed covariance
    for the inversion.

    \note Defined for two-class problems only
    \note Requires non-singular per-class covariances on the evaluated subset - see the
    remark in evaluate_narrowed() */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal_Bhattacharyya : public Clonable_Impl<Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< base class: the normal-model template-method skeleton
	typedef Clonable_Impl<Criterion_Normal_Bhattacharyya,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Normal_Bhattacharyya() {notify("Criterion_Normal_Bhattacharyya constructor.");}
	virtual ~Criterion_Normal_Bhattacharyya() {notify("Criterion_Normal_Bhattacharyya destructor.");}
protected:
	// Template-method hooks: initialize()/evaluate() skeletons live in Criterion_Normal
	virtual const char* criterion_name() const override {return "Criterion_Normal_Bhattacharyya";}
	virtual void allocate_buffers() override {
		parent::alloc_matrix(_cov12);
		parent::alloc_matrix(_cov12_div2);
		parent::alloc_matrix(_invbuf);
		parent::alloc_matrix(_LUtemp,true);
	}
	virtual void update_derived_model() override {
		_cov12->copy_raw(parent::_model->get_cov()[0]);
		_cov12->add_raw(parent::_model->get_cov()[1]);
		_cov12_div2->copy_constmul_raw(*_cov12,0.5);
	}
	virtual bool evaluate_narrowed(RETURNTYPE &result, const PSubset sub) override;
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; deep-copies the four criterion-specific work buffers (base _model/_meandif copied by parent_impl).
	Criterion_Normal_Bhattacharyya(const Criterion_Normal_Bhattacharyya& cnb);
private:
	std::unique_ptr<Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> > _cov12; //!< (cov[0]+cov[1]), computed by update_derived_model()
	std::unique_ptr<Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> > _cov12_div2; //!< _cov12/2, the pooled covariance whose determinant enters the Bhattacharyya distance
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf; //!< inverse of the narrowed _cov12, used in the quadratic-form term
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; //!< scratch LU-decomposition storage, reused for each matrix inversion/determinant
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal_Bhattacharyya(const Criterion_Normal_Bhattacharyya& cnb) :
	parent_impl(cnb)
{
	notify("Criterion_Normal_Bhattacharyya copy-constructor.");
	if(cnb._cov12) _cov12.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnb._cov12));
	if(cnb._cov12_div2) _cov12_div2.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnb._cov12_div2));
	if(cnb._invbuf) _invbuf.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnb._invbuf));
	if(cnb._LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnb._LUtemp));
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate_narrowed(RETURNTYPE &result, const PSubset sub)
{ // (model and _meandif have been narrowed to sub by Criterion_Normal::evaluate)
	notify("Criterion_Normal_Bhattacharyya::evaluate_narrowed().");
	assert(_cov12);
	assert(_cov12_div2);
	assert(_invbuf);
	assert(_LUtemp);
	assert(parent::get_n()==_cov12->get_n());
	assert(parent::get_n()==_cov12_div2->get_n());
	assert(parent::get_n()<=_invbuf->get_n_max());
	assert(parent::get_n()<=_LUtemp->get_n_max());

	RETURNTYPE logdet1,logdet2,logdet3;

	_cov12->narrow_to(sub);
	_cov12_div2->narrow_to(sub);

	_invbuf->redim(parent::get_d());
	_LUtemp->redim(parent::get_d());

	// The covariance term ln( |S| / sqrt(|S0| |S1|) ) is formed in the LOG domain from
	// log_determinant(): the three determinants themselves are products of d LU pivots and
	// leave the double range (overflow to inf, or underflow to 0 and a bogus "singular"
	// report) long before the matrices become ill-conditioned -- a 100-feature subset of
	// features with variance 1e8 has |S| ~ 1e800. The log form depends only on the geometry,
	// not on the feature scale, and is exact where the plain product would still be finite.
	_cov12->LUdecompose(*_LUtemp);
	_cov12->invert(*_invbuf,*_LUtemp);
	parent::_model->get_cov()[0].LUdecompose(*_LUtemp);
	logdet1=parent::_model->get_cov()[0].log_determinant(*_LUtemp);
	parent::_model->get_cov()[1].LUdecompose(*_LUtemp);
	logdet2=parent::_model->get_cov()[1].log_determinant(*_LUtemp);
	_cov12_div2->LUdecompose(*_LUtemp);
	logdet3=_cov12_div2->log_determinant(*_LUtemp);

	// LIMITATION: a singular class covariance on the evaluated subset makes the criterion
	// undefined and is reported by throwing, not by returning false. It happens when the
	// subset contains constant, duplicate or linearly dependent features, or when d exceeds
	// the number of training samples of a class. Remedies: reduce the target subset size,
	// drop the constant/collinear features, use more training data, or pick a criterion that
	// does not invert per-class covariances (e.g. a wrapper criterion). A zero LU pivot gives
	// ln 0 = -inf here (LUdecompose() itself throws on an all-zero row).
	if(!std::isfinite(logdet1) || !std::isfinite(logdet2))
		throw FST::fst_error("Criterion_Normal_Bhattacharyya::evaluate: singular class covariance matrix on the evaluated subset (zero determinant)");
	RETURNTYPE right=0.5*(logdet3-0.5*(logdet1+logdet2));
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
	result=0.25*left+right;
	return true;
}

} // namespace
#endif // FSTCRITERIONNORMALBHATTACHARYYA_H ///:~
