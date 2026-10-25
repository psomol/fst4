#ifndef FSTCRITERIONNORMALDIVERGENCE_H
#define FSTCRITERIONNORMALDIVERGENCE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_normal_divergence.hpp
   \brief   Implements Divergence (distance) based on normal (gaussian) model to serve as feature selection criterion
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

/*! \brief Implements Divergence (distance) based on normal (gaussian) model to serve as feature selection criterion

    The symmetric Kullback divergence between the two classes modelled as Gaussians
    (mu0,S0) and (mu1,S1) restricted to the d selected features, with m = mu0-mu1:

        J = 1/2 * m^T (S0^-1 + S1^-1) m  +  1/2 * ( tr(S0^-1 S1 + S1^-1 S0) - 2d )

    The first term measures mean separation in the metric of both inverse covariances,
    the second the dissimilarity of the covariances themselves. The -2d offset makes the
    second term vanish for equal covariances: S0 == S1 gives S0^-1 S1 = S1^-1 S0 = I,
    whose two traces add up to exactly 2d. Higher is better, and the criterion is
    non-negative.

    \note Defined for two-class problems only
    \warning Divergence is cardinality-monotonic in theory, so is_monotonic() reports true
    and Branch & Bound accepts it, but the two matrix inversions per evaluation make it
    numerically fragile: on ill-conditioned data the computed values can breach
    monotonicity and Branch & Bound may then miss the optimum. Prefer
    Criterion_Normal_Bhattacharyya (one inversion) whenever an optimal search is used.
    \note Requires non-singular per-class covariances on the evaluated subset - a subset
    of constant, duplicate or linearly dependent features, or one larger than a class's
    training sample count, makes the criterion undefined. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal_Divergence : public Clonable_Impl<Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< base class: the normal-model template-method skeleton
	typedef Clonable_Impl<Criterion_Normal_Divergence,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Normal_Divergence() {notify("Criterion_Normal_Divergence constructor.");}
	virtual ~Criterion_Normal_Divergence() {notify("Criterion_Normal_Divergence destructor.");}


protected:
	// Template-method hooks: initialize()/evaluate() skeletons live in Criterion_Normal
	virtual const char* criterion_name() const override {return "Criterion_Normal_Divergence";}
	virtual void allocate_buffers() override {
		parent::alloc_vector(_diag);
		parent::alloc_matrix(_invbuf1);
		parent::alloc_matrix(_invbuf2);
		parent::alloc_matrix(_LUtemp,true);
	}
	virtual void update_derived_model() override {} // no criterion-specific model derivations beyond the shared _meandif
	virtual bool evaluate_narrowed(RETURNTYPE &result, const PSubset sub) override;
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; deep-copies the four criterion-specific work buffers (base _model/_meandif copied by parent_impl).
	Criterion_Normal_Divergence(const Criterion_Normal_Divergence& cnd);
private:
	std::unique_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _diag; //!< per-feature diagonal of cov1^-1*cov2 + cov2^-1*cov1, summed into the trace term of the divergence
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf1; //!< inverse of the narrowed class-1 covariance
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf2; //!< inverse of the narrowed class-2 covariance
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; //!< scratch LU-decomposition storage, reused for each matrix inversion
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal_Divergence(const Criterion_Normal_Divergence& cnd) :
	parent_impl(cnd)
{
	notify("Criterion_Normal_Divergence copy-constructor.");
	if(cnd._diag) _diag.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*cnd._diag));
	if(cnd._invbuf1) _invbuf1.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnd._invbuf1));
	if(cnd._invbuf2) _invbuf2.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnd._invbuf2));
	if(cnd._LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnd._LUtemp));
}



template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate_narrowed(RETURNTYPE &result, const PSubset sub)
{ // (model and _meandif have been narrowed to sub by Criterion_Normal::evaluate)
	notify("Criterion_Normal_Divergence::evaluate_narrowed().");
	assert(_diag);
	assert(_invbuf1);
	assert(_invbuf2);
	assert(_LUtemp);
	assert(parent::get_n()==_diag->get_n());
	assert(parent::get_n()<=_invbuf1->get_n_max());
	assert(parent::get_n()<=_invbuf2->get_n_max());
	assert(parent::get_n()<=_LUtemp->get_n_max());

	_invbuf1->redim(parent::get_d());
	_invbuf2->redim(parent::get_d());
	_LUtemp->redim(parent::get_d());

	Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> &_cov1=(parent::_model->get_cov()[0]);
	Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> &_cov2=(parent::_model->get_cov()[1]);

	_cov1.LUdecompose(*_LUtemp);
	_cov1.invert(*_invbuf1,*_LUtemp);	
	_cov2.LUdecompose(*_LUtemp);
	_cov2.invert(*_invbuf2,*_LUtemp);

	RETURNTYPE left=0.0;
	RETURNTYPE pom;
	for(DIMTYPE j=0;j<parent::get_d();j++)
	{
		pom=0.0;
		for(DIMTYPE i=0;i<parent::get_d();i++)
		{
			pom+=parent::_meandif->at(i)*(_invbuf1->at_raw(i,j)+_invbuf2->at_raw(i,j));
		}
		left+=pom*parent::_meandif->at(j);
	}

	RETURNTYPE right=0.0;
	for(DIMTYPE i=0;i<parent::get_d();i++)
	{
		_diag->at_raw(i)=0.0;
		for(DIMTYPE k=0;k<parent::get_d();k++)
		{
			_diag->at_raw(i)+=_invbuf1->at_raw(i,k)*_cov2.at(k,i);
		}
	}
	for(DIMTYPE i=0;i<parent::get_d();i++)
	{
		for(DIMTYPE k=0;k<parent::get_d();k++)
		{
			_diag->at_raw(i)+=_invbuf2->at_raw(i,k)*_cov1.at(k,i);
		}
	}
	// trace of (cov1^-1*cov2 + cov2^-1*cov1) minus 2d - the offset that makes the term
	// vanish when the two covariances coincide (see the class documentation)
	for(DIMTYPE i=0;i<parent::get_d();i++) right+=_diag->at_raw(i)-2.0;

	result=0.5*left+0.5*right;
	return true;
}

} // namespace
#endif // FSTCRITERIONNORMALDIVERGENCE_H ///:~
