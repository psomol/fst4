#ifndef FSTREGRESSORRIDGE_H
#define FSTREGRESSORRIDGE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    regressor_ridge.hpp
   \brief   Defines Regressor_Ridge — closed-form L2-regularized linear regression for wrapper-based feature selection
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0
   \note    FST4 was developed using MinGW-W64 g++ 15.2 and requires
   \note    \li a C++17 conforming compiler and standard library (no Boost, unlike FST3),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/,
                version 3.37; not distributed with FST4 — drop svm.cpp/svm.h into ./libsvm/, see libsvm/readme.txt)
   \note    Note that LibSVM is required for SVM related tools only.

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
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"
#include "regressor.hpp"
#include "dense_train_tools.hpp"

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

/*! \brief Closed-form ridge regression (L2-regularized least squares) for
    wrapper-based feature selection.

    Solves the L2-regularized normal equations by Cholesky factorization on the
    internally standardized design (per-split feature mean/scale, like the
    logistic-regression classifier; the bias coordinate is unpenalized).
    Entirely deterministic and RNG-free - like logreg, the result depends on
    neither the seed nor the thread count.

    NOTE on the scale of lambda: the accumulated normal equations are divided by
    the training pattern count N BEFORE the penalty is added, so the system
    actually solved is

        (Z^T Z / N + lambda*I) w = Z^T y / N,

    equivalent to the textbook (Z^T Z + N*lambda*I) w = Z^T y. lambda is
    therefore a PER-SAMPLE penalty: its useful magnitude does not change when the
    training set grows, unlike the textbook parameter, but it is not numerically
    interchangeable with one quoted for the unscaled formulation.

    Dependent-variable convention MATCHES Regressor_Weighted_Neighbours (the
    wrapper-regressor pipeline): set_dependent_index(idx) selects a data
    column - which is then EXCLUDED from the model's features, so a subset
    containing it is fitted with one feature fewer than it holds; any
    out-of-range index (the (DIMTYPE)-1 sentinel, which is the default) selects
    the class label - "regression simulating classification". test() likewise
    mirrors the weighted-neighbours conventions (label mode: fraction of
    predictions within 0.5 of the label; column mode: negated mean absolute
    error), so the two regressors' test() values are directly comparable; for
    feature selection the regression metrics (R^2, negated RMSE/MAE) offered
    through test_metric() are the more natural choice, and the fst4 CLI defaults
    this regressor to R^2. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Regressor_Ridge : public Clonable_Impl<Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Regressor<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Regressor<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Regressor_Ridge,parent> parent_impl; //!< implements clone() via the (implicit, member-wise) copy-constructor - all members are value-semantic
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor; //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to the feature subset being trained/evaluated on
	typedef typename DATAACCESSOR::PPattern PPattern; //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Regressor_Ridge() : _lambda(0.01), _dep_var_idx((DIMTYPE)-1), _features(0), _subfeatures(0) {notify("Regressor_Ridge constructor.");}
	virtual ~Regressor_Ridge() {notify("Regressor_Ridge destructor.");}

	void set_lambda(const REALTYPE l) {assert(l>=0.0); _lambda=l;} //!< sets the L2 penalty weight applied to the (unpenalized-bias) normal equations; default 0.01. Per-sample units -- see the class documentation
	//! \brief Chooses the dependent variable: a raw data column index, or any out-of-range value
	//! (the (DIMTYPE)-1 default) to predict the class label instead.
	//! \warning A chosen column is REMOVED from the model's features - a feature must not help
	//! predict itself. A d-feature subset containing it is therefore fitted on d-1 features, and
	//! a subset consisting of it alone makes train() fail.
	void set_dependent_index(const DIMTYPE depvaridx) {_dep_var_idx=depvaridx;}

	void initialize(const PDataAccessor da); //!< records data shape, allocates the index buffer

	//! evaluates the trained linear model (standardized weights + bias) at \p pattern. Requires a
	//! successful train(): with assertions disabled an untrained model reads empty weight vectors.
	virtual bool predict(RETURNTYPE &depvar, const PPattern &pattern) override;
	//! Fits \p sub's features (excluding the dependent column, if selected) by Cholesky-solving the
	//! L2-regularized normal equations on the internally standardized design; returns false if the
	//! subset holds only the dependent column, or if the Gram-plus-ridge matrix is not solvable.
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override; // the measure Regressor_Weighted_Neighbours::test() also reports (see the class documentation)
	virtual bool test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec) override; // r2/neg-rmse/neg-mae

	virtual std::ostream& print(std::ostream& os) const override {os << "Regressor_Ridge(lambda=" << _lambda << ")"; return os;}
protected:
	REALTYPE _lambda; //!< L2 penalty weight; set by set_lambda()
	DIMTYPE _dep_var_idx; //!< dependent-variable data column, or an out-of-range sentinel ((DIMTYPE)-1 by default) meaning "use the class label"; set by set_dependent_index()

	DIMTYPE _features; //!< raw feature count recorded by initialize(), for sizing _index and for dependent_from()'s column-vs-label test
	DIMTYPE _subfeatures;          //!< d = features used by the current model (dependent column excluded)
	std::vector<DIMTYPE> _index;   //!< subset position -> raw feature index (first d entries valid)
	std::vector<REALTYPE> _mean;   //!< standardization shift per subset position
	std::vector<REALTYPE> _scale;  //!< standardization scale per subset position (1 for constant features)
	std::vector<REALTYPE> _w;      //!< d weights + bias at [d]

	// training scratch (copied by clones harmlessly)
	std::vector<REALTYPE> _A;      //!< (d+1)x(d+1) normal-equation matrix, row-major
	std::vector<REALTYPE> _b;      //!< d+1 right-hand side
	std::vector<REALTYPE> _Z;      //!< N x (d+1) standardized design
	std::vector<REALTYPE> _yv;     //!< N dependent values

	//! Resolves the dependent value for one pattern: \p pat[_dep_var_idx] when _dep_var_idx names a
	//! valid raw feature column, otherwise \p cls (the "regression simulating classification" convention
	//! shared with Regressor_Weighted_Neighbours). It cannot fail; the bool return exists only so a
	//! guarded failure mode could be added without changing the call sites.
	bool dependent_from(const PPattern &pat, const DIMTYPE cls, REALTYPE &dep) const
	{
		if(_dep_var_idx<_features) dep=(REALTYPE)pat[_dep_var_idx];
		else dep=(REALTYPE)cls; // sentinel: class label as dependent variable (the wrapper-regressor pipeline convention)
		return true;
	}
	// Cholesky factor/solve of the normal equations live in dense_train_tools.hpp (shared with classifier_logreg)
	bool collect_predictions(const PDataAccessor da, std::vector<std::pair<RETURNTYPE,RETURNTYPE>> &pairs); //!< the shared test-part (dependent,prediction) enumeration behind test() and test_metric()
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Regressor_Ridge::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Regressor_Ridge::train().");
	assert(da);
	assert(sub);
	assert(_features==da->getNoOfFeatures());
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	// buffer the feature subset index, excluding the dependent column if it is one
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {
		assert(_subfeatures<_features);
		if(fsel==_dep_var_idx) continue;
		_index[_subfeatures++]=fsel;
	}
	if(_subfeatures==0) return false; // subset held only the dependent column

	const std::size_t d=_subfeatures, dp1=d+1;

	// pass 1 + standardize via the shared helper (dense_train_tools.hpp); the appended
	// bias coordinate and the dependent value ride along in the per-pattern callback
	_yv.clear();
	const std::size_t N=standardize_design<REALTYPE,IDXTYPE>(da,da->getNoOfClasses(),d,dp1,_index,_Z,_mean,_scale,
		[this](const auto *pat, const DIMTYPE c){_Z.push_back(1.0); REALTYPE dep; dependent_from(pat,c,dep); _yv.push_back(dep);});
	if(N==0) return false;

	// normal equations: A = Z^T Z / N + diag(lambda,...,lambda,epsilon); b = Z^T y / N
	// NOTE the division by N happens BEFORE lambda is added, so lambda is a per-sample penalty
	// (equivalent to N*lambda in the unscaled textbook formulation) - see the class documentation
	_A.assign(dp1*dp1,0.0);
	_b.assign(dp1,0.0);
	for(std::size_t n=0;n<N;n++)
	{
		const REALTYPE *z=&_Z[n*dp1];
		const REALTYPE y=_yv[n];
		for(std::size_t f1=0;f1<dp1;f1++)
		{
			_b[f1]+=z[f1]*y;
			for(std::size_t f2=0;f2<=f1;f2++) _A[f1*dp1+f2]+=z[f1]*z[f2]; // lower triangle
		}
	}
	for(std::size_t j=0;j<dp1*dp1;j++) _A[j]/=(REALTYPE)N;
	for(std::size_t f=0;f<dp1;f++) _b[f]/=(REALTYPE)N;
	for(std::size_t f=0;f<d;f++) _A[f*dp1+f]+=_lambda;
	// bias unpenalized; the tiny ridge only keeps Cholesky defined on degenerate data. Its size
	// is arbitrary within reason - small enough not to shrink the bias measurably, large enough
	// to dominate rounding noise in the scaled Gram matrix.
	_A[d*dp1+d]+=(REALTYPE)1e-10;

	// in-place lower-Cholesky of _A (SPD by construction: Gram + ridge), then two triangular solves into _w
	if(!cholesky_factor(_A,dp1)) return false;
	_w.assign(dp1,0.0);
	cholesky_solve(_A,dp1,_b.data(),false,_w.data());
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::predict(RETURNTYPE &depvar, const PPattern &pattern)
{
	assert(_subfeatures>0); // train() must have succeeded
	const std::size_t d=_subfeatures;
	assert(_w.size()==d+1);
	REALTYPE v=_w[d]; // bias
	for(std::size_t f=0;f<d;f++) v+=_w[f]*(((REALTYPE)pattern[_index[f]]-_mean[f])/_scale[f]);
	depvar=(RETURNTYPE)v;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// mirrors Regressor_Weighted_Neighbours::test(), so the two regressors report the same measure:
	// label mode = fraction of predictions within 0.5 of the class label ("simulated classification"),
	// column mode = mean of negated absolute prediction errors (FST maximizes)
	notify("Regressor_Ridge::test().");
	std::vector<std::pair<RETURNTYPE,RETURNTYPE>> pairs;
	if(!collect_predictions(da,pairs)) return false;
	if(pairs.empty()) return false;
	RETURNTYPE errsum=0;
	const bool label_mode=!(_dep_var_idx<_features);
	for(const auto &pr : pairs)
	{
		if(label_mode) {if(std::fabs((REALTYPE)pr.first-(REALTYPE)pr.second)<0.5) errsum+=1;}
		else errsum+=-std::fabs((REALTYPE)pr.first-(REALTYPE)pr.second);
	}
	result=errsum/(RETURNTYPE)pairs.size();
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::collect_predictions(const PDataAccessor da, std::vector<std::pair<RETURNTYPE,RETURNTYPE>> &pairs)
{
	assert(da);
	assert(_subfeatures>0);
	typename DATAACCESSOR::PPattern p1;
	IDXTYPE s1,i1;
	const DIMTYPE _feats=da->getNoOfFeatures();
	RETURNTYPE predval;
	pairs.clear();
	for(DIMTYPE c_test=0;c_test<da->getNoOfClasses();c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p1=blk.data; s1=blk.size;
			for(i1=0;i1<s1;i1++)
			{
				REALTYPE depval;
				dependent_from(&p1[i1*_feats],c_test,depval);
				if(!predict(predval,&p1[i1*_feats])) return false;
				pairs.push_back(std::make_pair((RETURNTYPE)depval,predval));
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as a training accessor
		}
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Regressor_Ridge<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec)
{
	if(spec.is_default()) return test(result,da); // the default metric IS test()'s own measure
	if(!spec.is_regression_metric()) return false;
	std::vector<std::pair<RETURNTYPE,RETURNTYPE>> pairs;
	if(!collect_predictions(da,pairs)) return false;
	return metric_from_predictions<RETURNTYPE>(result,spec.metric,pairs);
}

} // namespace
#endif // FSTREGRESSORRIDGE_H ///:~
