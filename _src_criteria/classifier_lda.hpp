#ifndef FSTCLASSIFIERLDA_H
#define FSTCLASSIFIERLDA_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_lda.hpp
   \brief   Defines Classifier_LDA — shared-covariance Gaussian (linear) discriminant for wrapper-based feature selection
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
#include <cstring>
#include <cassert>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"
#include "classifier.hpp"
#include "model_normal.hpp"
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

/*! \brief Linear (Fisher/Gaussian shared-covariance) discriminant classifier.

    The pooled within-class covariance (per-class Model_Normal covariances
    weighted by the class priors) yields the closed-form linear discriminants
    delta_c(x) = a_c^T x + b_c with a_c = S_pooled^{-1} mu_c and
    b_c = -1/2 mu_c^T a_c + log P(c). Deterministic and RNG-free; the single
    matrix inverse goes through the same Indexed_Matrix LU machinery as the
    normal-model filters and the normal Bayes classifier — including the
    optional Eigen-backed path (FST4_USE_EIGEN), which stays in parity.

    Complements gmahalanobis (a two-class filter) and Classifier_Normal_Bayes
    (per-class covariances, QDA-like): one shared covariance = linear
    boundaries, K classes supported, roughly K times cheaper per training
    than the Bayes classifier at equal subset size (one inverse, not K).
    classify_scores() exposes the discriminants (log-posteriors up to a
    shared shift; computed in centered form relative to the prior-weighted
    mean of the class means, so a large common feature offset costs no
    digits) for the ranking metrics (AUC); they are uncalibrated -
    monotone in the posterior, but neither probabilities nor comparable
    across trainings. Like the other normal-model components, it fails on
    singular pooled covariance (e.g. constant features or subset size
    exceeding the sample count). */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_LDA : public Clonable_Impl<Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the abstract base type
	typedef Clonable_Impl<Classifier_LDA,parent> parent_impl; //!< implements clone() via the copy-constructor below
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being trained/classified on
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_LDA()
	{
		notify("Classifier_LDA constructor.");
		_model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
		_classes=0; _features=0; _subfeatures=0;
	}
	virtual ~Classifier_LDA() {notify("Classifier_LDA destructor.");}

	//! \brief Pre-allocates model and matrix buffers and records the class/feature counts of \p da.
	//! Must be called before the first train(); throws fst_error when \p da has fewer than 2 classes.
	void initialize(const PDataAccessor da);

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // argmax of the linear discriminants; first maximum wins
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override; // per-class discriminant values for ranking metrics (AUC)
	/*! \brief Learns per-class Gaussian parameters on the designated training part, pools the
	    per-class covariances (weighted by the class priors) into one within-class covariance,
	    inverts it once through the shared Indexed_Matrix LU machinery, and derives the
	    closed-form linear discriminants a_c, b_c (see class doc).
	    \pre initialize() has been called for a data accessor with the same class and
	    feature counts as \p da.
	    \return false if the pooled covariance is singular (any resulting discriminant
	    coefficient is non-finite); true otherwise. */
	virtual bool train(const PDataAccessor da, const PSubset sub) override; // learns per-class model, pools covariance, inverts once
	// test() = the canonical accuracy loop inherited from Classifier, driven by classify() above

	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_LDA()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! \brief Deep-copies the learned model and the pooled-covariance matrices (LU scratch,
	//! pooled covariance, its inverse); the discriminant vectors _a/_b and _index copy by value.
	Classifier_LDA(const Classifier_LDA& cld) :
		_classes(cld._classes),
		_features(cld._features),
		_subfeatures(cld._subfeatures),
		_index(cld._index),
		_a(cld._a),
		_b(cld._b),
		_m(cld._m)
	{
		notify("Classifier_LDA copy-constructor.");
		if(cld._model) _model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cld._model));
		if(cld._LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cld._LUtemp));
		if(cld._pooled) _pooled.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cld._pooled));
		if(cld._pooledinv) _pooledinv.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cld._pooledinv));
	}
protected:
	std::unique_ptr<Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model; //!< per-class means/covariances (learned by train(), see Model_Normal)
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp;    //!< LU scratch (lazy allocation)
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _pooled;    //!< pooled within-class covariance
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _pooledinv; //!< its inverse

	DIMTYPE _classes;                  //!< number of classes, as recorded by initialize()
	DIMTYPE _features;                 //!< raw (full) feature count, as recorded by initialize()
	DIMTYPE _subfeatures;              //!< d = current subset size
	std::vector<DIMTYPE> _index;       //!< subset position -> raw feature index (first d entries valid)
	std::vector<RETURNTYPE> _cs_sc;    //!< classify() score scratch (member to avoid per-pattern allocation)
	std::vector<REALTYPE> _a;          //!< classes x d discriminant directions, row-major
	std::vector<REALTYPE> _b;          //!< classes discriminant offsets
	std::vector<REALTYPE> _m;          //!< d-entry centering point of the discriminants: the prior-weighted mean of the class means (see train())
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_LDA::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_LDA requires at least 2 classes");
	if(!_LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(da->getNoOfFeatures(),true));
	if(da->getNoOfFeatures()>_LUtemp->get_n_max()) _LUtemp->reset(da->getNoOfFeatures(),true);
	if(!_pooled) _pooled.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(da->getNoOfFeatures(),false));
	if(da->getNoOfFeatures()>_pooled->get_n_max()) _pooled->reset(da->getNoOfFeatures(),false);
	if(!_pooledinv) _pooledinv.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(da->getNoOfFeatures(),false));
	if(da->getNoOfFeatures()>_pooledinv->get_n_max()) _pooledinv->reset(da->getNoOfFeatures(),false);
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_LDA::train().");
	assert(da);
	assert(sub);
	assert(_model);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	// buffer the feature subset index
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<_features); _index[_subfeatures++]=fsel;}
	assert(_subfeatures==sub->get_d_raw());
	assert(_subfeatures>0);
	const DIMTYPE d=_subfeatures;

	// per-class means/covariances on the designated training part (subset dimensionality)
	_model->learn(da,sub);
	_model->denarrow();

	// pooled within-class covariance: class covariances weighted by the priors
	_pooled->redim(d);
	for(DIMTYPE f1=0;f1<d;f1++)
		for(DIMTYPE f2=0;f2<d;f2++)
		{
			REALTYPE v=0.0;
			for(DIMTYPE c=0;c<_classes;c++) v+=_model->get_Pc(c)*_model->get_cov()[c].at_raw(f1,f2);
			_pooled->at_raw(f1,f2)=v;
		}

	// single inverse through the shared Indexed_Matrix LU machinery (Eigen-path parity)
	_pooledinv->redim(d);
	_LUtemp->redim(d);
	_pooled->LUdecompose(*_LUtemp);
	_pooled->invert(*_pooledinv,*_LUtemp);

	// closed-form discriminants in CENTERED form. With m the prior-weighted mean of the class
	// means and delta_c = mu_c - m, the LDA score  mu_c'S^-1 x - 1/2 mu_c'S^-1 mu_c + ln P(c)
	// equals  delta_c'S^-1 (x-m) - 1/2 delta_c'S^-1 delta_c + ln P(c)  plus a term that does not
	// depend on the class, so a_c = S^-1 delta_c and b_c = -1/2 delta_c.a_c + ln P(c), applied to
	// x-m, yield the same decisions and the same per-sample score differences. The uncentered
	// form builds O(1) class differences out of O(|mean|^2/var) terms and loses digits as soon
	// as the features carry a large common offset (with an offset of 1e8 it dropped from 0.73 to
	// 0.60 accuracy on data the centered form classifies exactly as without the offset).
	_m.assign(d,0.0);
	for(DIMTYPE c=0;c<_classes;c++)
	{
		const auto &mu=_model->get_mean()[c];
		const REALTYPE pc=_model->get_Pc(c);
		for(DIMTYPE f1=0;f1<d;f1++) _m[f1]+=pc*mu.at_raw(f1); // the priors sum to 1
	}
	_a.assign((std::size_t)_classes*d,0.0);
	_b.assign(_classes,0.0);
	for(DIMTYPE c=0;c<_classes;c++)
	{
		REALTYPE *ac=&_a[(std::size_t)c*d];
		auto &mu=_model->get_mean()[c];
		for(DIMTYPE f1=0;f1<d;f1++)
		{
			REALTYPE v=0.0;
			for(DIMTYPE f2=0;f2<d;f2++) v+=_pooledinv->at_raw(f1,f2)*(mu.at_raw(f2)-_m[f2]);
			ac[f1]=v;
		}
		REALTYPE dot=0.0;
		for(DIMTYPE f1=0;f1<d;f1++) dot+=ac[f1]*(mu.at_raw(f1)-_m[f1]);
		_b[c]=-0.5*dot+std::log(_model->get_Pc(c));
	}
	// singular pooled covariance produces non-finite coefficients — fail loudly like the normal-model filters
	for(std::size_t j=0;j<_a.size();j++) if(!std::isfinite(_a[j])) return false;
	for(DIMTYPE c=0;c<_classes;c++) if(!std::isfinite(_b[c])) return false;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	assert(_subfeatures>0); // train() must have succeeded
	const DIMTYPE d=_subfeatures;
	scores.resize(_classes);
	for(DIMTYPE c=0;c<_classes;c++)
	{
		const REALTYPE *ac=&_a[(std::size_t)c*d];
		REALTYPE v=_b[c];
		for(DIMTYPE f=0;f<d;f++) v+=ac[f]*((REALTYPE)pattern[_index[f]]-_m[f]); // centered, see train()
		scores[c]=(RETURNTYPE)v;
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LDA<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	if(!classify_scores(_cs_sc,pattern)) return false; // member scratch: avoids a per-pattern allocation
	cls=0;
	for(DIMTYPE c=1;c<_classes;c++) if(_cs_sc[c]>_cs_sc[cls]) cls=c; // first maximum wins
	return true;
}


} // namespace
#endif // FSTCLASSIFIERLDA_H ///:~
