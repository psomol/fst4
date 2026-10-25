#ifndef FSTCLASSIFIERNORMALBAYES_H
#define FSTCLASSIFIERNORMALBAYES_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    classifier_normal_bayes.hpp
   \brief   Implements Bayes classifier based on normal (gaussian) model
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
#include <sstream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846 //!< value of pi; POSIX \c \<cmath\> extension, provided here as a fallback since strict C++17 does not guarantee it
#endif
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
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

/*! \brief Implements Bayes classifier based on normal (gaussian) model

 Each class is modelled by its own mean vector and its own covariance matrix (the model
 is Model_Normal), so the decision boundaries are quadratic -- this is the QDA-style
 classifier; Classifier_LDA is the pooled-covariance, linear-boundary sibling. A pattern
 is assigned to the class of maximum posterior P(class|x), computed from the Gaussian
 densities and the class priors of the model.

 \note Classifier_Normal_Bayes should be able to handle _model learning
       in two ways: either [default] to learn() new subspace model in each train() - usable
       with FS involving multiple DataAccessor training splits, or to
       pre-learn() full space model and subsequently only narrow() the
       model in each train() - much faster, but usable only with non-changing
       set of training patterns, i.e., for one split data access only
 \warning Pre-learn mode must stay DISABLED (the default) when this classifier is driven by
       Criterion_Wrapper: the wrapper changes the training split between train() calls, and a
       pre-learned model would keep describing the data of the split it was learned on,
       silently returning optimistic and meaningless accuracies.
 \note Every class needs enough patterns for a non-singular covariance matrix of the evaluated
       subset (more patterns than features, and no exactly dependent features). A singular
       covariance makes the LU decomposition throw, which the criterion reports as a failed
       evaluation of that subset.
*/
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Normal_Bayes : public Clonable_Impl<Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Clonable_Impl<Classifier_Normal_Bayes,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Classifier
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset;            //!< shared pointer to the (fixed, non-reassignable) feature subset being trained/classified on
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_Normal_Bayes();
	virtual ~Classifier_Normal_Bayes() {notify("Classifier_Normal_Bayes destructor.");}

	/*! pre-learning mode: 
	    if true, calls once _model->learn() for full set size and later in train() uses only narrow()ing to access submatrixes
	             (makes sense only as long as training data do not change, i.e., within one split)
	    if false (default), learns new model in each train() call 
	*/
	void enable_prelearn_mode(const PDataAccessor da);
	void disable_prelearn_mode() {_prelearn_mode=false;} //!< reverts to the default: learn a new model in every train() call
	bool get_prelearn_mode() const {return _prelearn_mode;} //!< true if pre-learn mode is active (see enable_prelearn_mode())

	//! \brief (Re)allocates the per-class inverse-covariance, determinant, constant, posterior
	//! and log-domain-fallback buffers, sized to \p da's class/feature counts.
	//! \note initialize() does NOT itself learn mean[]/cov[] -- that happens in
	//!       enable_prelearn_mode() (once) or in every train() call (default, non-prelearn
	//!       mode). Must be called once whenever the training set's class/feature shape
	//!       changes, before train()/test().
	void initialize(const PDataAccessor da);

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // classifies pattern, returns the respective class index
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override
	{	// per-class posteriors P(class|x) for ranking metrics (AUC); classify() fills _Pwx on both its paths
		DIMTYPE cls;
		if(!classify(cls,pattern)) return false;
		scores.resize(_classes);
		for(DIMTYPE c=0;c<_classes;c++) scores[c]=(RETURNTYPE)_Pwx[c];
		return true;
	}
	virtual bool train(const PDataAccessor da, const PSubset sub) override; // learns from designated training part of data
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override; // estimates accuracy using designated test data
	
	DIMTYPE get_n() const {assert(_model); return _model->get_n();} //!< "full feature set" dimensionality the model was learned at (see Model_Normal::get_n())
	DIMTYPE get_d() const {assert(_model); return _model->get_d();} //!< dimensionality currently exposed by the model (see Model_Normal::get_d())


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_Normal_Bayes()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! \brief Deep-copies the learned model and all per-class buffers (inverse covariances,
	//! determinants, constants, posterior/fallback caches) and the feature index.
	Classifier_Normal_Bayes(const Classifier_Normal_Bayes& cnb); // copy-constructor
protected:
	std::unique_ptr<Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model; //!< the underlying normal (Gaussian) model supplying per-class mean/covariance

	DIMTYPE _classes;   //!< number of classes; sizes the per-class arrays below
	DIMTYPE _features;  //!< raw (full) feature count
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; //!< shared LU-decomposition scratch matrix, reused across classes in train()
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[]> _inverse; //!< per-class inverse covariance matrix (lazy allocation)
	std::unique_ptr<REALTYPE[]> _det; //!< per-class covariance determinant (lazy allocation)
	std::unique_ptr<REALTYPE[]> _constant; //!< per-class normalizing constant sqrt((2*pi)^d * _det[c]) (lazy allocation)
	std::unique_ptr<REALTYPE[]> _pxw; //!< per-class likelihood p(x|class), cached by classify() (lazy allocation)
	std::unique_ptr<REALTYPE[]> _Pwx; //!< per-class posterior P(class|x), cached by classify()/classify_scores() (lazy allocation)
	std::unique_ptr<REALTYPE[]> _log_constant; //!< ln of _constant, computed without forming _constant itself, for the log-domain fallback in classify() (lazy allocation)
	std::unique_ptr<REALTYPE[]> _ecoef; //!< per-class Mahalanobis terms buffered for the fallback (lazy allocation)

private:
	bool _prelearn_mode; //!< true = model learned once via enable_prelearn_mode(), narrow()ed per train(); false (default) = relearned in every train()
	std::unique_ptr<DIMTYPE[]> _index; //!< subset position -> raw feature index, filled by train() (first _subfeatures entries valid)
	DIMTYPE _subfeatures; //!< current subset size (== sub->get_d_raw() at the last train() call)
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Normal_Bayes(const Classifier_Normal_Bayes& cnb) :
	_classes(cnb._classes),
	_features(cnb._features),
	_prelearn_mode(cnb._prelearn_mode),
	_subfeatures(cnb._subfeatures)
{
	notify("Classifier_Normal_Bayes copy-constructor.");
	if(cnb._model) _model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cnb._model));
	if(cnb._LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(*cnb._LUtemp));
	if(_classes>0)
	{
		_inverse.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		for(DIMTYPE c=0;c<_classes;c++) _inverse[c]=cnb._inverse[c];
		_det.reset(new REALTYPE[_classes]); memcpy((void *)_det.get(),(void *)(cnb._det).get(),sizeof(REALTYPE)*_classes);
		_constant.reset(new REALTYPE[_classes]); memcpy((void *)_constant.get(),(void *)(cnb._constant).get(),sizeof(REALTYPE)*_classes);
		_pxw.reset(new REALTYPE[_classes]); memcpy((void *)_pxw.get(),(void *)(cnb._pxw).get(),sizeof(REALTYPE)*_classes);
		_Pwx.reset(new REALTYPE[_classes]); memcpy((void *)_Pwx.get(),(void *)(cnb._Pwx).get(),sizeof(REALTYPE)*_classes);
		if(cnb._log_constant) {_log_constant.reset(new REALTYPE[_classes]); memcpy((void *)_log_constant.get(),(void *)(cnb._log_constant).get(),sizeof(REALTYPE)*_classes);}
		if(cnb._ecoef) {_ecoef.reset(new REALTYPE[_classes]); memcpy((void *)_ecoef.get(),(void *)(cnb._ecoef).get(),sizeof(REALTYPE)*_classes);}
	}
	_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(cnb._index).get(),sizeof(DIMTYPE)*_features);
}



template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Normal_Bayes()
{
	notify("Classifier_Normal_Bayes constructor.");
	_model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
	_classes=0;
	_features=0;
	_subfeatures=0;
	_prelearn_mode=false;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::enable_prelearn_mode(const PDataAccessor da)
{
	notify("Classifier_Normal_Bayes::init_prelearn_mode().");
	assert(_model);
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	_model->learn(da);
	_prelearn_mode=true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Normal_Bayes::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	
	// (re)allocate buffers
	if(!_LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(da->getNoOfFeatures(),true));
	if(da->getNoOfFeatures()>_LUtemp->get_n_max()) _LUtemp->reset(da->getNoOfFeatures(),true);
	if(!_inverse || _classes!=da->getNoOfClasses()) _inverse.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[da->getNoOfClasses()]);
	for(DIMTYPE c=0;c<da->getNoOfClasses();c++) if(da->getNoOfFeatures()>_inverse[c].get_n_max()) _inverse[c].reset(da->getNoOfFeatures());
	if(!_det || _classes!=da->getNoOfClasses()) _det.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_constant || _classes!=da->getNoOfClasses()) _constant.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_pxw || _classes!=da->getNoOfClasses()) _pxw.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_Pwx || _classes!=da->getNoOfClasses()) _Pwx.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_log_constant || _classes!=da->getNoOfClasses()) _log_constant.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_ecoef || _classes!=da->getNoOfClasses()) _ecoef.reset(new REALTYPE[da->getNoOfClasses()]);
	_classes=da->getNoOfClasses();

	if(!_index || _features!=da->getNoOfFeatures()) _index.reset(new DIMTYPE[da->getNoOfFeatures()]);
	_features=da->getNoOfFeatures();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	// NOTE: initialize() must have been called for the current data shape (it allocates the
	//       per-class buffers this method fills). The model's mean/covariance are learned HERE,
	//       from the accessor's currently designated TRAIN part - which is why train() must be
	//       re-issued after every split change. When driven by Criterion_Wrapper, _prelearn_mode
	//       must therefore stay false (the default).

	notify("Classifier_Normal_Bayes::train().");
	assert(_model);
	assert(da);
	if(_prelearn_mode) assert(get_n()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_frozen_mode()==false);
	if(_prelearn_mode) assert(sub->get_n_raw()==get_n());
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	if(_prelearn_mode) _model->narrow_to(sub);
	else {_model->learn(da,sub); _model->denarrow();}

	// pre-compute matrix inverse, determinants depending on current subset
	REALTYPE tmpval=2.0*M_PI; for(DIMTYPE f=1;f<sub->get_d_raw();f++) tmpval*=2.0*M_PI; // tmpval = (2*pi)^d; overflows to inf for large d, which is what _log_constant exists for
	_LUtemp->redim(sub->get_d_raw());
	for(DIMTYPE c=0; c<_model->get_classes(); c++)
	{
		_inverse[c].redim(get_d());
		_model->get_cov()[c].LUdecompose(*_LUtemp);
		_model->get_cov()[c].invert(_inverse[c],*_LUtemp);
		_det[c]=_model->get_cov()[c].determinant(*_LUtemp);
		_constant[c]=sqrt(tmpval*_det[c]);
		// ln(_constant[c]) for the log-domain fallback in classify(), formed without ever
		// building the overflowing product: ln sqrt((2pi)^d * det) = 0.5*(d*ln(2pi) + ln det)
		_log_constant[c]=0.5*((REALTYPE)sub->get_d_raw()*std::log(2.0*M_PI)+_model->get_cov()[c].log_determinant(*_LUtemp));
	}

	// prepare feature subset index buffering
	DIMTYPE f;
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<get_n()); _index[_subfeatures++]=fsel;}
	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " "; sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
	assert(_subfeatures==sub->get_d_raw());
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// NOTE: train() must have been called - it learns (or narrows) the model and fills
	//       _inverse[], _det[], _constant[], _log_constant[] and the _index map this method
	//       reads. \p pattern must point at a full, raw feature vector: it is indexed through
	//       _index, not pre-narrowed by the caller.
	notify("Classifier_Normal_Bayes::classify().");
	assert(_model);
	assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(_index);
	assert(_subfeatures==_model->get_d());
	assert(_subfeatures>0);

	DIMTYPE f1, f2;
	DIMTYPE c_cand;
	REALTYPE ecoef, px, tmpval;

	// compute p(x|w) and p(x)
	// NOTE on the three index conventions in the inner loop: the pattern is raw and mapped
	//      through _index; the model's mean is read with at(), i.e. through the model's own
	//      narrowing map; the inverse covariance is read with at_raw(), because invert()
	//      writes its result raw (unindexed) in subset order. All three therefore address the
	//      same subset position - do not "unify" them.
	px=0.0;
	for(c_cand=0;c_cand<_classes;c_cand++)
	{
		ecoef=0.0; // the Mahalanobis form (x-mean)^T * inv(cov) * (x-mean)
		for(f1=0;f1<_subfeatures;f1++)
		{
			tmpval=0.0;
			for(f2=0;f2<_subfeatures;f2++)
			{
				tmpval+=(pattern[_index[f2]]-_model->get_mean()[c_cand].at(f2))*_inverse[c_cand].at_raw(f2,f1);
			}
			ecoef+=tmpval*(pattern[_index[f1]]-_model->get_mean()[c_cand].at(f1));
		}
		_ecoef[c_cand]=ecoef; // buffered for the log-domain fallback below
		_pxw[c_cand]=exp(-0.5*ecoef)/_constant[c_cand];
		px+=_pxw[c_cand]*_model->get_Pc(c_cand);
	}
	// log-domain fallback: without it, exp() underflow on distant patterns or (2pi)^d overflow
	// in high dimension yields px==0 (or inf/NaN) -> NaN posteriors -> the argmax below never
	// fires -> systematic misclassification into class 0 on exactly the hard patterns. The
	// fallback decides by log p(x|w)P(w) instead, which is mathematically the same argmax; the
	// plain-domain path above is used unchanged whenever it is healthy.
	// (A NaN px fails both tests below and so takes the fallback, as intended.)
	if(!(px>(REALTYPE)0.0) || !std::isfinite(px)) {
		REALTYPE lbest=0.0, lc;
		for(c_cand=0;c_cand<_classes;c_cand++) {
			lc=-0.5*_ecoef[c_cand]-_log_constant[c_cand]+std::log(_model->get_Pc(c_cand));
			_Pwx[c_cand]=lc; // temporarily holds log-posteriors (unnormalized)
			if(c_cand==0 || lc>lbest) {lbest=lc; cls=c_cand;}
		}
		// meaningful posteriors via max-shifted exponentiation
		REALTYPE psum=0.0;
		for(c_cand=0;c_cand<_classes;c_cand++) {_Pwx[c_cand]=exp(_Pwx[c_cand]-lbest); psum+=_Pwx[c_cand];}
		for(c_cand=0;c_cand<_classes;c_cand++) {_Pwx[c_cand]/=psum; _pxw[c_cand]=_Pwx[c_cand];} // _pxw left proportional, no absolute scale survives the shift
		return true;
	}
	// compute P(w|x)
	for(c_cand=0;c_cand<_classes;c_cand++) _Pwx[c_cand]=(_pxw[c_cand]*_model->get_Pc(c_cand))/px;
	// find maximum P(w|x)
	tmpval=_Pwx[0]; cls=0;
	for(c_cand=1;c_cand<_classes;c_cand++) if(_Pwx[c_cand]>tmpval) {tmpval=_Pwx[c_cand]; cls=c_cand;}

	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: train() must have been called for the subset being tested (it fills the model and
	//       the per-class buffers classify() reads); the accessor must be positioned on the
	//       split whose TEST part is to be classified
	notify("Classifier_Normal_Bayes::test().");
	assert(_model);
	assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(da);
	if(_prelearn_mode) assert(da->getNoOfFeatures()==get_n()); 
	else assert(da->getNoOfFeatures()>=get_n());
	assert(da->getNoOfClasses()>0);
	assert(_index);
	assert(_subfeatures==_model->get_d());
	assert(_subfeatures>0);
		
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	IDXTYPE count, correct;
	DIMTYPE clstmp;
	DIMTYPE _features=da->getNoOfFeatures();

	
	count=0;
	correct=0;
	for(DIMTYPE c_test=0;c_test<_classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
		{
			if(!classify(clstmp,&p[(std::size_t)i*_features])) return false;
			if(clstmp==c_test) correct++;
			count++;
		}
	}
	assert(count>0);
	result=(RETURNTYPE)correct/(RETURNTYPE)count;
#ifdef DEBUG
	{
		std::ostringstream sos; sos << " result=" << result << std::endl;
		syncout::print(std::cout,sos);
	}
#endif	
	return true;
}


} // namespace
#endif // FSTCLASSIFIERNORMALBAYES_H ///:~
