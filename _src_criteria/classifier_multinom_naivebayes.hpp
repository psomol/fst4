#ifndef FSTCLASSIFIERMULTINOMNAIVEBAYES_H
#define FSTCLASSIFIERMULTINOMNAIVEBAYES_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    classifier_multinom_naivebayes.hpp
   \brief   Implements Naive-like Bayes classifier based on multinomial model
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
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "classifier.hpp"
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

/*! \brief Implements Naive-like Bayes classifier based on multinomial model

 The document classification rule of the multinomial model (Model_Multinomial): a pattern is
 assigned to the class maximizing

     ln P(c) + sum over the selected features f of  x[f] * ln theta(f|c)

 where x[f] is the pattern's term count and theta(f|c) the class-conditional term probability
 the model learned. This is the log-posterior up to the constants (the multinomial coefficient
 and the evidence) that are the same for every class and therefore cannot change the argmax.

 "Naive-LIKE" rather than plain Naive Bayes: the per-feature factors are the multinomial term
 probabilities of one shared word distribution per class, not independent per-feature
 distributions - the feature contributions are coupled through theta's normalization over the
 selected features, so the usual conditional-independence reading does not apply literally.

 \warning A ONE-FEATURE subset is degenerate here: with a single feature theta(f|c)==1 in every
          class, every log term vanishes and the score no longer depends on the data at all.
          The class treats such subsets as unusable rather than pretending to classify them, and
          does so consistently: classify() returns false, test() reports accuracy 0, the
          confusion matrix holds one off-diagonal count (so every confusion-based metric is 0 as
          well), and classify_scores() returns all-equal scores (AUC 0.5). Feature selection
          driven by this classifier therefore never settles on a single feature.
*/
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Multinomial_NaiveBayes : public Clonable_Impl<Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Clonable_Impl<Classifier_Multinomial_NaiveBayes,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > parent_impl; //!< implements clone(); sharing/stateless_clone throw in Classifier
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor; //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset; //!< shared pointer to the (fixed, non-reassignable) feature subset being trained/evaluated on
	typedef typename DATAACCESSOR::PPattern PPattern; //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_Multinomial_NaiveBayes();
	virtual ~Classifier_Multinomial_NaiveBayes() {notify("Classifier_Multinomial_NaiveBayes destructor.");}

	/*! pre-learning mode: 
	    if true, calls once _model->learn() for full set size and later in train() uses only narrow()ing to access submatrixes
	             (makes sense only as long as training data do not change, i.e., within one split)
	    if false (default), learns new model in each train() call 
	*/
	void enable_prelearn_mode(const PDataAccessor da);
	void disable_prelearn_mode() {_prelearn_mode=false;} //!< reverts to the default (per-train() relearning) mode; see enable_prelearn_mode()
	bool get_prelearn_mode() const {return _prelearn_mode;} //!< true if enable_prelearn_mode() is currently in effect

	//! (Re)allocates the _Pcd and _index buffers to \p da's current class/feature counts. train() calls
	//! this itself on every invocation, so an explicit call is only needed to size these buffers ahead
	//! of train() (the fst4 CLI's classifier factory does this right after construction).
	//! \note It allocates only; it learns nothing. The multinomial model itself is learned (or
	//! narrowed) inside train().
	void initialize(const PDataAccessor da);

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // classifies pattern, returns the respective class index
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override
	{	// per-class log-posteriors for ranking metrics (AUC); single features are unusable
		// (theta=1 in each class, see classify()) -> equal scores = uninformative ranking (AUC 0.5)
		if(_subfeatures==1) {scores.assign(_classes,(RETURNTYPE)0.0); return true;}
		DIMTYPE cls;
		if(!classify(cls,pattern)) return false;
		scores.resize(_classes);
		for(DIMTYPE c=0;c<_classes;c++) scores[c]=(RETURNTYPE)_Pcd[c];
		return true;
	}
	virtual bool test_confusion(std::vector<RETURNTYPE> &confusion, DIMTYPE &classes, const PDataAccessor da) override
	{	// mirrors test()'s "single features unusable" convention (correct=0, count=1):
		// one off-diagonal count -> every confusion metric scores 0, like accuracy does
		if(_subfeatures==1) {
			classes=da->getNoOfClasses();
			if(classes<2) return false;
			confusion.assign((size_t)classes*(size_t)classes,(RETURNTYPE)0.0);
			confusion[1]=1.0; // true class 0 predicted as class 1
			return true;
		}
		return Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_confusion(confusion,classes,da);
	}
	virtual bool train(const PDataAccessor da, const PSubset sub) override; // learns from designated training part of data
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override; // estimates accuracy using designated test data
	
	DIMTYPE get_n() const {assert(_model); return _model->get_n();} //!< the underlying model's full ("raw") learned dimensionality (Model_Multinomial::get_n())
	DIMTYPE get_d() const {assert(_model); return _model->get_d();} //!< the underlying model's current virtual (narrowed) dimensionality (Model_Multinomial::get_d())


	
	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_Multinomial_NaiveBayes()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	Classifier_Multinomial_NaiveBayes(const Classifier_Multinomial_NaiveBayes& cmnb); //!< copy-constructor (deep-copies _model, _Pcd and _index, each only if the source had them allocated, i.e. only after initialize())
protected:
	std::unique_ptr<Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model; //!< the underlying multinomial model, learned/narrowed by train()

	DIMTYPE _classes; //!< class count, sized by initialize() (dimensions _Pcd)
	DIMTYPE _features; //!< raw feature count, sized by initialize() (dimensions _index)
	std::unique_ptr<REALTYPE[]> _Pcd; //!< per-class score of the pattern last passed to classify(): the UNNORMALIZED log-posterior ln P(c) + sum_f x[f]*ln theta(f|c), not a probability (lazy allocation)

private:
	bool _prelearn_mode; //!< see enable_prelearn_mode()/disable_prelearn_mode()/get_prelearn_mode()
	std::unique_ptr<DIMTYPE[]> _index; //!< subset position -> raw feature index (first _subfeatures entries valid), filled by train()
	DIMTYPE _subfeatures; //!< number of features in the currently trained subset (sub->get_d_raw() at the last train() call)
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Multinomial_NaiveBayes(const Classifier_Multinomial_NaiveBayes& cmnb) :
	_classes(cmnb._classes),
	_features(cmnb._features),
	_prelearn_mode(cmnb._prelearn_mode),
	_subfeatures(cmnb._subfeatures)
{
	notify("Classifier_Multinomial_NaiveBayes copy-constructor.");
	if(cmnb._model) _model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cmnb._model));
	if(_classes>0)
	{
		_Pcd.reset(new REALTYPE[_classes]); memcpy((void *)_Pcd.get(),(void *)(cmnb._Pcd).get(),sizeof(REALTYPE)*_classes);
	}
	if(_features>0) // guards against memcpy from cmnb._index==null when copying a never-initialize()d instance (_features>0 implies _index was allocated -- see initialize())
	{
		_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(cmnb._index).get(),sizeof(DIMTYPE)*_features);
	}
}



template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Multinomial_NaiveBayes()
{
	notify("Classifier_Multinomial_NaiveBayes constructor.");
	_model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
	_classes=0;
	_features=0;
	_subfeatures=0;
	_prelearn_mode=false;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::enable_prelearn_mode(const PDataAccessor da)
{
	notify("Classifier_Multinomial_NaiveBayes::init_prelearn_mode().");
	assert(_model);
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	_model->learn(da);
	_prelearn_mode=true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Multinomial_NaiveBayes::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	
	if(!_Pcd || _classes!=da->getNoOfClasses()) _Pcd.reset(new REALTYPE[da->getNoOfClasses()]);
	_classes=da->getNoOfClasses();

	if(!_index || _features!=da->getNoOfFeatures()) _index.reset(new DIMTYPE[da->getNoOfFeatures()]);
	_features=da->getNoOfFeatures();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	// NOTE: initialize() is called unconditionally on every train(), so the buffers always match
	//       the accessor currently in use even when the wrapper switches datasets or splits. It
	//       only (re)allocates when the class or feature count changed, so the repeated call is
	//       cheap; it could be skipped while evaluating several subsets over one unchanged split.
	initialize(da);

	notify("Classifier_Multinomial_NaiveBayes::train().");
	assert(_model);
	assert(da);
	if(_prelearn_mode) assert(get_n()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_frozen_mode()==false);
	if(_prelearn_mode) assert(sub->get_n_raw()==get_n());
	assert(sub->get_n_raw()==da->getNoOfFeatures());
	
	if(_prelearn_mode) _model->narrow_to(sub); // assume _n-dimensional model is pre-learned and needs to be narrow()ed only (NOTE: training data must not change since)
	else {_model->learn(da,sub); _model->denarrow();} // re-learn from scratch, training data has changed (i.e., after switch to next data split)

	// prepare feature subset index buffering
	DIMTYPE f;
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<get_n()); _index[_subfeatures++]=fsel;}
	
	_model->compute_theta(); //thetas to be used in classify() and test()
	
#ifdef DEBUG
	{std::ostringstream sos; sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " "; syncout::print(std::cout,sos);}
#endif
	assert(_subfeatures==sub->get_d_raw());
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// NOTE: train() must have been called - it learns (or narrows) the model, fills the _index
	//       map and computes the model's theta buffer this method reads. \p pattern must point
	//       at a full, raw feature vector: it is indexed through _index, not pre-narrowed.
	notify("Classifier_Multinomial_NaiveBayes::classify().");
	assert(_model);
	if(_prelearn_mode) assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(_index);
	assert(_subfeatures==_model->get_d());
	if(!_prelearn_mode) assert(_subfeatures==_model->get_n());
	assert(_subfeatures>0);

	DIMTYPE f;
	DIMTYPE c_cand;
	REALTYPE res;
	DIMTYPE wTH;
	// a non-const pointer into the model's theta buffer, only read from here; taking it is safe
	// because this object owns _model exclusively and nothing reallocates the buffer during the
	// loop below (the class-major stride is _subfeatures, the model's current get_d())
	REALTYPE *theta=&(_model->get_theta()[0]);
	REALTYPE *theta_tmp;

	if(_subfeatures==1) { 
		// NOTE: single features are unusable for classification because theta=1 in each class,
		//       thus log(theta)=0 and consequently Pcd[] does not depend on feature frequency
		cls=0; // consider all single features equally unusable
		return false;
	} else {
		// compute P(c|pattern) for each class
		wTH=0;
		for(c_cand=0;c_cand<_classes;c_cand++)
		{
			theta_tmp=&theta[wTH];
			res=0.0;
			for(f=0;f<_subfeatures;f++)
			{
				res+=(REALTYPE)pattern[_index[f]] * log(theta_tmp[f]);
			}
			_Pcd[c_cand]=log(_model->get_Pc(c_cand))+res;
			wTH+=_subfeatures;
		}
		// find maximum P[c|pattern]
		res=_Pcd[0]; cls=0;
		for(c_cand=1;c_cand<_classes;c_cand++) if(_Pcd[c_cand]>res) {res=_Pcd[c_cand]; cls=c_cand;}
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: train() must have been called for the subset being tested; the accessor must be
	//       positioned on the split whose TEST part is to be classified
	notify("Classifier_Multinomial_NaiveBayes::test().");
	assert(_model);
	if(_prelearn_mode) assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(da);
	if(_prelearn_mode) assert(da->getNoOfFeatures()==get_n());
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfClasses()==_classes);
	assert(_index);
	assert(_subfeatures==_model->get_d());
	if(!_prelearn_mode) assert(_subfeatures==_model->get_n());
	assert(_subfeatures>0);
		
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	IDXTYPE count, correct;
	DIMTYPE _features=da->getNoOfFeatures();
	DIMTYPE clstmp;

	if(_subfeatures==1) { 
		// NOTE: single features are unusable for classification because theta=1 in each class,
		//       thus log(theta)=0 and consequently Pcd[] does not depend on feature frequency
		correct=0; count=1; // consider all single features equally unusable
	} else {

	
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
#endif // FSTCLASSIFIERMULTINOMNAIVEBAYES_H ///:~
