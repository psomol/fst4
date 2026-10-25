#ifndef FSTMODELNORMAL_H
#define FSTMODELNORMAL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    model_normal.hpp
   \brief   Implements normal (gaussian) model
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
#include <sstream>
#include <limits>
#include <cmath>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "model.hpp"
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

/*! \brief Implements normal (gaussian) model

    Per class: the mean vector and the full covariance matrix, estimated from the accessor's
    currently designated TRAIN part, and the prior probability P(c), estimated from the class
    sizes of the WHOLE data set the accessor holds (getClassSizeSum(), see get_Pc()) - identical
    to the train-part frequency under stratified splitting. Each class keeps its OWN covariance
    (no pooling), which is what makes the components built on this model quadratic in the data:
    Classifier_Normal_Bayes is the QDA-style classifier, and the normal-model filter criteria
    (Bhattacharyya, generalized Mahalanobis, divergence) all read these per-class moments.
    Classifier_LDA is the pooled-covariance, linear counterpart and reuses the same moments.

    Two independent notions of dimensionality live here. learn(da,sub) fixes what the model IS:
    it estimates the moments over sub's selected features only, and get_n() is that count.
    narrow_to() then selects a sub-subspace of the LEARNED one for reading, reported by get_d();
    that is a cheap re-indexing, not a re-estimation. Feature indices passed to narrow_to() are
    therefore positions within the learned subspace, not raw dataset feature ids.

    \note Covariances are the unbiased estimates (divided by the class's training pattern count
          minus one). A class with fewer training patterns than features - or with exactly
          dependent features - yields a singular covariance, which the consumers cannot invert.
*/
template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Model_Normal : public Model<SUBSET,DATAACCESSOR> {
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor; //!< shared pointer to the data accessor supplying training data
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset, as consumed by learn(da,sub)/narrow_to()
	Model_Normal() : _classes(0), _n_max(0), _n(0), _d(0) {notify("Model_Normal (empty) constructor");}
	Model_Normal(const Model_Normal& mn); //!< copy-constructor (deep-copies the per-class mean/covariance arrays and the learn-time index map)
	virtual ~Model_Normal() {notify("Model_Normal destructor");}

	virtual void learn(PDataAccessor da) override; // learn model of full dimensionality
	//! \brief Learns per-class mean vectors and covariance matrices restricted to \p sub's selected features.
	//! Sets get_n()==get_d()==sub->get_d_raw() and records the raw dataset feature id of each learned
	//! dimension in _learn_index; a subsequent narrow_to() must pass a subset with get_n_raw()==get_n()
	//! (i.e. a further restriction of THIS learned subspace, not of \p da's full feature space).
	virtual void learn(PDataAccessor da, const PSubset sub);

	// NOTE: the learn() methods create model structures. narrow() methods allow access to sub-space in existing learned structures
	void narrow_to(const PSubset sub) override; // fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow() override; // reset _index[] -> access is reset to the full matrix

	DIMTYPE get_n_max() const {return _n_max;} //!< the allocated capacity (max n learn() can grow to without reallocating)
	DIMTYPE get_n() const {return _n;} //!< the full ("raw") dimensionality the model is currently learned at, get_n()<=get_n_max()
	DIMTYPE get_d() const {return _d;} //!< the current virtual (narrow_to()'d) dimensionality, get_d()<=get_n()
	DIMTYPE get_classes() const {return _classes;} //!< number of classes the model was learned over (learn()'s da->getNoOfClasses())

	//! Class prior probability P(c), estimated in learn() as class c's relative sample frequency over \p da.
	//! \note caller must ensure this is called only while the model's learned structures (_mean, _Pc, _cov) exist.
	REALTYPE get_Pc(const DIMTYPE c) const {assert(c<_classes); return _Pc[c];}
protected:
	template<class, class, class, class, class, class, class> friend class Classifier_Normal_Bayes;
	template<class, class, class, class, class, class, class> friend class Classifier_LDA; // pooled-covariance discriminant, pools the learned per-class moments
	template<class, class, class, class, class, class, class> friend class Criterion_Normal; // owns the shared _meandif computation for all normal-model criteria
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_Bhattacharyya;
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_GMahalanobis;
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_Divergence;
	const std::unique_ptr<Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[]>& get_mean() const {return _mean;} //!< per-class learned mean vectors (array of get_classes() Indexed_Vector, each of dimensionality get_n()/get_d()); for the friend criteria/classifiers above
	const std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[]>& get_cov() const {return _cov;} //!< per-class learned covariance matrices (array of get_classes() Indexed_Matrix, each get_n()/get_d() sized); for the friend criteria/classifiers above
protected:
	DIMTYPE _classes; //!< number of classes the model is learned over; see get_classes() -- NOTE: expressions like _classes*_n*_n are evaluated in std::size_t where they occur, so they do not overflow DIMTYPE
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max; //!< allocated capacity: the largest sub->get_n_raw() (the subset's full allocation size, at any learn(da,sub) call) _mean[]/_cov[] have been grown to hold
	DIMTYPE _n; //!< full ("raw") dimensionality the model is currently learned at, _n<=_n_max; see get_n() -- NOTE: expressions like _classes*_n*_n are evaluated in std::size_t where they occur, so they do not overflow DIMTYPE
	std::unique_ptr<REALTYPE[]> _Pc; //!< per-class prior probabilities P(c), get_classes() entries; see get_Pc()
	std::unique_ptr<Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[]> _mean; //!< per-class learned mean vectors, get_classes() entries; see get_mean()
	std::unique_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[]> _cov; //!< per-class learned covariance matrices, get_classes() entries; see get_cov()
	// NOTE: every cov[] and mean[] carries its own copy of the same subspace index map, so a
	// narrow_to() has to walk all 2*_classes of them. It is a deliberate trade: the vectors and
	// matrices stay self-contained and usable on their own.
	DIMTYPE _d; //!< current virtual (narrow_to()'d) dimensionality, _d<=_n; see get_d()
	// NOTE: _n and _d must always stay synchronized with those inside _mean and _cov - nothing
	//       enforces it, so every method that changes one must change all of them (see learn(),
	//       narrow_to(), denarrow())

	//! Maps each learned dimension (0.._n-1) back to the raw feature id it was learned from (the id
	//! supplied via learn(da,sub)'s sub->selected_features()); NOTE this is distinct from the narrow_to()
	//! mechanism inside _mean[]/_cov[] -- it exists only to support learn(da,sub), not narrow()ing.
	std::unique_ptr<DIMTYPE[]> _learn_index;
};

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Model_Normal(const Model_Normal& mn) : // copy-constructor
	Model<SUBSET,DATAACCESSOR>(mn),
	_classes(mn._classes),
	_n_max(mn._n_max),
	_n(mn._n),
	_d(mn._d)
{
	notify("Model_Normal copy-constructor.");
	if(_classes>0) {
		_Pc.reset(new REALTYPE[_classes]); memcpy((void *)_Pc.get(),(void *)(mn._Pc).get(),sizeof(REALTYPE)*_classes);
		// default-constructed, then assigned one by one below: the array forms cannot pass
		// constructor arguments, and the assignment operators do the deep copy anyway
		_mean.reset(new Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_cov.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		for(DIMTYPE c=0;c<_classes;c++) {_mean[c]=mn._mean[c]; _cov[c]=mn._cov[c];}
	}
	if(_n_max>0)
	{
		// to store subspace info for the to-be-learned model
		_learn_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_learn_index.get(),(void *)(mn._learn_index).get(),sizeof(DIMTYPE)*_n_max);
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da)
{
	assert(da);
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	PSubset fullset(new SUBSET(da->getNoOfFeatures()));
	fullset->select_all();
#ifdef DEBUG
	{std::ostringstream sos; sos << "fullset = "<< *fullset << std::endl; syncout::print(std::cout,sos);}
#endif
	learn(da,fullset);
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da, const PSubset sub)
{
	notify("Model_Normal::learn():");
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_n_raw()==da->getNoOfFeatures());
	assert(sub->get_d_raw()>0);

	if(_classes!=da->getNoOfClasses())
	{
		_classes=da->getNoOfClasses();
		_mean.reset(new Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_cov.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_Pc.reset(new REALTYPE[_classes]);
		// the fresh vectors/matrices have no storage yet; zeroing the recorded capacity makes
		// the "insufficient buffers" test below fire unconditionally and size them
		_n_max=0;
	}
	// NOTE: the class priors are re-estimated on every learn() call, even when da has not changed.
	//       It is a negligible cost next to the covariance accumulation and keeps learn() free of
	//       any assumption about what the caller changed since the last call.
	// NOTE: class priors are estimated from complete data available through da.
	//       Note that when only parts of data are processed (i.e., using data splits), the
	//       estimates below would possibly not precisely depict the relative frequency of documents
	//       in stratified class parts, esp. if class sizes are small
	IDXTYPE _allpatterns=da->getClassSizeSum(); // just to estimate class priors
	assert(_allpatterns>0);
	for(DIMTYPE c=0;c<_classes;c++) _Pc[c]=(REALTYPE)da->getClassSize(c)/(REALTYPE)_allpatterns;

	if(sub->get_n_raw()>_n_max) // insufficient buffers - reset constants and reallocate
	{
		_n_max=sub->get_n_raw();
		for(DIMTYPE c=0;c<_classes;c++) {
			_mean[c].reset(_n_max);
			_cov[c].reset(_n_max);
		}
		_learn_index.reset(new DIMTYPE[_n_max]); // to store subspace info for the to-be-learned model
	}
	
	_n=sub->get_d_raw(); // actual to-be-learned model dimensionality is to be the one defined by the subset defined in sub
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].redim(_n); _mean[c].set_all_raw_to(0.0);
		_cov[c].redim(_n); _cov[c].set_all_raw_to(0.0);
	}
	_d=_n; // access to learned model is initially not narrow()ed
	DIMTYPE fi=0;
	for(DIMTYPE f : sub->selected_features()) {_learn_index[fi++]=f;}
	assert(fi==_n);
	
	typename DATAACCESSOR::PPattern p;
	IDXTYPE cnt;
	// pattern rows are laid out with the CURRENT dataset's raw feature count as stride. Note
	// that this is sub->get_n_raw() and NOT _n_max: the latter only grows, so after the model
	// has been reused on a smaller dataset the two differ and _n_max-strided reads would walk
	// off the row boundaries.
	const DIMTYPE _stride=sub->get_n_raw();
	assert(da->getSplitIndex()>0); // splitting loop to be maintained externally
	// mean
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		assert(da->getNoOfPatterns(TRAIN)>0);
#ifdef DEBUG
		{std::ostringstream sos; sos << "da->getNoOfPatterns(TRAIN) [class="<<c<<"] = "<<da->getNoOfPatterns(TRAIN)<<std::endl; syncout::print(std::cout,sos);}
#endif
		for(const auto &blk : da->blocks(TRAIN)) { p=blk.data; cnt=blk.size;
			for(IDXTYPE i=0;i<cnt;i++)
				for(DIMTYPE f=0;f<_n;f++) (_mean[c])[f]+=p[(std::size_t)i*_stride+_learn_index[f]];
		}
		for(DIMTYPE f=0;f<_n;f++) (_mean[c])[f]/=da->getNoOfPatterns(TRAIN);
	}
	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_Pc:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) sos << _Pc[c] << " ";
		sos << std::endl;
		sos << "_mean:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f=0;f<_n;f++) sos << (_mean[c])[f] << " "; sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
	// cov
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		for(const auto &blk : da->blocks(TRAIN)) { p=blk.data; cnt=blk.size;
			for(IDXTYPE i=0;i<cnt;i++) {
				for(DIMTYPE f1=0;f1<_n;f1++)
				for(DIMTYPE f2=f1;f2<_n;f2++){
					_cov[c].at_raw(f1,f2)+=(p[(std::size_t)i*_stride+_learn_index[f1]]-(_mean[c])[f1])*(p[(std::size_t)i*_stride+_learn_index[f2]]-(_mean[c])[f2]);
				}
			}
		}
	}
	// normalize and copy to lower triangle
	// ORDER MATTERS: the upper triangle is mirrored into the lower one FIRST and the division
	// by the pattern count is applied to the whole matrix afterwards. Dividing first and
	// mirroring second would work too, but mirroring after the division of only one triangle
	// would not - do not reorder these two loops.
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		// the unbiased (n-1) divisor. A class with a single training pattern would make it 0:
		// the assert catches that in a debug build, the clamp keeps a release build from
		// dividing by zero (the covariance is meaningless either way).
		REALTYPE pats=da->getNoOfPatterns(TRAIN)-1;
		assert(pats>0);
		if(pats<1) pats=1;
		for(DIMTYPE f1=0;f1<_n;f1++)
			for(DIMTYPE f2=0;f2<f1;f2++) {
				_cov[c].at_raw(f1,f2)=_cov[c].at_raw(f2,f1);
			}
		for(DIMTYPE f1=0;f1<_n;f1++)
			for(DIMTYPE f2=0;f2<_n;f2++) {
				_cov[c].at_raw(f1,f2)/=pats;
			}
	}

	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_cov:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f1=0;f1<_n;f1++) {for(DIMTYPE f2=0;f2<_n;f2++) sos << _cov[c].at_raw(f1,f2) << " "; sos << std::endl;} sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
}


template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::narrow_to(const PSubset sub)
{
	assert(_mean);
	assert(_cov);
	assert(sub->get_n_raw()==_n); // NOTE: sub should represent subspace of the learned subspace ! not of the full space
	assert(sub->get_d_raw()>0);
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].narrow_to(sub);
		_cov[c].narrow_to(sub);
	}
	_d=sub->get_d_raw();
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::denarrow()
{
	assert(_mean);
	assert(_cov);
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].denarrow();
		_cov[c].denarrow();
	}
	_d=_n;
}

} // namespace
#endif // FSTMODELNORMAL_H ///:~
