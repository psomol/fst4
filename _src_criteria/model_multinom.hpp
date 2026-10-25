#ifndef FSTMODELMULTINOM_H
#define FSTMODELMULTINOM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    model_multinom.hpp
   \brief   Implements multinomial model
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
#include <limits>
#include <cmath>
#include <cstring> // memcpy
#include <cstddef> // std::size_t (linearized-index arithmetic)
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

/*! \brief Multinomial ("bag of words") model: per-class term-count sums and the
    Laplace-smoothed term probabilities P(term|class) derived from them.

    For every class the per-feature term counts of all training patterns are summed
    (compute_Nsuminclass()); compute_theta() turns those sums into the smoothed conditional
    probabilities

        theta(f|c) = (1 + N(f,c)) / (d + sum over the exposed f' of N(f',c))

    that is, add-one (Laplace) smoothing over the ACTIVE vocabulary: the d features currently
    exposed by narrow_to(), not the full feature set, so the probabilities of a feature subset
    are normalized within that subset. Class priors P(c) are estimated from the class sizes of
    the WHOLE data set the accessor holds (not the train part alone; identical under stratified
    splitting), while the term counts come from the TRAIN part.
    compute_MI() and compute_IB() add the per-feature Mutual Information and "Individual
    Bhattacharyya" rankings consumed by the multinomial criteria.

    Unlike Model_Normal this model carries no mean/covariance and makes no continuity
    assumption -- it is the discrete counterpart, intended for count data such as document
    term frequencies.

    \note The data values are assumed to be non-negative term counts. Nothing verifies it;
          negative or fractional values make the sums and the smoothed probabilities meaningless.
    \note Call order is learn() (or learn(da,sub)), then compute_theta(), then optionally
          compute_MI() / compute_IB() -- and those two require compute_theta() to have been run
          at the FULL (denarrow()ed) feature set. The order is not enforced; calling out of
          order silently uses stale buffers.
*/
template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Model_Multinomial : public Model<SUBSET,DATAACCESSOR> {
public:
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to a feature subset, as consumed by learn()/narrow_to()
	Model_Multinomial() : _classes(0), _n_max(0), _n(0), _d(0), _allpatterns(0), _doc_avg_length(0) {notify("Model_Multinomial constructor");}
	//! \brief Deep-copies all learned buffers (sufficient statistics, theta/MI/IB caches, index maps).
	Model_Multinomial(const Model_Multinomial& mm); // copy-constructor
	virtual ~Model_Multinomial() {notify("Model_Multinomial destructor");}

	virtual void learn(PDataAccessor da) override; // learn model of full dimensionality
	/*! \brief Learns the model restricted to \p sub's selected features.
	    Computes only the sufficient statistics (_Nsuminclass, class priors _Pc, _Pc_d) via
	    compute_Nsuminclass(); it does NOT fill _theta/_MI/_IB -- call compute_theta() (and,
	    for the full feature set, compute_MI()/compute_IB()) explicitly afterwards. Reallocates
	    buffers when the class count or \p sub's raw dimensionality exceeds current capacity;
	    otherwise reuses them. Leaves the model unnarrowed (_d=_n). */
	virtual void learn(PDataAccessor da, const PSubset sub); // learn model of lower dimensionality

	void narrow_to(const PSubset sub) override; // fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow() override; // reset _index[] -> access is reset to the full matrix

	//! \brief Computes the Laplace-smoothed term-conditional probabilities _theta = P(feature|class)
	//! and _doc_avg_length, over the currently exposed dimensionality (_d, via narrow_to()/denarrow()).
	//! \note Requires learn()/learn(da,sub) to have populated _Nsuminclass/_allpatterns first.
	//!       compute_MI()/compute_IB() additionally require this to have been called while
	//!       denarrow()ed (_d==_n) -- they index _theta over the full _n range directly.
	void compute_theta();
	//! \brief Computes per-feature Mutual Information (fills _MI, full _n-length).
	//! \note Requires compute_theta() to have been called at the FULL (denarrow()ed) feature set --
	//!       this reads _theta densely over 0.._n per class, not through _index.
	void compute_MI(); // Mutual Information of individual features
	//! \brief Computes per-feature "Individual Bhattacharyya" (fills _IB, full _n-length): the
	//! class-pair-averaged Bhattacharyya-like coefficient of each single feature's term
	//! distribution, weighted by _Pc_d and scaled by _doc_avg_length.
	//! \note Same full-feature-set precondition on _theta as compute_MI().
	void compute_IB(); // Individual Bhattacharyya of individual features

	DIMTYPE get_n_max() const {return _n_max;} //!< allocated buffer capacity (dimensionality upper bound)
	DIMTYPE get_n() const {return _n;} //!< "full feature set" dimensionality the model was learned at
	DIMTYPE get_d() const {return _d;} //!< dimensionality currently exposed (post narrow_to()/denarrow())
	DIMTYPE get_classes() const {return _classes;} //!< number of classes, as set by the last learn() call

	REALTYPE get_doc_avg_length() const {return _doc_avg_length;} //!< average total term length per training pattern; valid after compute_theta()
	// NOTE: this and the protected getters below read internal buffers directly - the caller must
	//       ensure the model is alive, learned, and not being re-learn()ed (which reallocates them)
	REALTYPE get_Pc(const DIMTYPE c) const {assert(c<_classes); return _Pc[c];} //!< class prior P(class=c), estimated in learn() from class sizes
protected:
	template<class, class, class, class, class, class, class> friend class Criterion_Multinomial_Bhattacharyya;
	template<class, class, class, class, class, class, class> friend class Criterion_Multinomial_MI;
	template<class, class, class, class, class, class, class> friend class Classifier_Multinomial_NaiveBayes;
	const std::unique_ptr<REALTYPE[]>& get_theta() const {return _theta;} //!< raw P(term|class) buffer, class-major, valid at the dimensionality last passed to compute_theta()
	const std::unique_ptr<REALTYPE[]>& get_IB() const {return _IB;} //!< raw per-feature Individual Bhattacharyya buffer (full _n length), valid after compute_IB()
	const std::unique_ptr<REALTYPE[]>& get_MI() const {return _MI;} //!< raw per-feature Mutual Information buffer (full _n length), valid after compute_MI()
protected:
	DIMTYPE _classes; //!< number of classes; initialized in learn(). NOTE: expressions like _classes*_n*_n are evaluated in std::size_t where they occur, so they do not overflow DIMTYPE
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max; //!< allocated buffer capacity (dimensionality upper bound); see the NOTE above
	DIMTYPE _n; //!< "full feature set" dimensionality the model was learned at (<=_n_max); initialized in learn(). NOTE: expressions like _classes*_n*_n are evaluated in std::size_t where they occur, so they do not overflow DIMTYPE
	//! \brief Sums raw per-class, per-(learn-subspace)-feature term counts into _Nsuminclass and sets _allpatterns.
	//! Called from learn(da,sub); the subspace is the one recorded in _learn_index.
	void compute_Nsuminclass(PDataAccessor da);
	std::unique_ptr<DATATYPE[]> _Nsuminclass; //!< per-class, per-learn-subspace-feature raw term-count sums (class-major, stride _n); filled by compute_Nsuminclass()
	std::unique_ptr<REALTYPE[]> _theta; //!< P(term|class), Laplace-smoothed; class-major, stride = the _d in effect at the last compute_theta() call
	std::unique_ptr<REALTYPE[]> _Pc; //!< per-class prior P(class), estimated in learn() from da->getClassSize()/getClassSizeSum()
	std::unique_ptr<REALTYPE[]> _Pc_d; //!< per-class share of total learn-subspace term length; computed in learn(), consumed by compute_MI()/compute_IB()
	std::unique_ptr<REALTYPE[]> _Pv; //!< per-feature marginal term probability scratch, used internally by compute_MI()
	std::unique_ptr<REALTYPE[]> _MI; //!< per-feature (full _n-length) Mutual Information scores, filled by compute_MI()
	std::unique_ptr<REALTYPE[]> _IB; //!< per-feature (full _n-length) Individual Bhattacharyya scores, filled by compute_IB()

	DIMTYPE _d; //!< "virtual subset" dimensionality currently exposed (<=_n), via narrow_to()/denarrow()
	// NOTE: _index defines a sub-space of subs-space defined in _learn_index
	std::unique_ptr<DIMTYPE[]> _index; //!< maps feature subset indexes to raw (full set) feature indexes, used for narrow()ing

	IDXTYPE _allpatterns; //!< total training pattern count summed over all classes; set by compute_Nsuminclass() (called from learn())
	REALTYPE _doc_avg_length; //!< average total term length per training pattern; computed in compute_theta(), needed by compute_IB()

	// NOTE: _learn_index here is not used for narrow()ing, but for learn(da,sub)
	std::unique_ptr<DIMTYPE[]> _learn_index; //!< maps the to-be-learned model's subspace positions to raw (full-set) feature indices; set by learn(da,sub)
};

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Model_Multinomial(const Model_Multinomial& mm) : // copy-constructor
	Model<SUBSET,DATAACCESSOR>(mm),
	_classes(mm._classes),
	_n_max(mm._n_max),
	_n(mm._n),
	_d(mm._d),
	_allpatterns(mm._allpatterns),
	_doc_avg_length(mm._doc_avg_length)
{
	notify("Model_Multinomial copy-constructor.");
	if(_classes>0) {
		_Pc.reset(new REALTYPE[_classes]); memcpy((void *)_Pc.get(),(void *)(mm._Pc).get(),sizeof(REALTYPE)*_classes);
		_Pc_d.reset(new REALTYPE[_classes]); memcpy((void *)_Pc_d.get(),(void *)(mm._Pc_d).get(),sizeof(REALTYPE)*_classes);
		if(_n_max>0) {
			_Nsuminclass.reset(new DATATYPE[(std::size_t)_n_max*_classes]); memcpy((void *)_Nsuminclass.get(),(void *)(mm._Nsuminclass).get(),sizeof(DATATYPE)*_n_max*_classes);
			_theta.reset(new REALTYPE[(std::size_t)_n_max*_classes]); memcpy((void *)_theta.get(),(void *)(mm._theta).get(),sizeof(REALTYPE)*_n_max*_classes);
		}
	}
	if(_n_max>0)
	{
		_Pv.reset(new REALTYPE[_n_max]); memcpy((void *)_Pv.get(),(void *)(mm._Pv).get(),sizeof(REALTYPE)*_n_max);
		_MI.reset(new REALTYPE[_n_max]); memcpy((void *)_MI.get(),(void *)(mm._MI).get(),sizeof(REALTYPE)*_n_max);
		_IB.reset(new REALTYPE[_n_max]); memcpy((void *)_IB.get(),(void *)(mm._IB).get(),sizeof(REALTYPE)*_n_max);
		// to store subsubspace info of the (possibly subspace-learned) learned model
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(mm._index).get(),sizeof(DIMTYPE)*_n_max);
		// to store subspace info for the to-be-learned model
		_learn_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_learn_index.get(),(void *)(mm._learn_index).get(),sizeof(DIMTYPE)*_n_max);
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da)
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
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da, const PSubset sub)
{
#ifdef DEBUG
	{std::ostringstream sos; sos <<"Model_Multinomial::learn():"<<std::endl; syncout::print(std::cout,sos);}
#endif
	assert(sub);
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	if(_classes!=da->getNoOfClasses() || sub->get_n_raw()>_n_max) // insufficient buffers - reset constants and reallocate
	{
		_classes=da->getNoOfClasses();
		_n_max=sub->get_n_raw();

		_learn_index.reset(new DIMTYPE[_n_max]); // to store subspace info about the to-be-learned model
		_index.reset(new DIMTYPE[_n_max]); for(DIMTYPE f=0;f<_n_max;f++) _index[f]=f; // to store subsubspace info of the (possibly subspace-learned) learned model
		_Nsuminclass.reset(new DATATYPE[(std::size_t)_n_max*_classes]);
		_theta.reset(new REALTYPE[(std::size_t)_n_max*_classes]);
		_Pv.reset(new REALTYPE[_n_max]);
		_MI.reset(new REALTYPE[_n_max]);
		_IB.reset(new REALTYPE[_n_max]);
		_Pc.reset(new REALTYPE[_classes]);
		_Pc_d.reset(new REALTYPE[_classes]);
		// NOTE: class priors are estimated from complete data available through da.
		//       Note that when only parts of data are processed (i.e., using data splits), the
		//       estimates below would possibly not precisely depict the relative frequency of documents
		//       in stratified class parts, esp. if class sizes are small
		IDXTYPE _classsizesum=da->getClassSizeSum();
		assert(_classsizesum>0);
		if(_classsizesum==0) throw FST::fst_error("Model_Multinomial::learn: zero _classsizesum");
		for(DIMTYPE c=0;c<_classes;c++) _Pc[c]=(REALTYPE)da->getClassSize(c)/(REALTYPE)_classsizesum;
	}
	_n=sub->get_d_raw(); // actual to-be-learned model dimensionality is to be the one defined by the subset defined in sub
	_d=_n; // access to learned model is initially not narrow()ed
	DIMTYPE fi=0;
	for(DIMTYPE f : sub->selected_features()) {_learn_index[fi++]=f;}
	assert(fi==_n);
	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_Pc:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) sos << _Pc[c] << " ";
		sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
	compute_Nsuminclass(da); // sub info passed in form of _learn_index
	
	DATATYPE total_sum_length=0;
	std::size_t wCV=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		for(DIMTYPE f=0;f<_n;f++) total_sum_length+=_Nsuminclass[wCV+f];
		wCV+=_n;
	}
	if(total_sum_length==0) {
		for(DIMTYPE c=0;c<_classes;c++) _Pc_d[c]=1.0/(REALTYPE)_classes; // workaround to prevent division by zero below
	} else {
		wCV=0;
		for(DIMTYPE c=0;c<_classes;c++) {
			DATATYPE class_sum_length=0;
			for(DIMTYPE f=0;f<_n;f++) class_sum_length+=_Nsuminclass[wCV+f]; // in subset case this is done only for selected features
			_Pc_d[c]=(REALTYPE)class_sum_length/(REALTYPE)total_sum_length;
			wCV+=_n;
		}
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_Nsuminclass(PDataAccessor da)
{
	assert(_learn_index);
	assert(_Nsuminclass);
	for(std::size_t i=0;i<(std::size_t)_classes*_n;i++) _Nsuminclass[i]=0;
	typename DATAACCESSOR::PPattern p;
	IDXTYPE cnt;
	assert(da->getSplitIndex()>0); // a split must be in effect - the counts come from its TRAIN part
	std::size_t wCV=0;
	DIMTYPE _features=da->getNoOfFeatures();
	_allpatterns=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		for(const auto &blk : da->blocks(TRAIN)) { p=blk.data; cnt=blk.size;
			for(IDXTYPE i=0;i<cnt;i++) {
				for(DIMTYPE f=0;f<_n;f++) {_Nsuminclass[wCV+f]+=p[(std::size_t)i*_features+_learn_index[f]];}
			}
			_allpatterns+=cnt;
		}
		wCV+=_n;
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_Nsuminclass:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f=0;f<_n;f++) sos << _Nsuminclass[(std::size_t)c*_n+f] << " "; sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_theta()
{
	assert(_classes>0);
	assert(_n>0);
	assert(_d>0 && _d<=_n);
	assert(_index);
	assert(_allpatterns>0);
	assert(_Nsuminclass);
	assert(_theta);
	DATATYPE total_sum_length=0;
	std::size_t wCV=0, wCd=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		DATATYPE class_sum_length=0;
		for(DIMTYPE f=0;f<_d;f++) class_sum_length+=_Nsuminclass[wCV+_index[f]]; // in subset case this is done only for selected features
		total_sum_length+=class_sum_length;
		for(DIMTYPE f=0;f<_d;f++) { // in subset case this is done only for selected features
			// Laplace (add-one) smoothing over the ACTIVE vocabulary: the denominator uses _d,
			// the number of features currently exposed, so a narrowed model's probabilities sum
			// to 1 within the subset instead of leaving the mass of the unselected features out
			_theta[wCd++]=(1.0+(REALTYPE)_Nsuminclass[wCV+_index[f]])/((REALTYPE)_d+(REALTYPE)class_sum_length);
		}
		wCV+=_n;
	}
	if(_allpatterns==0) throw FST::fst_error("Model_Multinomial::compute_theta: division by zero _allpatterns");
	_doc_avg_length=(REALTYPE)total_sum_length/(REALTYPE)_allpatterns;
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_theta:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {
			REALTYPE tmp=0.0;
			for(DIMTYPE f=0;f<_d;f++) {tmp+=_theta[(std::size_t)c*_d+f]; sos << _theta[(std::size_t)c*_d+f] << " "; sos << std::endl;}
			sos << "theta sum over class<"<<c<<">: "<<tmp<<std::endl;
		}
		sos << "total_sum_length="<<total_sum_length<<", doc_avg_length="<<_doc_avg_length<<std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::narrow_to(const PSubset sub)
{
	assert(_index);
	assert(sub);
	assert(sub->get_n_raw()==_n); // NOTE: sub should represent subspace of the learned subspace ! not of the full space
	assert(sub->get_d_raw()>0);
	_d=0;
	for(DIMTYPE f : sub->selected_features()) _index[_d++]=f;
	assert(_d>0 && _d<=_n);
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::denarrow()
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_MI()
{
	// NOTE: needs _theta computed for full feature set
	assert(_classes>0);
	assert(_n>0);
	assert(_Nsuminclass);
	assert(_theta);
	assert(_Pc);
	assert(_Pv);
	assert(_MI);
	// Per-term mutual information between the term-occurrence event and the class under the
	// multinomial model. With P(c) = the share of class c in the total training term length
	// (_Pc_d), the probability that a randomly drawn term occurrence is term f AND comes from
	// class c is P(c)*theta(f|c); the term's marginal probability is
	//     P(f) = sum_c P(c) * theta(f|c)
	// and the term's contribution to I(term; class) is
	//     MI(f) = sum_c P(c) * theta(f|c) * ln( theta(f|c) / P(f) ).
	// Numerator and marginal use the SAME Laplace-smoothed theta, so P(f) is exactly the
	// marginal of the joint distribution; by the log-sum inequality every MI(f) is then >= 0,
	// which is what makes the sum over a subset cardinality-monotonic
	// (Criterion_Multinomial::is_monotonic()). Mixing a smoothed numerator with the raw corpus
	// frequency would instead score a term seen once in a small class by several nats and
	// drive a fraction of the vocabulary negative.
	std::size_t wCV;
	for(DIMTYPE f=0;f<_n;f++) {
		_Pv[f]=0.0;
		wCV=0;
		for(DIMTYPE c=0;c<_classes;c++) {_Pv[f]+=_Pc_d[c]*_theta[wCV+f]; wCV+=_n;}
		_MI[f]=0.0;
		if(_Pv[f]>0.0) {
			wCV=0;
			for(DIMTYPE c=0;c<_classes;c++) {
				const REALTYPE joint=_Pc_d[c]*_theta[wCV+f];
				if(joint>0.0) _MI[f]+=joint*(REALTYPE)log(_theta[wCV+f]/_Pv[f]);
				wCV+=_n;
			}
		}
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_MI:"<<std::endl;
		for(DIMTYPE f=0;f<_n;f++) sos << _MI[f] << " ";
		sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_IB()
{
	// NOTE: needs _theta computed for full feature set
	// For each feature f and each class pair (c1,c2) the Bhattacharyya coefficient of the two
	// Bernoulli term distributions is
	//     B = sqrt(theta1*theta2) + sqrt((1-theta1)*(1-theta2)),
	// turned into a distance by -doc_avg_length*ln(B) (the average document length scales the
	// single-term distance up to a whole document) and weighted by the two classes' term-length
	// shares Pc_d; _IB[f] is the mean over all class pairs. Larger = the feature separates better.
	assert(_classes>0);
	assert(_n>0);
	assert(_doc_avg_length>0);
	assert(_theta);
	assert(_Pc);
	assert(_IB);
	for(DIMTYPE f=0;f<_n;f++) _IB[f]=0.0;
	std::size_t wCV1,wCV2;
	DIMTYPE combs;
	REALTYPE _thetasum, value;
	for(DIMTYPE f=0;f<_n;f++) {
		value=0.0; combs=0;
		wCV1=0;
		for(DIMTYPE c1=0;c1<_classes;c1++) {
			wCV2=wCV1+_n;
			for(DIMTYPE c2=c1+1;c2<_classes;c2++) {
				_thetasum=sqrt(_theta[wCV1+f]*_theta[wCV2+f])+sqrt((1.0-_theta[wCV1+f])*(1.0-_theta[wCV2+f]));
#ifdef DEBUG
				{std::ostringstream sos; sos << _thetasum<< std::endl; syncout::print(std::cout,sos);}
#endif
				value+=((-_doc_avg_length)*(REALTYPE)log(_thetasum))*_Pc_d[c1]*_Pc_d[c2];
				combs++;
				wCV2+=_n;
			}
			wCV1+=_n;
		}
		assert(combs>0);
		if(combs==0) throw FST::fst_error("Model_Multinomial::compute_IB: division by zero combs");
		_IB[f]=value/(REALTYPE)combs; 
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_IB:"<<std::endl;
		for(DIMTYPE f=0;f<_n;f++) sos << _IB[f] << " ";
		sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}


} // namespace
#endif // FSTMODELMULTINOM_H ///:~
