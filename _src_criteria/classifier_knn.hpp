#ifndef FSTCLASSIFIERKNN_H
#define FSTCLASSIFIERKNN_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    classifier_knn.hpp
   \brief   Implements k-Nearest Neighbor classifier
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
#include <list>
#include "classifier.hpp"
#include "error.hpp"
#include "global.hpp"

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

/*! \brief Implements k-Nearest Neighbor classifier

    The decision rule: collect the \e k training patterns nearest to the classified
    pattern under the supplied DISTANCE, count the class votes among them, and take the
    argmax; the first maximum wins, i.e. ties are resolved in favour of the lowest class
    index. classify() and classify_scores() implement the same rule - classify() is the
    argmax of the vote fractions classify_scores() returns - so accuracy, the confusion
    matrix and the ranking metrics (AUC, ROC) all describe one and the same classifier. */
template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
class Classifier_kNN : public Clonable_Impl<Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Classifier_kNN,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Classifier
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor; //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to the feature subset being trained/evaluated on
	typedef typename DATAACCESSOR::PPattern PPattern; //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_kNN() {set_k(1); _nns_max_size=1; notify("Classifier_kNN constructor.");}
	virtual ~Classifier_kNN() {notify("Classifier_kNN destructor.");}

	void set_k(const DIMTYPE k) {assert(k>0); _k=k; _nns_max_size=_k;} //!< sets the number of nearest neighbours collected and voted over; \p k must be > 0
	DIMTYPE get_k() const {return _k;} //!< the k set by set_k() (default 1)

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // classifies pattern, returns the respective class index
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override; // per-class vote fractions among the k nearest neighbours for ranking metrics (AUC)
	//! With kNN there is actually no training: this just narrows/allocates the distance metric to
	//! \p sub's subspace and stores \p da, to be enumerated as training neighbours from classify()/test().
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	// test() = the canonical accuracy loop inherited from Classifier, driven by classify() below



	virtual std::ostream& print(std::ostream& os) const override;
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	Classifier_kNN(const Classifier_kNN& cknn); //!< copy-constructor (clones _distance if allocated; _nns/_k/_nns_max_size are value-copied)
protected:
	// NOTE: neighbours at equal distance but of different classes get no special treatment -
	//       see the tie remark in sort_in()

	//! Holds information on distance and data-class membership of neighbors processed in Classifier_kNN
	class Neighbour{
	public:
		Neighbour() {}
		Neighbour(const RETURNTYPE value, const DIMTYPE cls) {_value=value; _cls=cls;} //!< constructs from a distance \p value and the neighbour's class \p cls
		Neighbour(const Neighbour& nei) {_value=nei._value; _cls=nei._cls;}	//!< copy constructor
		RETURNTYPE _value; //!< distance from the pattern being classified to this training neighbour
		DIMTYPE _cls; //!< the training neighbour's class
	};
	DIMTYPE _k; //!< number of nearest neighbours voted over; set by set_k()
	DIMTYPE _nns_max_size; //!< cap on _nns's size while collecting neighbours: k, the number voted over
	void collect_neighbours(const PPattern &pattern); //!< fills _nns with the pattern's nearest training neighbours (shared by classify() and classify_scores())
	//! Inserts a candidate neighbour (\p value, \p cls) into the descending-sorted _nns, capped at
	//! _nns_max_size (the farthest entry is dropped once the cap is exceeded); used by collect_neighbours().
	void sort_in(const RETURNTYPE value, const DIMTYPE cls);
	typename std::list<Neighbour>::iterator iter; //!< scratch iterator used internally by sort_in() to walk _nns during insertion
	std::list<Neighbour> _nns; //!< implemented descending ... the closest neighbour is the last
	std::vector<RETURNTYPE> _cs_sc; //!< classify() score scratch (member to avoid per-pattern allocation)

	std::unique_ptr<DISTANCE> _distance; //!< distance metric narrowed to the trained subspace; allocated/renarrowed by train()
	PDataAccessor _da_train; //!< training data accessor cached by train() (kNN has no real training step); enumerated as neighbours by collect_neighbours()
};

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::Classifier_kNN(const Classifier_kNN& cknn) :
	_k(cknn._k),
	_nns_max_size(cknn._nns_max_size),
	_nns(cknn._nns)
{
	notify("Classifier_kNN constructor.");
	if(cknn._distance) _distance=clone_unique(*cknn._distance);
}




template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::train(const PDataAccessor da, const PSubset sub)
{
	assert(da);
	assert(sub);
	if(!_distance || _distance->get_n()<sub->get_n_raw()) _distance.reset(new DISTANCE(sub->get_n_raw()));
	_distance->narrow_to(sub);
	_da_train=da;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
void Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::collect_neighbours(const PPattern &pattern)
{
	// NOTE: iterates training data via its own RAII block range
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	assert(_da_train);
	assert(_distance);
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	assert(_distance->get_d()>0);
	typename DATAACCESSOR::PPattern p2;
	IDXTYPE s2,i2;
	DIMTYPE _feats=_da_train->getNoOfFeatures();
	RETURNTYPE val;



	_nns_max_size=_k; // the k closest entries are collected and voted over

	_nns.clear();
	for(DIMTYPE c_train=0;c_train<_da_train->getNoOfClasses();c_train++)
	{
		_da_train->setClass(c_train);
		for(const auto &blk : _da_train->blocks(TRAIN)) for(i2=0,p2=blk.data,s2=blk.size;i2<s2;i2++)
		{
			// p2[i2*_feats] is the beginning of the current pattern
			val=_distance->distance(pattern,&p2[i2*_feats]);
			sort_in(val,c_train);
		}
	}
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// the decision rule IS the classify_scores() rule - argmax of the per-class vote
	// fractions among the k nearest neighbours, first maximum wins (lowest class index
	// on ties, the convention shared by all score-deriving classifiers)
	if(!classify_scores(_cs_sc,pattern)) return false;
	cls=0;
	const DIMTYPE classes=_da_train->getNoOfClasses();
	for(DIMTYPE c=1;c<classes;c++) if(_cs_sc[c]>_cs_sc[cls]) cls=c; // first maximum wins
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	// per-class vote fractions among the k nearest training neighbours (coarse but valid
	// posteriors for ranking metrics); the neighbour list is only read, not consumed
	collect_neighbours(pattern);
	const DIMTYPE classes=_da_train->getNoOfClasses();
	scores.assign(classes,(RETURNTYPE)0.0);
	DIMTYPE taken=0;
	for(auto it=_nns.rbegin();it!=_nns.rend() && taken<_k;++it,++taken) // closest neighbour is the last
		scores[it->_cls]+=1.0;
	if(taken==0) return false;
	for(DIMTYPE c=0;c<classes;c++) scores[c]/=(RETURNTYPE)taken;
	return true;
}


template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
void Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::sort_in(const RETURNTYPE value, const DIMTYPE cls)
{
	iter=_nns.begin();
	while(iter!=_nns.end() && value < (*iter)._value) iter++;
	if(_nns.size()<_nns_max_size || iter!=_nns.begin()) {
		Neighbour tmp(value,cls);
		_nns.insert(iter,tmp);
		// NOTE: known limitation: when several neighbours are tied at the k-th distance, the list cap below
		//       drops the excess ones, so the vote depends on training-sample order at the tie boundary.
		//       This is the conventional kNN tie behaviour and is deterministic for a fixed data order.
		if(_nns.size()>_nns_max_size) _nns.pop_front();
	}
}


template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
std::ostream& Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::print(std::ostream& os) const 
{
	os << "Classifier_kNN(k=" << _k << ")";
	if(_distance) os << *_distance;
	return os;
}

} // namespace
#endif // FSTCLASSIFIERKNN_H ///:~
