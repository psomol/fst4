#ifndef FSTREGRESSORWEIGHTEDNEIGHBOURS_H
#define FSTREGRESSORWEIGHTEDNEIGHBOURS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    regressor_weighted_neighbours.hpp
   \brief   Experimental regressor, produces weighted average of the predicted variable over neighbours, weighted by distance
   \author  Pavel Pudil (pudil@fm.vse.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    2012 (last updated July 2026)
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
#ifndef M_PI
#define M_PI 3.14159265358979323846 //!< value of pi; POSIX \c \<cmath\> extension, provided here as a fallback since strict C++17 does not guarantee it
#endif
#include <list>
#include "regressor.hpp"
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

/*! \brief Implements experimental regressor that produces weighted average of the predicted
    variable over neighbours, weighted by distance

    A prediction is the Gaussian-kernel weighted average of the dependent variable over ALL
    training patterns (not the k nearest ones): weight(dist) falls off with the distance in the
    selected subspace, the kernel width being the standard deviation of the training set's
    pairwise distances times the multiplier set by set_kernel_multiplier().

    EXPERIMENTAL means: the method is offered for exploration, not as a tuned or validated
    regressor. Specifically -- the kernel width is derived from the data rather than
    cross-validated; every training pattern contributes to every prediction, so there is no
    locality parameter to tune; and no distance normalization across features is applied beyond
    whatever the DISTANCE template argument and the data scaler do.

    \note The DEPENDENT VARIABLE is by default the CLASS LABEL, not a data column: with the
	  default (IDXTYPE)-1 sentinel the class index of each training pattern is the value
	  being averaged ("regression simulating classification"), and test() then counts a
	  prediction within 0.5 of the true label as correct, so its result reads like an
	  accuracy. set_dependent_index() switches to a genuine data column, in which case
	  train() excludes that column from the distance computation and test() returns the
	  negated mean absolute error (FST4 always maximizes).
    \warning COST: train() computes ALL pairs of training-set distances, i.e. O(N^2) distance
	  evaluations in the subset dimensionality, and keeps one value per pair in memory.
	  predict() costs O(N) per pattern, so one test() pass over M test patterns costs
	  O(M*N). This regressor is meant for small datasets.
    \note Degenerate data (all training patterns identical, or a kernel width of zero) leads to
	  divisions by zero that are only caught by assertions -- see train() and predict().
*/
template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
class Regressor_Weighted_Neighbours : public Clonable_Impl<Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>,Regressor<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Regressor<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Regressor_Weighted_Neighbours,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Regressor
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor; //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to the feature subset being trained/evaluated on
	typedef typename DATAACCESSOR::PPattern PPattern; //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Regressor_Weighted_Neighbours() : _dep_var_idx((IDXTYPE)-1), _dist_mean(0), _dist_stddev(0), _kernel_multiplier(1) {notify("Regressor_Weighted_Neighbours constructor.");}
	virtual ~Regressor_Weighted_Neighbours() {notify("Regressor_Weighted_Neighbours destructor.");}

	void set_kernel_multiplier(const RETURNTYPE m) {assert(m > 0); _kernel_multiplier = m;} //!< scales the Gaussian kernel width used by get_weight(); width = distance stddev * \p m (must be > 0)
	RETURNTYPE get_kernel_size() const {return _dist_stddev * _kernel_multiplier;} //!< the effective kernel width (training-set pairwise-distance stddev times the multiplier set by set_kernel_multiplier())
	void set_dependent_index(const IDXTYPE depvaridx) {_dep_var_idx = depvaridx;} //!< data column to predict; the (IDXTYPE)-1 default/sentinel selects the class label instead ("regression simulating classification")

	virtual bool predict(RETURNTYPE &depvar, const PPattern &pattern) override;  // predicts dependent variable for pattern
	//! With RWN there is actually no training in the usual sense: this caches \p da (by shared
	//! pointer -- the accessor must outlive the predictions) and computes the distance kernel's
	//! width (mean/stddev of all pairwise training-set distances in \p sub's subspace) used by
	//! get_weight(). O(N^2) distances; see the class warning. When a data column was chosen by
	//! set_dependent_index(), that column is excluded from the distance -- as in Regressor_Ridge,
	//! a feature must not help predict itself.
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	//! Over the designated test part: the fraction of predictions within 0.5 of the true class
	//! label (an accuracy) in the default class-label mode, or the negated mean absolute error
	//! when a data column is the dependent variable. Returns false if the test part is empty.
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override;
	virtual bool test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec) override; // r2/neg-rmse/neg-mae over the same dependent-variable convention as test()



	virtual std::ostream& print(std::ostream& os) const override;
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	Regressor_Weighted_Neighbours(const Regressor_Weighted_Neighbours& cknn); //!< copy-constructor (clones _distance if allocated; scalar members are value-copied)
protected:
	virtual RETURNTYPE get_weight(RETURNTYPE dist); //!< the (unnormalized) Gaussian kernel weight for a neighbour at distance \p dist, using get_kernel_size() as the kernel width
	DIMTYPE _dep_var_idx; //!< dependent-variable data column, or the (IDXTYPE)-1 sentinel meaning "use the class label"; set by set_dependent_index()
	std::unique_ptr<DISTANCE> _distance; //!< distance metric narrowed to the trained subspace, used by predict()/train() to weigh training neighbours
	RETURNTYPE _dist_mean; //!< mean of all pairwise training-set distances (in the trained subspace), computed by train()
	RETURNTYPE _dist_stddev; //!< standard deviation of all pairwise training-set distances, computed by train(); the kernel width before set_kernel_multiplier() scaling
	RETURNTYPE _kernel_multiplier; //!< scales _dist_stddev into the kernel width returned by get_kernel_size(); set by set_kernel_multiplier(), default 1
	PDataAccessor _da_train; //!< training data accessor cached by train() (RWN has no real training step); read back by predict()/test() to enumerate training neighbours
};

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::Regressor_Weighted_Neighbours(const Regressor_Weighted_Neighbours& rwn) :
	_dep_var_idx(rwn._dep_var_idx),
	_dist_mean(rwn._dist_mean),
	_dist_stddev(rwn._dist_stddev),
	_kernel_multiplier(rwn._kernel_multiplier)
{
	notify("Regressor_Weighted_Neighbours copy-constructor.");
	if(rwn._distance) _distance=clone_unique(*rwn._distance);
}




template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::train(const PDataAccessor da, const PSubset sub)
{
	assert(da);
	assert(sub);
	if(!_distance || _distance->get_n()<sub->get_n_raw()) _distance.reset(new DISTANCE(sub->get_n_raw()));
	// No dependent COLUMN in range (the default sentinel, or an out-of-range index): the class
	// label is the dependent variable and the distance uses the subset as it stands. Otherwise
	// the distance must be measured WITHOUT the dependent column - a copy of the subset with
	// that feature deselected is narrowed to instead, leaving the caller's subset untouched.
	if(_dep_var_idx >= sub->get_n_raw()) {
		_dep_var_idx = (IDXTYPE)-1;
		_distance->narrow_to(sub);
	} else {
		PSubset dsub(new SUBSET(*sub)); // clone
		dsub->deselect_raw(_dep_var_idx);
		_distance->narrow_to(dsub);
	}
	_da_train=da;

	// recompute kernel weight params
	assert(_distance);
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	assert(_distance->get_d()>0);
	typename DATAACCESSOR::PPattern p0,p1;
	IDXTYPE s0,i0, s1,i1;
	DIMTYPE _feats=_da_train->getNoOfFeatures();
	RETURNTYPE dist;
	
	_dist_mean = 0;
	IDXTYPE _dist_count = 0;
	typename std::list<RETURNTYPE> _dists;
	typename std::list<RETURNTYPE>::const_iterator diter;
	

	for(DIMTYPE c_train0=0;c_train0<_da_train->getNoOfClasses();c_train0++)
	{
		_da_train->setClass(c_train0);
		for(const auto &blk0 : _da_train->blocks(TRAIN)) for(i0=0,p0=blk0.data,s0=blk0.size;i0<s0;i0++)
		{
			    
			for(DIMTYPE c_train1=0;c_train1<_da_train->getNoOfClasses();c_train1++)
			{
				_da_train->setClass(c_train1);
				for(const auto &blk1 : _da_train->blocks(TRAIN)) for(i1=0,p1=blk1.data,s1=blk1.size;i1<s1;i1++)
				{
					// p1[i1*_feats] is the beginning of the current pattern
					dist=_distance->distance(&p0[i0*_feats],&p1[i1*_feats]);
					// zero distances are skipped: they include every pattern
					// against itself, and would otherwise drag the kernel
					// width towards 0. Exact duplicates in the data are
					// skipped as well, as a side effect.
					if(dist > 0) {
					    _dists.push_back(dist);
					    _dist_mean += dist;
					    _dist_count++;
					}
				}
			}
			_da_train->setClass(c_train0);
		}
	}
	// _dist_count==0 means every pair of training patterns coincides (or there is only one
	// pattern): there is no scale to build a kernel from. With assertions disabled this
	// divides by zero and leaves the kernel width NaN, which propagates into every
	// prediction - the degenerate case named in the class documentation.
	assert(_dist_count > 0);
	assert(_dist_count == _dists.size());
	_dist_mean /= (RETURNTYPE)_dist_count;
	_dist_stddev = 0;
	for(diter = _dists.begin(); diter != _dists.end(); diter++) {
	       _dist_stddev += (*diter - _dist_mean) * (*diter - _dist_mean);
	}
	_dist_stddev = sqrt(_dist_stddev / (RETURNTYPE)_dist_count); // population divisor, not _dist_count-1
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
RETURNTYPE Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::get_weight(RETURNTYPE dist)
{
	// Gaussian kernel of width get_kernel_size(). The normalizing factor 1/(stddev*sqrt(2pi))
	// is the same for every neighbour and thus cancels in predict()'s weighted average; it is
	// kept so that the returned value is a proper density. A zero kernel width (identical
	// training patterns, see train()) divides by zero here.
	const RETURNTYPE stddev = _dist_stddev * _kernel_multiplier;
	return exp( -(dist*dist)/(2.0*stddev*stddev) ) / (stddev * sqrt(2.0*M_PI));
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::predict(RETURNTYPE &depvar, const PPattern &pattern)
{
	// NOTE: iterates training data via its own RAII block range
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	assert(_da_train);
	assert(_distance);
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	assert(_distance->get_d()>0);
	typename DATAACCESSOR::PPattern p2;
	IDXTYPE s2,i2;
	const DIMTYPE _feats=_da_train->getNoOfFeatures();
	RETURNTYPE dist;
	RETURNTYPE weight;
	
	RETURNTYPE predval = 0;
	RETURNTYPE weightsum = 0;
	
	

	
	for(DIMTYPE c_train=0;c_train<_da_train->getNoOfClasses();c_train++)
	{
		_da_train->setClass(c_train);
		for(const auto &blk : _da_train->blocks(TRAIN)) for(i2=0,p2=blk.data,s2=blk.size;i2<s2;i2++)
		{
			RETURNTYPE depval;
			if(_dep_var_idx == (IDXTYPE)-1) {
			    depval = c_train; // the documented primary path: class label = dependent variable ("regression simulating classification", the -1 sentinel)
			} else {
			    depval = p2[i2*_feats + _dep_var_idx];
			}
			// p2[i2*_feats] is the beginning of the current pattern
			dist=_distance->distance(pattern,&p2[i2*_feats]);

			weight = get_weight( dist );
			predval += weight * depval;
			weightsum += weight;
		}
	}
	// weightsum==0 requires every training pattern to be so far away that its Gaussian weight
	// underflows to 0 (or a NaN kernel width from a degenerate train()); with assertions
	// disabled the prediction becomes NaN or infinite rather than failing
	assert(weightsum != 0);
	depvar = predval / weightsum;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	assert(da);
	assert(_da_train);
	assert(_distance);
	assert(da->getNoOfFeatures()==_da_train->getNoOfFeatures());
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	typename DATAACCESSOR::PPattern p1;
	IDXTYPE s1,i1;
	IDXTYPE count=0;
	RETURNTYPE errsum = 0; // despite the name: a HIT count in class-label mode, a negated error sum otherwise
	DIMTYPE _feats=da->getNoOfFeatures();
	RETURNTYPE predval;
	

		
	for(DIMTYPE c_test=0;c_test<da->getNoOfClasses();c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p1=blk.data; s1=blk.size;
			for(i1=0;i1<s1;i1++)
			{
				RETURNTYPE depval;
				if(_dep_var_idx == (IDXTYPE)-1) {
				    depval = c_test; // the documented primary path: class label = dependent variable ("regression simulating classification", the -1 sentinel)
				} else {
				    depval = p1[i1*_feats + _dep_var_idx];
				}

				// predict() runs its own block loop over the TRAIN part of
				// _da_train, which may be this very accessor - hence the setClass()
				// restore after each block below
				if(!predict(predval, &p1[i1*_feats])) return false;
				
				if(_dep_var_idx == (IDXTYPE)-1) {
					if( std::abs( depval - predval ) < 0.5 ) errsum += 1; // simulate classification (std:: qualified: bare abs(double) risks the int overload)
				} else {
					errsum += - std::abs( depval - predval ); // sum negatives of prediction errors (FST maximizes; std:: qualified: bare abs(double) risks the int overload)
				}
				++count;
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as _da_train
		}
	}
	if(count==0) return false;
	result = (RETURNTYPE)errsum/(RETURNTYPE)count;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::test_metric(RETURNTYPE &result, const PDataAccessor da, const Metric_Spec<RETURNTYPE> &spec)
{
	// the test() loop, collecting (true value, prediction) pairs for r2/neg-rmse/neg-mae;
	// the dependent-variable convention (data column, or class label when _dep_var_idx==-1) is test()'s
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	if(spec.is_default()) return test(result,da); // the default metric IS test()'s own measure
	if(!spec.is_regression_metric()) return false; // classification metrics do not apply to regression stacks
	assert(da);
	assert(_da_train);
	assert(_distance);
	typename DATAACCESSOR::PPattern p1;
	IDXTYPE s1,i1;
	DIMTYPE _feats=da->getNoOfFeatures();
	RETURNTYPE predval;
	std::vector<std::pair<RETURNTYPE,RETURNTYPE>> pairs;

	for(DIMTYPE c_test=0;c_test<da->getNoOfClasses();c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) { p1=blk.data; s1=blk.size;
			for(i1=0;i1<s1;i1++)
			{
				RETURNTYPE depval;
				if(_dep_var_idx == (IDXTYPE)-1) depval = c_test; // the documented primary path: class label = dependent variable ("regression simulating classification", the -1 sentinel)
				else depval = p1[i1*_feats + _dep_var_idx];
				if(!predict(predval, &p1[i1*_feats])) return false;
				pairs.push_back(std::make_pair(depval,predval));
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as _da_train
		}
	}
	return metric_from_predictions<RETURNTYPE>(result,spec.metric,pairs);
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
std::ostream& Regressor_Weighted_Neighbours<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::print(std::ostream& os) const
{
	os << "Regressor_Weighted_Neighbours()";
	if(_distance) os << *_distance;
	return os;
}

} // namespace
#endif // FSTREGRESSORWEIGHTEDNEIGHBOURS_H ///:~
