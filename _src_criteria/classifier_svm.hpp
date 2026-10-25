#ifndef FSTCLASSIFIERSVM_H
#define FSTCLASSIFIERSVM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    classifier_svm.hpp
   \brief   Wraps external Support Vector Machine implementation (in LibSVM) to serve as FST4 classifier
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
#include <list>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
#include "classifier.hpp"
#include "svm.h"

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

// NOTE: this is the proxy to LIBSVM external library
//       therefore some FST coding conventions can not
//       be strictly kept

/*! \brief Wraps external Support Vector Machine implementation (in LibSVM) to serve as FST4 classifier
    \note clone() yields an UNTRAINED clone (LibSVM model structures cannot be deep-copied);
          using an untrained clone fails loudly in classify()/test() - call train() first.
    \note LibSVM is NOT distributed with FST4. Obtain svm.cpp and svm.h (version 3.37, or any
          version compatible with the public LibSVM API used here) from the LibSVM site and
          place them into ./libsvm/ before building anything that includes this header;
          libsvm/readme.txt in the FST4 distribution states the exact steps.

 \warning LIBSVM especially with LINEAR kernel seems to have occassional problems with certain C values on certain datasets and may freeze.
          (Other kernels are more stable but not completely immune to this problem.) This is a problem outside FST4.
          Classifier_Linear_SVM and Classifier_RBF_SVM are FST4's own, dependency-free alternatives.
*/

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_LIBSVM : public Clonable_Impl<Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,CLONE_TO_STATELESS,CLONE_INHERIT,CLONE_WEAK> {
public:
	/*! parent_impl: clone() forwards to stateless_clone() (LibSVM model structures cannot be cloned,
	    so clones are UNTRAINED - which is what makes threaded wrapper usage possible at all, as the
	    wrapper always train()s a clone before using it); stateless_clone() uses the weak
	    copy-constructor; sharing_clone() throws in Classifier. Using an untrained clone fails
	    loudly in classify()/test(). */
	typedef Clonable_Impl<Classifier_LIBSVM,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,CLONE_TO_STATELESS,CLONE_INHERIT,CLONE_WEAK> parent_impl;
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset;            //!< shared pointer to the (fixed) feature subset being trained/evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR
	Classifier_LIBSVM();
	virtual ~Classifier_LIBSVM() {notify("Classifier_LIBSVM destructor."); cleanup();}

	//! \brief Must be called (once, or again after a dataset-shape change) before train()/test().
	//! Discards any previously allocated buffers (cleanup()), records the new dataset shape
	//! (pattern/class/feature counts) from \p da, then pre-allocates the LibSVM problem/parameter
	//! transfer buffers sized to it (allocate()).
	void initialize(const PDataAccessor da);
protected:
	//! Pre-allocates the svm_problem/svm_parameter transfer buffers sized to the current
	//! _all_patterns/_classes/_features (the caller -- initialize() or the weak copy-constructor --
	//! must set these first). Throws fst_error (via cleanup()) on malloc failure.
	void allocate();
	//! Frees every LibSVM-related buffer and resets the shape counters to 0; idempotent
	//! (each pointer is individually guarded) -- safe to call from allocate()'s failure
	//! handler, initialize(), and the destructor.
	void cleanup();
public:
	void set_parameter_C(double newC) {parameters.C = newC;}                      //!< sets the LibSVM regularization parameter C
	void set_parameter_gamma(double newgamma) {parameters.gamma = newgamma;}      //!< sets the LibSVM kernel gamma (RBF/POLY/SIGMOID)
	void set_parameter_coef0(double newcoef0) {parameters.coef0 = newcoef0;}      //!< sets the LibSVM kernel coef0 (POLY/SIGMOID)
	void set_kernel_type(int kernel_type) {parameters.kernel_type = kernel_type;} //!< LINEAR/POLY/RBF/SIGMOID -- see svm.h for admissible values
	double get_parameter_C() const {return parameters.C;}           //!< current LibSVM regularization parameter C
	double get_parameter_gamma() const {return parameters.gamma;}   //!< current LibSVM kernel gamma (RBF/POLY/SIGMOID)
	double get_parameter_coef0() const {return parameters.coef0;}   //!< current LibSVM kernel coef0 (POLY/SIGMOID)
	int get_kernel_type() const {return parameters.kernel_type;} //!< see svm.h for admissible values

	//! classifies \p pattern, returns the respective class index in \p cls. Throws fst_error when no
	//! model has been trained yet (in particular on a fresh clone, which never inherits the trained state).
	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;
	//! LibSVM decision values for ranking metrics (AUC): binary = signed decision value, K>2 = per-class
	//! one-vs-one aggregation; deterministic, unlike LibSVM's probability mode. Throws fst_error when
	//! untrained; returns false when the trained model's class labels cannot be mapped onto [0,classes).
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override;
	//! learns from the designated TRAIN part of the data, restricted to the features in \p sub.
	//! Requires a preceding initialize() for the same dataset shape. Returns false when LibSVM
	//! returns no model; throws fst_error on transfer failures (the object is cleanup()'d first).
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	//! estimates accuracy (fraction of correctly classified patterns) over the designated TEST part
	//! of the data, using the subset train() was last called with. Throws fst_error when untrained.
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override;

	/*! \brief Searches for better SVM hyperparameters C, gamma and coef0 and adopts the best set found.

	    A space-filling random search, not a grid: \p max_points candidate parameter sets are evaluated
	    one after another; each is chosen by drawing \p max_throws random candidates and keeping the one
	    whose distance to the already-evaluated candidates (normalized per axis, weighted by their
	    accuracy so that the neighbourhood of bad candidates is explored less) is largest. A candidate is
	    evaluated by train()+test() over the ENTIRE split loop of \p da with the fixed feature subset
	    \p sub; its value is the mean accuracy over the splits. The current parameter values are
	    evaluated first and take part in the competition.

	    \param da data accessor; initialize() must have been called for it. Its split loop is traversed
	              once per candidate, so the cost is (max_points+1) * splits * (train+test).
	    \param sub the feature subset to optimize for; must be non-empty and stays fixed.
	    \param max_points number of candidate parameter sets to evaluate (>0).
	    \param max_throws number of random draws per candidate to choose from (>0); larger values spread
	                      the candidates more evenly at negligible cost (no training is involved).
	    \param lgC_min,lgC_max,lggamma_min,lggamma_max,lgcoef0_min,lgcoef0_max search bounds, given as
	           NATURAL logarithms of the parameter despite the "lg" in the names: a candidate is
	           exp(uniform(min,max)). The defaults therefore span C in [e^-5,e^9], gamma in
	           [e^-15,e^3], coef0 in [e^-2,e^5]. Each max must exceed its min. gamma is only meaningful
	           for RBF/POLY/SIGMOID and coef0 only for POLY/SIGMOID -- the unused ones are still drawn
	           and set, but they do not enter the candidate distance and do not affect the model.
	    \param os stream the progress report is written to (one line per candidate, plus a before/after
	              summary); pass a null stream to silence it.
	    \return true when the search completed; false as soon as a train() or test() call fails, in
	            which case the object keeps the parameters of the candidate that was being evaluated.

	    \note On success the object's C, gamma and coef0 are left set to the best candidate, but the
	          trained model is still the one from the LAST candidate evaluated -- call train() again
	          before classify()/test() to use the optimized parameters.
	    \note Randomness comes from fst_rand(), so a given seed reproduces the whole search. Not
	          reentrant (it uses a member iterator), and it must not run concurrently on one object.
	    \warning LIBSVM LINEAR kernel seems to have problems with certain C values on certain datasets.
	             If the optimization process seemingly freezes, try narrower lgC_min and lgC_max.
	             Especially the lower bound seems important to be increased.
	*/
	bool optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points=100, const int max_throws=100, const double lgC_min=-5, const double lgC_max=9, const double lggamma_min=-15, const double lggamma_max=3, const double lgcoef0_min=-2, const double lgcoef0_max=5, std::ostream& os=std::cout);


	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_LIBSVM(kernel="<<get_kernel_type()<<")"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the weak copy-constructor
	//! \brief Weak copy-constructor backing stateless_clone() (see parent_impl above): copies
	//! configuration (parameters, shape counters, svm_class_weighing, the narrowing _index) but
	//! deliberately NOT the trained LibSVM model/problem structures -- it allocate()s fresh,
	//! empty transfer buffers instead. The resulting clone is therefore UNTRAINED; classify()/
	//! test() throw until train() is called on it.
	Classifier_LIBSVM(const Classifier_LIBSVM& csvm, int);
protected:
	// the following to be initialized using initialize() based on dataaccessor info
	IDXTYPE _all_patterns;             //!< total pattern count across all classes, set by initialize()
	DIMTYPE _classes;                  //!< class count, set by initialize()
	DIMTYPE _features;                 //!< raw (full, unnarrowed) feature count, set by initialize()
	struct svm_problem problem;        //!< LibSVM training data (problem.x/y), rebuilt by train() from the designated TRAIN blocks
	struct svm_parameter parameters;   //!< LibSVM training parameters (C, gamma, kernel_type, ...) -- see set_parameter_*()/set_kernel_type()
	struct svm_model *model;           //!< the trained LibSVM model, or nullptr before the first successful train()
	struct svm_node *onepattern;       //!< single-pattern LibSVM transfer buffer, filled by fill_onepattern() for classify()/classify_scores()

	bool svm_class_weighing;           //!< vestigial class-weighing switch: set false by the constructor and never read -- allocate() unconditionally forces parameters.nr_weight=0 regardless of this flag

	//! Nested class to hold parameter candidates in the course of optimize_parameters() run
	class ParameterSet {
	public:
		ParameterSet(): _crit_value(0.0), _Cpar(1.0), _gamma(1.0), _coef0(1.0) {}
		ParameterSet(const RETURNTYPE crit_value, const double C, const double gamma, const double coef0) : _crit_value(crit_value), _Cpar(C), _gamma(gamma), _coef0(coef0) {} //!< constructs a fully-specified candidate
		ParameterSet(const ParameterSet& ps) : _crit_value(ps._crit_value), _Cpar(ps._Cpar), _gamma(ps._gamma), _coef0(ps._coef0) {} //!< copy-constructor
		void operator=(const ParameterSet& ps) {_crit_value=ps._crit_value; _Cpar=ps._Cpar; _gamma=ps._gamma; _coef0=ps._coef0;} //!< copy-assignment
		RETURNTYPE _crit_value; //!< mean criterion value (accuracy, averaged over the accessor's split loop) this candidate achieved, or 0.0 before evaluation
		double _Cpar; //!< candidate C value (named to avoid clashing with the enclosing class's parameters.C)
		double _gamma; //!< candidate gamma value
		double _coef0; //!< candidate coef0 value
	};
	typedef std::list<ParameterSet> PARAMSETLIST;               //!< the growing list of parameter candidates evaluated so far, in evaluation order

private:
	std::unique_ptr<DIMTYPE[]> _index;   //!< subset position -> raw feature index map, filled by train() from sub->selected_features(); regrown on demand
	DIMTYPE _subfeatures;                //!< current subset size (number of valid entries in _index[]), set by train()
	std::vector<double> _dec;            //!< classify_scores() one-vs-one decision-value scratch (member to avoid per-pattern allocation)
	//! transfers the pattern's nonzero selected features into onepattern (the shared classify()/classify_scores() transfer loop)
	void fill_onepattern(const PPattern &pattern)
	{
		assert(_index);
		assert(_subfeatures>0 && _subfeatures<=_features);
		DIMTYPE nzfeat=0; // no. of nonzero features processed in current pattern
		for(DIMTYPE f=0;f<_subfeatures;f++)
		{
			const DIMTYPE ftmp=_index[f]; // (full set) feature index
			const double tmp=pattern[ftmp]; // feature value
			if(tmp!=0.0) {
				onepattern[nzfeat].index=ftmp+1;
				onepattern[nzfeat].value=tmp;
				nzfeat++;
			}
		}
		onepattern[nzfeat].index=-1;
	}
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_LIBSVM(const Classifier_LIBSVM& csvm, int) : // copy-constructor
	_all_patterns(csvm._all_patterns),
	_classes(csvm._classes),
	_features(csvm._features),
	svm_class_weighing(csvm.svm_class_weighing),
	_subfeatures(csvm._subfeatures)
{
	notify("Classifier_LIBSVM (weak) copy-constructor.");
	// does not copy LibSVM-defined structures, except for 'parameters'
	// other-than 'parameters' LibSVM structures are only pre-allocated using allocate()
	parameters.svm_type=csvm.parameters.svm_type;
	parameters.kernel_type=csvm.parameters.kernel_type;
	parameters.degree=csvm.parameters.degree;
	parameters.gamma=csvm.parameters.gamma;
	parameters.coef0=csvm.parameters.coef0;
	parameters.cache_size=csvm.parameters.cache_size;
	parameters.eps=csvm.parameters.eps;
	parameters.C=csvm.parameters.C;
	parameters.nr_weight=csvm.parameters.nr_weight;
	parameters.nu=csvm.parameters.nu;
	parameters.p=csvm.parameters.p;
	parameters.shrinking=csvm.parameters.shrinking;
	parameters.probability=csvm.parameters.probability;
	// pre-null everything allocate()/cleanup() touch: on an allocation failure cleanup()
	// would otherwise free() garbage pointers of this partially constructed object
	problem.x=nullptr;
	problem.y=nullptr;
	onepattern=nullptr;
	parameters.weight_label=nullptr;
	parameters.weight=nullptr;
	model=nullptr;
	allocate();
	if(csvm.parameters.weight_label) memcpy((void *)parameters.weight_label,(void *)csvm.parameters.weight_label,(long)(_classes) * sizeof(int));
	if(csvm.parameters.weight) memcpy((void *)parameters.weight,(void *)csvm.parameters.weight,(long)(_classes) * sizeof(double));
	if(csvm._index) {_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(csvm._index).get(),sizeof(DIMTYPE)*_features);}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_LIBSVM()
{
	notify("Classifier_LIBSVM constructor.");
	_all_patterns=0;
	_classes=0;
	_features=0;
	problem.x=nullptr;
	problem.y=nullptr;
	onepattern=nullptr;
	parameters.weight_label=nullptr;
	parameters.weight=nullptr;
	model=nullptr;
	_subfeatures=0;

	// default parameters
	parameters.svm_type=C_SVC;
	parameters.kernel_type=RBF; //LINEAR;
	parameters.degree=2;
	parameters.gamma=/*standard_gamma=*/1.0;
	parameters.coef0=1.0;
	parameters.cache_size=10;
	parameters.eps=0.001;
	parameters.C=/*standard_C=*/1.0;
	parameters.nr_weight=0;
	parameters.weight_label=nullptr; // just sharing a pointer
	parameters.weight=nullptr; // just sharing a pointer
	parameters.nu=1.0;
	parameters.p=1.0;
	parameters.shrinking=0;
	parameters.probability=0;
	svm_class_weighing=false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points, const int max_throws, const double lgC_min, const double lgC_max, const double lggamma_min, const double lggamma_max, const double lgcoef0_min, const double lgcoef0_max, std::ostream& os)
{
	typename PARAMSETLIST::iterator iter; // cursor over _paramlist for the scan loops below
	// NOTE: assumes Classifier_LIBSVM::initialize() had been called for this da
	//       assumes initial C, gamma, and coef0 have been set using set_parameter_*(), or that
	//       the default values are OK (the initial values are the first candidate evaluated)
	assert(da);
	assert(sub);
	assert(sub->get_d()>0);
	assert(max_points>0);
	assert(max_throws>0);
	assert(lgC_max>lgC_min);
	assert(lggamma_max>lggamma_min);
	assert(lgcoef0_max>lgcoef0_min);
	const double lgCrange=lgC_max-lgC_min;
	const double lggammarange=lggamma_max-lggamma_min;
	const double lgcoef0range=lgcoef0_max-lgcoef0_min;
	PARAMSETLIST _paramlist;
	ParameterSet tmpset, bestset;
	// evaluate the initial (current) parameter set
	RETURNTYPE val, maxval, firstval;
	RETURNTYPE result=0.0;
	RETURNTYPE cnt=0.0;
	for(bool b=da->getFirstSplit();b==true;b=da->getNextSplit()) {
		if(!train(da,sub)) return false;
		if(!test(val,da)) return false;
		result+=val;
		cnt+=1.0;
	}
	firstval=maxval=val=result/(RETURNTYPE)cnt;
	{
		std::ostringstream sos;
		sos << "SVM before optimization: accuracy=" << val << ", C=" << get_parameter_C();
		switch(get_kernel_type()) {
			case LINEAR:  sos << std::endl; break;
			case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
			case POLY:    
			case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
		}
		syncout::print(os,sos);
	}
	_paramlist.push_back(ParameterSet(val,get_parameter_C(),get_parameter_gamma(),get_parameter_coef0()));
	for(int i=0;i<max_points;i++)
	{ // find and evaluate next parameter set candidate
		double maxdistmin=0.0;
		for(int j=0;j<max_throws;j++)
		{ // find next parameter set candidate
			// the bounds are natural logarithms: a candidate is exp(uniform(min,max))
			tmpset._Cpar =exp(lgC_min+(((double)fst_rand()*(double)(lgC_max-lgC_min))/(double)FST_RAND_MAX));
			tmpset._gamma=exp(lggamma_min+(((double)fst_rand()*(double)(lggamma_max-lggamma_min))/(double)FST_RAND_MAX));
			tmpset._coef0=exp(lgcoef0_min+(((double)fst_rand()*(double)(lgcoef0_max-lgcoef0_min))/(double)FST_RAND_MAX));
			
			double distmin=-1.0; // potential new candidate (weighted) distance to already evaluated candidates
			for(iter=_paramlist.begin();iter!=_paramlist.end();iter++)
			{
				double dist=0.0;
				assert(get_kernel_type()==LINEAR || get_kernel_type()==POLY || get_kernel_type()==RBF || get_kernel_type()==SIGMOID);
				switch(get_kernel_type()) {
					case LINEAR:  dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)); break;
					case RBF:     dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)+((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)*((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)); break;
					case POLY:    
					case SIGMOID: dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)+((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)*((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)+((log(tmpset._coef0)-log(iter->_coef0))/lgcoef0range)*((log(tmpset._coef0)-log(iter->_coef0))/lgcoef0range)); break;
				}
				// candidates more distant from all others are preferred
				// distances are weighed to penalize closeness to worse-performing candidates
				// NOTE: the weight is the candidate's accuracy, so an already-evaluated candidate
				//       of accuracy 0 nullifies every distance measured against it (with asserts
				//       disabled this makes the draw for the next point effectively arbitrary)
				assert(iter->_crit_value>0.0);
				if(distmin<0 || dist*iter->_crit_value<distmin) distmin=dist*iter->_crit_value;
			}
			assert(distmin>=0.0);
			if(distmin>maxdistmin)
			{ // better (more distant from those already tested, esp. from the bad ones) parameter set candidate found
				maxdistmin=distmin;
				bestset=tmpset;
			}
		}
		// evaluate the chosen candidate and add it to list
		set_parameter_C(bestset._Cpar);
		set_parameter_gamma(bestset._gamma);
		set_parameter_coef0(bestset._coef0);
		assert(svm_check_parameter(&problem,&parameters)==nullptr);
		result=0.0;
		cnt=0.0;
		for(bool b=da->getFirstSplit();b==true;b=da->getNextSplit()) {
			if(!train(da,sub)) return false;
			if(!test(val,da)) return false;
			result+=val;
			cnt+=1.0;
		}
		val=result/(RETURNTYPE)cnt;
		bestset._crit_value=val;
		_paramlist.push_back(bestset);
		{
			std::ostringstream sos;
			sos << i+1 << ". ";
			if(val>maxval) {maxval=val; sos << "BEST accuracy=";} else sos << "test accuracy=";
			sos << bestset._crit_value << ", C=" << get_parameter_C();
			switch(get_kernel_type()) {
				case LINEAR:  sos << std::endl; break;
				case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
				case POLY:    
				case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
			}
			syncout::print(os,sos);
		}
	}
	// identify the best parameter set among all tested ones
	assert(!_paramlist.empty());
	iter=_paramlist.begin();
	bestset=(*iter);
	while(iter!=_paramlist.end())
	{
		if(iter->_crit_value>bestset._crit_value) bestset=(*iter);
		iter++;
	}
	set_parameter_C(bestset._Cpar);
	set_parameter_gamma(bestset._gamma);
	set_parameter_coef0(bestset._coef0);
	{
		std::ostringstream sos;
		sos << "SVM after optimization: accuracy=" << bestset._crit_value << ", C=" << get_parameter_C();
		switch(get_kernel_type()) {
			case LINEAR:  sos << std::endl; break;
			case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
			case POLY:    
			case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
		}
		sos << "Accuracy difference=" << bestset._crit_value-firstval << std::endl << std::endl;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::allocate()
{
	try {
		// NOTE: cleanup() must NOT be called here - it would destroy the original data
		//       when allocate() runs from the weak copy-constructor
		assert(_all_patterns>0);
		assert(_classes>1);
		assert(_features>0);
		
		problem.y =               (double *)          malloc((std::size_t)(_all_patterns) * sizeof(double));
		problem.x =               (struct svm_node **)malloc((std::size_t)(_all_patterns) * sizeof(struct svm_node *));
		parameters.weight_label = (int *)             malloc((std::size_t)(_classes) * sizeof(int));
		parameters.weight =       (double *)          malloc((std::size_t)(_classes) * sizeof(double));
		onepattern =              (struct svm_node *) malloc((std::size_t)(_features+1) * sizeof(struct svm_node));
		// malloc does not throw: the failure must be tested explicitly, or the catch
		// below is dead code and a null problem.x gets dereferenced further down
		if(!problem.y || !problem.x || !parameters.weight_label || !parameters.weight || !onepattern)
			throw fst_error("Classifier_LIBSVM allocation problem (malloc failed).");
		for(IDXTYPE p=0;p<_all_patterns;p++) problem.x[p]=nullptr;
		// pre-allocate all problem.x[] arrays to max size, to prevent re-allocation speed efficiency problems
		// - WARNING! this leads to uneconomical use of memory
		for(IDXTYPE p=0;p<_all_patterns;p++) {
			problem.x[p] = (struct svm_node *) malloc((long)(_features+1) * sizeof(struct svm_node));
			if(!problem.x[p]) throw fst_error("Classifier_LIBSVM allocation problem (malloc failed).");
		}

		// class penalty weighing disabled
		parameters.nr_weight=0;
		problem.l = _all_patterns;		
		assert(svm_check_parameter(&problem,&parameters)==nullptr);
	}
	catch (...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM allocation problem.");
	}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	assert(da);
	try {
		cleanup();
		_all_patterns = da->getClassSizeSum(); assert(_all_patterns>0);
		_classes = da->getNoOfClasses(); assert(_classes>1);
		_features = da->getNoOfFeatures(); assert(_features>0);
		allocate();
	}
	catch (...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM initialization problem.");
	}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::cleanup()
{
	// NOTE: called from destructor
		if(problem.x!=nullptr) {
			for(IDXTYPE p=0;p<_all_patterns;p++) if(problem.x[p]!=nullptr) {free(problem.x[p]); problem.x[p]=nullptr;}
			free(problem.x); problem.x=nullptr; // the array only, not the actual pointers
		}
		if(problem.y!=nullptr) free(problem.y);
		problem.y=nullptr;
		if(onepattern!=nullptr) free(onepattern);
		onepattern=nullptr;
		if(parameters.weight_label!=nullptr) free(parameters.weight_label);
		parameters.weight_label=nullptr;
		if(parameters.weight!=nullptr) free(parameters.weight);
		parameters.weight=nullptr;
		if(model!=nullptr) {svm_free_and_destroy_model(&model); model=nullptr;}
		// _index is sized after _features, so it must be dropped here too: a re-initialize()
		// on a larger dataset would otherwise keep (and overrun) the smaller buffer
		_index.reset();

		_all_patterns=0;
		_classes=0;
		_features=0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	assert(da);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(_all_patterns==da->getClassSizeSum());
	assert(problem.x!=nullptr);
	assert(problem.y!=nullptr);
	assert(onepattern!=nullptr);
	assert(parameters.weight_label!=nullptr);
	assert(parameters.weight!=nullptr);
	assert(svm_check_parameter(&problem,&parameters)==nullptr);
	try {
		// feature index buffering to accelerate data transfer
		DIMTYPE f,ftmp;
		if(!_index || _features<sub->get_n_raw()) _index.reset(new DIMTYPE[_features]);
		_subfeatures=0;
		for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<_features); _index[_subfeatures++]=fsel;}
#ifdef DEBUG
		{
			std::ostringstream sos;
			sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " ";
			syncout::print(std::cout,sos);
		}
#endif
		assert(_subfeatures==sub->get_d_raw());
		
		typename DATAACCESSOR::PPattern p,ptmp;
		IDXTYPE s,i;
		IDXTYPE count;
		DIMTYPE nzfeat;
		double tmp;

	
		count=0;
		for(DIMTYPE c_train=0;c_train<_classes;c_train++)
		{
			da->setClass(c_train);
			for(const auto &blk : da->blocks(TRAIN)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
			{
				problem.y[count]=c_train; // current pattern class info
				nzfeat=0; // no. of nonzero features processed in current pattern
				ptmp=&p[(std::size_t)i*_features]; // temp pointer to current pattern
				for(f=0;f<_subfeatures;f++)
				{
					ftmp=_index[f]; // (full set) feature index
					tmp=ptmp[ftmp]; // feature value
					if(tmp!=0.0) {
						problem.x[count][nzfeat].index=ftmp+1;
						problem.x[count][nzfeat].value=tmp; 
						nzfeat++;
					}
				}
				problem.x[count][nzfeat].index=-1;
				count++;
			}
		}
		assert(count>0);
		problem.l=count;
		if(model!=nullptr) svm_free_and_destroy_model(&model);
		// train SVM model
		model=svm_train(&problem, &parameters);
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::train() error.");
	}	
	if(model==nullptr) return false;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)  // classifies pattern, returns the respective class index
{
	if(model==nullptr) throw fst_error("Classifier_LIBSVM::classify(): no trained model. Note that clones do not inherit the trained state (LibSVM models can not be cloned) - call train() first.");
	assert(svm_check_parameter(&problem,&parameters)==nullptr);
	try {
		// assumes feature index buffering to accelerate data transfer
		fill_onepattern(pattern);
		// now classify the pattern
		const RETURNTYPE tmpval=svm_predict(model, onepattern);
		cls=(DIMTYPE)tmpval;
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::classify() error.");
	}
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	// per-class scores from the (deterministic) LibSVM decision values: two-class = the
	// single decision value signed per label side; K>2 = the one-vs-one decision values
	// aggregated per class (score[i] += dec(i,j), score[j] -= dec(i,j) over all pairs --
	// continuous, deterministic, no ties beyond genuine score equality).
	// NOTE: LibSVM's probability mode is deliberately NOT used here - it retrains with an
	//       internal cross-validation driven by libc rand() (nondeterministic across runs
	//       and platform-dependent; vendored LibSVM is the fst_rand-shim exception)
	if(model==nullptr) throw fst_error("Classifier_LIBSVM::classify_scores(): no trained model. Note that clones do not inherit the trained state (LibSVM models can not be cloned) - call train() first.");
	try {
		fill_onepattern(pattern); // same transfer as classify()
		const int k=svm_get_nr_class(model); // the MODEL's class count guards the accesses below, not the
		                                     // dataset's: a training fold that contained a single class
		                                     // yields k==1, and model->label[1] then does not exist
		if(_classes==2) { // two-class: one decision value, split over the two label sides
			if(k!=2) return false;
			double dec=0.0;
			svm_predict_values(model, onepattern, &dec); // sign points towards model->label[0]
			scores.assign(2,(RETURNTYPE)0.0);
			const int l0=model->label[0], l1=model->label[1];
			if(l0<0 || l0>1 || l1<0 || l1>1) return false;
			scores[l0]=(RETURNTYPE)dec;
			scores[l1]=(RETURNTYPE)(-dec);
		}
		else { // K>2: aggregate the one-vs-one decision values per class (sign of pair (i,j) points towards model->label[i]);
		       // classes unseen in training never appear among model labels and keep score 0 (deterministic)
			if(k<2) return false;
			_dec.assign((size_t)k*(size_t)(k-1)/2,0.0); // member scratch: avoids a per-pattern allocation
			svm_predict_values(model, onepattern, _dec.data());
			scores.assign(_classes,(RETURNTYPE)0.0);
			int pos=0;
			for(int i=0;i<k;i++) for(int j=i+1;j<k;j++,pos++) {
				const int li=model->label[i], lj=model->label[j];
				if(li<0 || li>=(int)_classes || lj<0 || lj>=(int)_classes) return false;
				scores[li]+=(RETURNTYPE)_dec[pos];
				scores[lj]-=(RETURNTYPE)_dec[pos];
			}
		}
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::classify_scores() error.");
	}
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	assert(da);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(_all_patterns==da->getClassSizeSum());
	if(model==nullptr) throw fst_error("Classifier_LIBSVM::test(): no trained model. Note that clones do not inherit the trained state (LibSVM models can not be cloned) - call train() first.");
	assert(svm_check_parameter(&problem,&parameters)==nullptr);
	try {
		// assumes feature index buffering to accelerate data transfer
		assert(_index);
		assert(_subfeatures>0);
		
		typename DATAACCESSOR::PPattern p;
		IDXTYPE s,i;
		IDXTYPE count, correct;
		DIMTYPE clstmp;

	
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
			std::ostringstream sos;
			sos << " result=" << result << std::endl;
			syncout::print(std::cout,sos);
		}
#endif
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::test() error.");
	}	
	return true;
}

} // namespace
#endif // FSTCLASSIFIERSVM_H ///:~
