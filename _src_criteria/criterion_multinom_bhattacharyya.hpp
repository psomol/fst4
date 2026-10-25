#ifndef FSTCRITERIONMULTINOMBHATTACHARYYA_H
#define FSTCRITERIONMULTINOMBHATTACHARYYA_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    criterion_multinom_bhattacharyya.hpp
   \brief   Implements Bhattacharyya distance based on multinomial model to serve as feature selection criterion
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
#include "error.hpp"
#include "global.hpp"
#include "criterion_multinom.hpp"
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

/*! \brief Implements Bhattacharyya distance based on multinomial model to serve as feature selection criterion

    \warning The value returned depends on the subset SIZE in kind, not only in magnitude:
    for d > 1 it is the general multinomial Bhattacharyya distance (a prior-weighted sum
    over class pairs of -log of the Bhattacharyya coefficient of the two term
    distributions, scaled by the average document length), while for d == 1 the general
    form degenerates and the criterion returns the cached INDIVIDUAL Bhattacharyya value
    of that one feature instead. The two are on different scales, so values obtained for
    singletons must not be compared with values obtained for larger subsets. In practice
    this matters when a search visits subsets of size 1 (e.g. the first step of a forward
    sequential search or a ranking search) - the ordering WITHIN each size is meaningful
    regardless. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Multinomial_Bhattacharyya : public Clonable_Impl<Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< base class: the multinomial-model learning/interface skeleton
	typedef Clonable_Impl<Criterion_Multinomial_Bhattacharyya,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Multinomial_Bhattacharyya() {_IB_computed=false; notify("Criterion_Multinomial_Bhattacharyya constructor.");}
	virtual ~Criterion_Multinomial_Bhattacharyya() {notify("Criterion_Multinomial_Bhattacharyya destructor.");}

	//! \brief Scores \p sub: the general multinomial Bhattacharyya distance over class pairs, or (on a singleton) the individual Bhattacharyya term.
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;
	virtual bool initialize(PDataAccessor da) override; //!< learns the model (parent::initialize()) and resets the lazily-computed individual-Bhattacharyya cache



	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Multinomial_Bhattacharyya()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; copies _IB_computed (base _model copied by parent_impl).
	Criterion_Multinomial_Bhattacharyya(const Criterion_Multinomial_Bhattacharyya& cmb);
private:
	bool _IB_computed; //!< true once the per-feature individual-Bhattacharyya cache (Model_Multinomial::get_IB()) has been computed for the current model
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Multinomial_Bhattacharyya(const Criterion_Multinomial_Bhattacharyya& cmb) :
	parent_impl(cmb),
	_IB_computed(cmb._IB_computed)
{
	notify("Criterion_Multinomial_Bhattacharyya copy-constructor.");
}



template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	notify("Criterion_Multinomial_Bhattacharyya::evaluate().");
	assert(parent::_model);
	assert(parent::get_n()>0);
	assert(sub);
	if(sub->get_d_raw()==0) return false;
	DIMTYPE _d = sub->get_d_raw();
	DIMTYPE f;

	if(_d==1) {// general form of Bhattacharyya unusable, return the individual Bhattacharyya (see the class warning: different scale)
		if(!_IB_computed) {
			parent::_model->denarrow();
			parent::_model->compute_theta();
			parent::_model->compute_IB();
			_IB_computed=true;	
		}
		{ // expects a singleton subset: read its single selected feature index
			auto r=sub->selected_features(); auto it=r.begin();
			if(!(it!=r.end())) return false;
			f=*it;
		}
		assert(f<parent::_model->get_n());
		result = parent::_model->get_IB()[f]; // individual Bhatt - higher values denote better terms
	} else {
		parent::_model->narrow_to(sub);
		parent::_model->compute_theta();
		REALTYPE doc_avg_length = parent::_model->get_doc_avg_length(); // valid after compute_theta() call
		DIMTYPE _classes=parent::_model->get_classes();
		
		REALTYPE *tmpoint1, *tmpoint2;
		RETURNTYPE thetasum;
		RETURNTYPE value=0.0;
		DIMTYPE c1,c2;
		// the DEBUG blocks below trace the per-class-pair accumulation of the Bhattacharyya
		// coefficient; they are compiled in only in DEBUG builds and print nothing otherwise
#ifdef DEBUG
		{
			std::ostringstream sos; sos << "d="<<_d<<", doc_avg_length="<<doc_avg_length<<std::endl;
			syncout::print(std::cout,sos);
		}
#endif
		for(c1=0;c1<_classes;c1++) for(c2=c1+1;c2<_classes;c2++) // for class pair c1,c2
		{
			tmpoint1=&(parent::_model->get_theta()[c1*_d]);
			tmpoint2=&(parent::_model->get_theta()[c2*_d]);
			thetasum=0.0;
			for(f=0;f<_d;f++) 
			{
				thetasum+=sqrt(tmpoint1[f]*tmpoint2[f]);
#ifdef DEBUG
				{
					std::ostringstream sos; sos << "classes<"<<c1<<","<<c2<<">: feat "<<f<<".thetasum="<<thetasum<<", tmpoint1[i="<<f<<"]="<< tmpoint1[f]<< ", tmpoint2[i="<<f<<"]="<< tmpoint2[f]<<std::endl;
					syncout::print(std::cout,sos);
				}
#endif
			}
			value+= (double)log(thetasum) * parent::_model->get_Pc(c1) * parent::_model->get_Pc(c2); // weighted
		}
		result = (-doc_avg_length)*value;
#ifdef DEBUG
		{
			std::ostringstream sos; sos << "result="<<result<<std::endl<<std::endl;
			syncout::print(std::cout,sos);
		}
#endif
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Multinomial_Bhattacharyya::initialize().");
	parent::initialize(da);
	_IB_computed=false;
	return true; 
}


//----------------------------------------------------------------------------

/*! \brief Implements individual Mutual Information based on multinomial model to serve as feature selection criterion in Best Individual Feature setting (feature ranking)
    \note Subset value = SUM of the per-feature individual MI values (on a singleton
          this is the individual MI itself). The sum is the BIF-consistent extension -
          the top-d ranking maximizes it - and is monotonic, but it is NOT a joint
          dependence measure: feature redundancy is invisible to it. Intended for ranking
          use (`--search bif` and `--criterion multinom-mi` in the fst4 CLI). */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Multinomial_MI : public Clonable_Impl<Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< base class: the multinomial-model learning/interface skeleton
	typedef Clonable_Impl<Criterion_Multinomial_MI,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor the model is learned from
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	Criterion_Multinomial_MI() {_MI_computed=false; notify("Criterion_Multinomial_MI constructor.");}
	virtual ~Criterion_Multinomial_MI() {notify("Criterion_Multinomial_MI destructor.");}

	//! \brief Scores \p sub as the sum of per-feature individual Mutual Information values over its selected features.
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;
	virtual bool initialize(PDataAccessor da) override; //!< learns the model (parent::initialize()) and resets the lazily-computed individual-MI cache

	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Multinomial_MI()"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; copies _MI_computed (base _model copied by parent_impl).
	Criterion_Multinomial_MI(const Criterion_Multinomial_MI &cmi) : parent_impl(cmi), _MI_computed(cmi._MI_computed) {notify("Criterion_Multinomial_MI copy-constructor.");}
	bool _MI_computed; //!< true once the per-feature individual-MI cache (Model_Multinomial::get_MI()) has been computed for the current model
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	notify("Criterion_Multinomial_MI::evaluate().");
	assert(parent::_model);
	assert(parent::get_n()>0);
	assert(sub);
	if(sub->get_d_raw()==0) return false;

	if(!_MI_computed) {
		parent::_model->denarrow();
		parent::_model->compute_theta();
		parent::_model->compute_MI();
		_MI_computed=true;
	}
	{ // sum of individual MI over the selected features;
	  // on a singleton (the ranking phase) this equals the individual MI itself
		result=0;
		bool any=false;
		for(DIMTYPE f : sub->selected_features()) {
			assert(f<parent::_model->get_n());
			result += parent::_model->get_MI()[f]; // individual MI - higher values denote better terms
			any=true;
		}
		if(!any) return false;
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Multinomial_MI::initialize().");
	parent::initialize(da);
	_MI_computed=false;
	return true; 
}

} // namespace
#endif // FSTCRITERIONMULTINOMBHATTACHARYYA_H ///:~
