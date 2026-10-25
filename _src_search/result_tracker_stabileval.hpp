#ifndef FSTRESULTTRACKERSTABILEVAL_H
#define FSTRESULTTRACKERSTABILEVAL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    result_tracker_stabileval.hpp
   \brief   Enables collecting multiple results, then evaluating various stability measures
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
#include <list>
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"

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

/*! \brief Collects multiple results to evaluate various stability measures on the collection.

    Feed one feature subset per selection trial (different data splits, different
    bootstrap samples, repeated randomized runs) with add(), then read a stability
    measure off the collection, or compare two such collections with the similarity
    measures. All eleven measures are normalized to the interval [0,1], and in all of
    them a HIGHER value means more stable (respectively more similar) output; each
    declaration below states the convention again.

    The measures and their properties are defined in
    P. Somol, J. Novovičová: Evaluating Stability and Comparing Output of Feature
    Selectors that Optimize Feature Subset Cardinality, IEEE Transactions on Pattern
    Analysis and Machine Intelligence 32(11), 2010, pp. 1921–1939, except where a
    declaration names a different source.

    \note Duplicates are permitted.
    \note Setting nonzero capacity limit here should be avoided unless absolutely necessary, otherwise the stability measure value computation would be based on incomplete (and possibly misleading) information
    \note The stability and similarity measures are defined on an empty collection:
          they return 0.0 (the similarity ones also emit a warning). The descriptive
          statistics value_mean(), value_stddev(), size_mean() and size_stddev(),
          however, THROW fst_error when the collection is empty — test size() first.
*/
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Stability_Evaluator : public Clonable_Impl<Result_Tracker_Stability_Evaluator<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>,Result_Tracker<RETURNTYPE,SUBSET>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Result_Tracker<RETURNTYPE,SUBSET> parent; //!< the abstract Result_Tracker interface this class implements
	typedef Clonable_Impl<Result_Tracker_Stability_Evaluator,parent,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone() via the copy-constructor (CLONE_DEEP); clone()/sharing_clone() are inherited from Result_Tracker, which throws for both
	typedef typename parent::ResultRec ResultRec; //!< see Result_Tracker::ResultRec
	typedef ResultRec* PResultRec; //!< raw pointer to a ResultRec, as used by get_first()/get_next() iteration
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a tracked feature subset
	//! \param capacity_limit if nonzero, caps the number of retained results (the tail, i.e. the worst-valued results, is trimmed as new results are inserted); 0 = unlimited -- avoid a nonzero limit here unless necessary, as it biases the stability measures towards incomplete information (see the class note)
	Result_Tracker_Stability_Evaluator(const IDXTYPE capacity_limit=0):_capacity_limit(capacity_limit) {notify("Result_Tracker_Stability_Evaluator constructor.");}
	Result_Tracker_Stability_Evaluator(const Result_Tracker_Stability_Evaluator& rtse):parent_impl(rtse), _capacity_limit(rtse._capacity_limit) {notify("Result_Tracker_Stability_Evaluator copy-constructor.");} //!< copies the capacity limit only; the \c results list is NOT copied (starts empty) -- this constructor backs stateless_clone() (CLONE_DEEP), which by contract does not replicate exact object state
	virtual ~Result_Tracker_Stability_Evaluator() {notify("Result_Tracker_Stability_Evaluator destructor.");}

	virtual bool add(const RETURNTYPE value, const PSubset sub) override;
	virtual void clear() override {results.clear();}
	virtual std::size_t size() const override {return results.size();}
	virtual bool get_first(PResultRec &r) override; //!< iterator
	virtual bool get_next(PResultRec &r) override; //!< iterator

	RETURNTYPE value_mean() const; //!< mean value of the ResultRec::value values; throws fst_error on an empty collection
	RETURNTYPE value_stddev() const; //!< standard deviation of the ResultRec::value values; throws fst_error on an empty collection
	RETURNTYPE size_mean() const; //!< mean value of the size of ResultRec::sub subsets; throws fst_error on an empty collection
	RETURNTYPE size_stddev() const; //!< standard deviation of the size of ResultRec::sub subsets; throws fst_error on an empty collection

	// Stability of THIS collection. All return a value in [0,1], higher = more stable
	// (1.0 = all trials selected the same subset), and all return 1.0 for a single held
	// result and 0.0 for an empty collection. Measures taking Ysiz need the size of the
	// whole feature universe Y, i.e. the number of features the selection chose FROM.
	RETURNTYPE stability_C() const; //!< stability measure "C" (consistency: mean over features of the relative frequency of co-occurrence in the held subsets); [0,1], higher = more stable
	RETURNTYPE stability_CW() const; //!< stability measure "CW" (weighted consistency: as "C", but each feature contributes proportionally to how often it was selected); [0,1], higher = more stable
	//! \brief stability measure "CW_{rel}": "CW" rescaled between the minimum and maximum attainable
	//! for the observed number of subsets and their total cardinality, which removes the dependence on
	//! subset size and on the size of Y. Range [0,1], higher = more stable.
	//! \param Ysiz size of the feature universe Y (must be >0)
	RETURNTYPE stability_CWrel(const DIMTYPE Ysiz) const;
	RETURNTYPE stability_SH() const; //!< stability measure "SH" (Shannon-entropy based); [0,1], higher = more stable. Cf. Somol, Novovičová, Pudil: A New Measure of Feature Selection Algorithms’ Stability, 2009 IEEE International Conference on Data Mining Workshops
	//! \brief stability measure "PH" (relative Hamming distance between the held subsets, expressed as
	//! agreement). Range [0,1], higher = more stable.
	//! \param Ysiz size of the feature universe Y (must be >0)
	RETURNTYPE stability_PH(const DIMTYPE Ysiz) const;
	RETURNTYPE stability_ATI() const; //!< stability measure "ATI" (average pairwise Tanimoto index over all pairs of held subsets); [0,1], higher = more stable. Cf. A. Kalousis, J. Prados, and M. Hilario, “Stability of feature selection algorithms,” in Proc. of 5th IEEE International Conference on Data Mining (ICDM 05), Houston, Texas, 2005, pp. 218–225
	//! \brief stability measure "ANHI" (average pairwise normalized Hamming index over all pairs of held
	//! subsets). Range [0,1], higher = more stable.
	//! \param Ysiz size of the feature universe Y (must be >0)
	RETURNTYPE stability_ANHI(const DIMTYPE Ysiz) const;

	// Similarity of this collection to ANOTHER collection \p rtse. All return a value in
	// [0,1], higher = more similar (1.0 = indistinguishable feature-occurrence pattern),
	// and 0.0 with a warning if either collection is empty. The relation is symmetric.
	RETURNTYPE similarity_IC(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "IC", the two-collection counterpart of stability_C(); [0,1], higher = more similar
	RETURNTYPE similarity_ICW(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "ICW", the two-collection counterpart of stability_CW(); [0,1], higher = more similar
	RETURNTYPE similarity_IATI(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "IATI": mean Tanimoto index over all cross pairs (one subset from each collection); [0,1], higher = more similar
	//! \brief similarity measure "IANHI": mean normalized Hamming index over all cross pairs (one subset
	//! from each collection). Range [0,1], higher = more similar.
	//! \param Ysiz size of the feature universe Y (must be >0)
	//! \param rtse the other collection
	RETURNTYPE similarity_IANHI(const DIMTYPE Ysiz, const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const;
	


	virtual std::ostream& print(std::ostream& os) const override {os << "Result_Tracker_Stability_Evaluator(limit=" << _capacity_limit << ") size " << size(); return os;}
protected:
	std::list<ResultRec> results; //!< held results, sorted descending by value (ties broken toward the smaller subset); duplicates are permitted (see the class note)
	const IDXTYPE _capacity_limit; //!< set by the constructor; 0 = unlimited

	RETURNTYPE get_TI(const PSubset &sub1, const PSubset &sub2) const; //!< pairwise Tanimoto-style index of \p sub1 and \p sub2: (size of selected-feature intersection) / (size of selected-feature union), over a Ysiz=max(sub1,sub2 raw size) feature universe; -1.0 (with a warning) if the union is empty. Feeds stability_ATI()/similarity_IATI()
	RETURNTYPE get_NHI(const DIMTYPE Ysiz, const PSubset &sub1, const PSubset &sub2) const; //!< pairwise normalized Hamming index of \p sub1 and \p sub2 over an explicit \p Ysiz-feature universe: 1 - (symmetric-difference size)/Ysiz. Feeds stability_ANHI()/similarity_IANHI()
	DIMTYPE max_features() const; //!< size of the feature universe the held results were selected from: the largest sub->get_n_raw() over the collection (NOT the largest subset size); 0 when the collection is empty

public:
	typedef typename std::list<ResultRec>::const_iterator CONSTRESULTSITER; //!< read-only iterator over results, used throughout the stability/similarity measure computations
	typedef typename std::list<ResultRec>::iterator RESULTSITER; //!< mutable iterator over results, as returned by begin()/end()
	RESULTSITER begin() {return results.begin();} //!< iterator to the best (first) held result
	RESULTSITER end() {return results.end();} //!< iterator past the last held result

private:
	RESULTSITER getiter; //!< iteration cursor for get_first()/get_next() (single shared cursor, not reentrant)
	// Scratch buffers for get_TI/get_NHI/similarity_IC/similarity_ICW. They are per-object
	// members ON PURPOSE: shared (e.g. static) buffers would be written by every instance and
	// every thread at once. Holding them as vectors keeps the class copyable for CLONE_DEEP.
	// Consequently the const measure methods are NOT safe to call concurrently on one object.
	mutable std::vector<int> _ti_mask; //!< get_TI() scratch: per-feature selection count (0/1/2) over the current pair
	mutable std::vector<int> _nhi_mask; //!< get_NHI() scratch: per-feature selection count (0/1/2) over the current pair
	mutable std::vector<int> _sim_F1; //!< similarity_IC()/similarity_ICW() scratch: per-feature selection count over this tracker's results
	mutable std::vector<int> _sim_F2; //!< similarity_IC()/similarity_ICW() scratch: per-feature selection count over the other tracker's results
};



template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::add(const RETURNTYPE value, const PSubset sub)
{
	assert(sub);
	ResultRec rr; rr.value=value; rr.sub.reset(new SUBSET(*sub));
	if(!results.empty()) 
	{
		RESULTSITER iter=results.begin();
		while(iter!=results.end() && (value<iter->value || (value==iter->value && sub->get_d()>iter->sub->get_d()))) iter++;
		results.insert(iter,rr);
		while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
	} else results.push_back(rr);
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_first(PResultRec &r)
{
	if(results.empty()) return false;
	getiter=results.begin();
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_next(PResultRec &r)
{
	if(getiter==results.end()) return false;
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::value_mean() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::value_mean() called on empty list.");
	RETURNTYPE mean=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) mean+=iter->value;
	return mean/(RETURNTYPE)results.size();
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::value_stddev() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::value_stddev() called on empty list.");
	const RETURNTYPE mean=value_mean();
	RETURNTYPE sd=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) sd+=(iter->value-mean)*(iter->value-mean);
	return sqrt(sd/(RETURNTYPE)results.size());
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::size_mean() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::size_mean() called on empty list.");
	RETURNTYPE mean=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); mean+=iter->sub->get_d_raw();}
	return mean/(RETURNTYPE)results.size();
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::size_stddev() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::size_stddev() called on empty list.");
	const RETURNTYPE mean=size_mean();
	RETURNTYPE sd=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); sd+=(iter->sub->get_d_raw()-mean)*(iter->sub->get_d_raw()-mean);}
	return sqrt(sd/(RETURNTYPE)results.size());
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
DIMTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::max_features() const
{
	DIMTYPE maxn=0;
	if(!results.empty())
	{
		CONSTRESULTSITER iter=results.begin();
		while(iter!=results.end()) {
			DIMTYPE ss=iter->sub->get_n_raw();
			if(ss>maxn) maxn=ss;
			iter++;
		}
	}
	return maxn;	
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_C() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			CONSTRESULTSITER iter;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=(RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1); 
				count++;
			}
		}
		if(count==0) return 0.0;
		return result/(RETURNTYPE)count;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_CW() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long fcntsum=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); fcntsum+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=((RETURNTYPE)(fcnt)/(RETURNTYPE)(fcntsum)) * ((RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1)); 
				count++;
			}
		}
		// unlike "C", the weighting already normalizes the sum, so count only tells
		// whether any feature was ever selected at all
		if(count==0) return 0.0;
		return result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_CWrel(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long N=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); N+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<Ysiz;f++) { // over the whole feature universe Y, not only the features seen in the held subsets
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=((RETURNTYPE)(fcnt)/(RETURNTYPE)(N)) * ((RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1)); 
				count++;
			}
		}
		if(count==0) return 0.0;
		// Rescale the raw weighted consistency between the extremes attainable for this
		// many subsets (size) of this total cardinality (N) over this feature universe (Y):
		// CWmin is reached when the N selections are spread as evenly as possible over Y
		// features, CWmax when they are packed into as few features as possible. D and H are
		// the remainders of those two even/packed splits. Both closed forms are from the
		// TPAMI paper cited in the class documentation.
		long Y=Ysiz;
		long D=N%Y, H=N%size;
		RETURNTYPE CWmin=(RETURNTYPE)((RETURNTYPE)N*(RETURNTYPE)N-(RETURNTYPE)Y*(RETURNTYPE)(N-D)-D*D)/(RETURNTYPE)((RETURNTYPE)Y*(RETURNTYPE)N*(RETURNTYPE)(size-1));
		RETURNTYPE CWmax=(RETURNTYPE)(H*H+N*(size-1)-H*size)/(RETURNTYPE)(N*(size-1));
		if(CWmax>CWmin) return (result-CWmin)/(CWmax-CWmin); else return result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_SH() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long fcntsum=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); fcntsum+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=fcnt * (log((double)fcnt)/log((double)2.0)); 
				count++;
			}
		}
		if(count==0) return 0.0;
		return result/(fcntsum*(log((double)size)/log((double)2.0)));
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_PH(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		for(DIMTYPE f=0;f<Ysiz;f++) {
			long fcnt=0;
			for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			result+=((fcnt>size-fcnt)?(RETURNTYPE)fcnt/(RETURNTYPE)size:(RETURNTYPE)(size-fcnt)/(RETURNTYPE)size);
		}
		return 2.0*result/(RETURNTYPE)Ysiz -1.0;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_TI(const PSubset &sub1, const PSubset &sub2) const
{
	assert(sub1); 
	assert(sub2);
	const DIMTYPE Ysiz=maxf<DIMTYPE>(sub1->get_n_raw(),sub2->get_n_raw());
	std::vector<int> &TI_mask=_ti_mask; // per-object scratch, see the note at its declaration
	assert(Ysiz>0);
	if((DIMTYPE)TI_mask.size()<Ysiz) {
		TI_mask.resize(Ysiz);
		notify("get_ATI reallocation.");
	}

	DIMTYPE f;
	for(f=0;f<Ysiz;f++) TI_mask[f]=0;
	// mark each feature with how many of the two subsets contain it (0, 1 or 2)
	for(DIMTYPE fsel : sub1->selected_features()) {assert(fsel<Ysiz); TI_mask[fsel]++;}
	for(DIMTYPE fsel : sub2->selected_features()) {assert(fsel<Ysiz); TI_mask[fsel]++;}
	
	DIMTYPE SAS=0, SUS=0;
	for(f=0;f<Ysiz;f++) {if(TI_mask[f]==2) SAS++; if(TI_mask[f]!=0) SUS++;}
	if(SUS==0) {notify("Warning: get_TI(): SUS==0."); return -1.0;}
	else return (RETURNTYPE)SAS/(RETURNTYPE)SUS;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_ATI() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1,iter2;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			iter2=iter1; iter2++;
			while(iter2!=results.end()) {
				result+=get_TI(iter1->sub,iter2->sub);
				iter2++;
			}
		}
		return (2.0/(size*(size-1)))*result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_NHI(const DIMTYPE Ysiz, const PSubset &sub1, const PSubset &sub2) const
{
	assert(sub1); 
	assert(sub2);
	assert(Ysiz>0);
	std::vector<int> &NHI_mask=_nhi_mask; // per-object scratch, see the note at its declaration
	if((DIMTYPE)NHI_mask.size()<Ysiz) {
		NHI_mask.resize(Ysiz);
		notify("get_ANHI reallocation.");
	}

	DIMTYPE f;
	for(f=0;f<Ysiz;f++) NHI_mask[f]=0;
	// mark each feature with how many of the two subsets contain it (0, 1 or 2)
	for(DIMTYPE fsel : sub1->selected_features()) {assert(fsel<Ysiz); NHI_mask[fsel]++;}
	for(DIMTYPE fsel : sub2->selected_features()) {assert(fsel<Ysiz); NHI_mask[fsel]++;}
	
	DIMTYPE SES=0;
	for(f=0;f<Ysiz;f++) if(NHI_mask[f]==1) SES++;
	return 1.0-(RETURNTYPE)SES/(RETURNTYPE)Ysiz;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_ANHI(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1,iter2;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			iter2=iter1; iter2++;
			while(iter2!=results.end()) {
				result+=get_NHI(Ysiz,iter1->sub,iter2->sub);
				iter2++;
			}
		}
		return (2.0/(size*(size-1)))*result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IC(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		const DIMTYPE Ysiz=maxf<DIMTYPE>(max_features(),rtse.max_features());
		assert(Ysiz>0);
		std::vector<int> &F1=_sim_F1, &F2=_sim_F2; // per-object scratch, see the note at its declaration
		if((DIMTYPE)F1.size()<Ysiz) {
			F1.resize(Ysiz);
			F2.resize(Ysiz);
			notify("similarity_IC reallocation.");
		}
		DIMTYPE f;
		for(f=0;f<Ysiz;f++) F1[f]=F2[f]=0;
		for(f=0;f<Ysiz;f++) {
			CONSTRESULTSITER iter1;
			for(iter1=results.begin();iter1!=results.end();iter1++) {assert(iter1->sub); if(iter1->sub->selected_raw(f)) F1[f]++;}
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) {assert(iter2->sub); if(iter2->sub->selected_raw(f)) F2[f]++;}
		}
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		for(f=0;f<Ysiz;f++) {
			if(F1[f]>0 || F2[f]>0) { 
				result+=fabs( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) - (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ); 
				count++;
			}
		}
		assert(count>0);
		return 1.0 - result/(RETURNTYPE)count;
	} else {notify("Warning: similarity_IC(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_ICW(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		const DIMTYPE Ysiz=maxf<DIMTYPE>(max_features(),rtse.max_features());
		assert(Ysiz>0);
		std::vector<int> &F1=_sim_F1, &F2=_sim_F2; // per-object scratch, see the note at its declaration
		if((DIMTYPE)F1.size()<Ysiz) {
			F1.resize(Ysiz);
			F2.resize(Ysiz);
			notify("similarity_ICW reallocation.");
		}
		DIMTYPE f;
		for(f=0;f<Ysiz;f++) F1[f]=F2[f]=0;
		for(f=0;f<Ysiz;f++) {
			CONSTRESULTSITER iter1;
			for(iter1=results.begin();iter1!=results.end();iter1++) {assert(iter1->sub); if(iter1->sub->selected_raw(f)) F1[f]++;}
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) {assert(iter2->sub); if(iter2->sub->selected_raw(f)) F2[f]++;}
		}
		RETURNTYPE fcntsum=0.0;
		for(f=0;f<Ysiz;f++) {
			fcntsum+= maxf<RETURNTYPE>( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) , (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) );
		}
		assert(fcntsum>0);
		RETURNTYPE result=0.0;
		for(f=0;f<Ysiz;f++) {
			result+=(maxf<RETURNTYPE>( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) , (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ) / fcntsum) * fabs( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) - (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ); 
		}
		return 1.0 - result;	
	} 
	else {notify("Warning: similarity_ICW(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IATI(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			assert(iter1->sub);
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) 
			{
				assert(iter2->sub);
				result+=get_TI(iter1->sub,iter2->sub);
			}
		}
		return (1.0/((RETURNTYPE)size1*(RETURNTYPE)(size2)))*result;	
	} 
	else {notify("Warning: similarity_IATI(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IANHI(const DIMTYPE Ysiz, const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		assert(Ysiz>0);
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			assert(iter1->sub);
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) 
			{
				assert(iter2->sub);
				result+=get_NHI(Ysiz,iter1->sub,iter2->sub);
			}
		}
		return (1.0/((RETURNTYPE)size1*(RETURNTYPE)(size2)))*result;	
	} 
	else {notify("Warning: similarity_IANHI(): nothing to compare."); return 0.0;}
}

} // namespace
#endif // FSTRESULTTRACKERSTABILEVAL_H ///:~
