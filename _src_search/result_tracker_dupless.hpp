#ifndef FSTRESULTTRACKERDUPLESS_H
#define FSTRESULTTRACKERDUPLESS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    result_tracker_dupless.hpp
   \brief   Enables collecting multiple results
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

/*! \brief Collects multiple results, avoiding duplicates. 

    See Result_Tracker description why this is needed.
    Upon adding, sorts results descending according to value, in case of
    ties smaller subset is considered better.
*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Dupless : public Clonable_Impl<Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>,Result_Tracker<RETURNTYPE,SUBSET>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Result_Tracker<RETURNTYPE,SUBSET> parent; //!< the abstract Result_Tracker interface this class implements
	typedef Clonable_Impl<Result_Tracker_Dupless,parent,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone() via the copy-constructor (CLONE_DEEP); clone()/sharing_clone() are inherited from Result_Tracker, which throws for both
	typedef typename parent::ResultRec ResultRec;     //!< see Result_Tracker::ResultRec
	typedef typename parent::PResultRec PResultRec;   //!< see Result_Tracker::PResultRec
	typedef std::shared_ptr<SUBSET> PSubset;          //!< shared pointer to a tracked feature subset
	//! \param capacity_limit if nonzero, caps the number of retained results (the tail, i.e. the worst-valued results, is trimmed as new results are inserted); 0 = unlimited
	Result_Tracker_Dupless(const IDXTYPE capacity_limit=0):_margin(0),_restrict_to_d(0),_capacity_limit(capacity_limit) {notify("Result_Tracker_Dupless constructor.");}
	Result_Tracker_Dupless(const Result_Tracker_Dupless& rtd):parent_impl(rtd), _margin(rtd._margin), _restrict_to_d(rtd._restrict_to_d), _capacity_limit(rtd._capacity_limit) {notify("Result_Tracker_Dupless copy-constructor.");} //!< copies the configuration (margin, cardinality restriction, capacity limit) only; the \c results list is NOT copied (starts empty) -- this constructor backs stateless_clone() (CLONE_DEEP), which by contract does not replicate exact object state
	virtual ~Result_Tracker_Dupless() {notify("Result_Tracker_Dupless destructor.");}

	/*! \brief Offers one (value, subset) result for inclusion; \p sub is copied, the caller keeps ownership of its own object.

		The result is declined (false is returned) when
		\li a cardinality restriction is in effect and sub->get_d() differs from it
		    (see set_inclusion_cardinality_restriction()),
		\li a margin is in effect and \p value falls more than the margin below the
		    best value held so far (see set_inclusion_margin()),
		\li an equivalent subset of the same value and the same cardinality is
		    already held (duplicates are looked for only among results with exactly
		    that value and cardinality - results differing in value are never
		    compared feature-wise).

		Otherwise the result is sorted into the list, which stays ordered by
		descending value, and among equal values by ascending cardinality (the
		smaller subset counts as the better one). A result that becomes the new best
		additionally trims the tail of results that thereby fell outside the margin.
		Finally, a nonzero capacity limit trims the worst-valued results away - which
		may discard the just inserted result itself if it ended up at the tail;
		true only reports that the result was accepted for insertion. */
	virtual bool add(const RETURNTYPE value, const PSubset sub) override;
	virtual void clear() override {results.clear();}
	virtual std::size_t size() const override {return results.size();}
	virtual std::size_t size(const RETURNTYPE margin) const; //!< number of currently held results whose value is within \p margin of the best (first, highest-value) held result
	virtual bool get_first(PResultRec &r) override; //!< starts iteration at the best held result; false if nothing is held. The returned pointer refers into this tracker's own storage and is invalidated by add(), join() and clear()
	virtual bool get_next(PResultRec &r) override; //!< hands out the next result in descending-value order; false when the iteration is exhausted. There is one shared cursor per tracker, so nested or concurrent iterations of the same tracker interfere
	virtual void join(Result_Tracker<RETURNTYPE,SUBSET> &src) override; //!< merges the results held by \p src into this tracker, keeping the ordering and skipping duplicates exactly as add() does, then re-applies the capacity limit and the margin to the merged list. Consumes \p src's iteration cursor; \p src itself is left unchanged

	        void set_inclusion_margin(const RETURNTYPE margin=0) {_margin=margin;} //!< restrict adding to only subsets with criterion value within given distance (margin) from best known
	        void set_inclusion_cardinality_restriction(const DIMTYPE d) {_restrict_to_d=d;} //!< for d>0 restricts adding only to subsets of size d, for d=0 subset size restriction is removed
	


	virtual std::ostream& print(std::ostream& os) const override {os << "Result_Tracker_Dupless(limit=" << _capacity_limit << ", margin=" << _margin << ") size " << size(); return os;}
protected:
	std::list<ResultRec> results; //!< held results, sorted descending by value (ties broken toward the smaller subset); see add() for the exact insertion/dedup/trim rule
	RETURNTYPE _margin; //!< set via set_inclusion_margin(); 0 = no margin restriction
	DIMTYPE _restrict_to_d; //!< set via set_inclusion_cardinality_restriction(); 0 = no subset-size restriction
	const IDXTYPE _capacity_limit; //!< set by the constructor; 0 = unlimited

public:
	typedef typename std::list<ResultRec>::const_iterator CONSTRESULTSITER; //!< read-only iterator over results, for use by subclasses/friends inspecting the collected results
	typedef typename std::list<ResultRec>::iterator RESULTSITER; //!< mutable iterator over results, as returned by begin()/end()
	RESULTSITER begin() {return results.begin();} //!< iterator to the best (first) held result
	RESULTSITER end() {return results.end();} //!< iterator past the last held result

private:
	RESULTSITER getiter; //!< iteration cursor for get_first()/get_next() (single shared cursor, not reentrant)
};



template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::add(const RETURNTYPE value, const PSubset sub)
{
	assert(_margin>=0);
	assert(sub);
	if(_restrict_to_d>0 && sub->get_d()!=_restrict_to_d) return false;
	ResultRec rr; rr.value=value; rr.sub.reset(new SUBSET(*sub));
	bool added=false;
	if(!results.empty()) 
	{
		RESULTSITER iter=results.begin();
		if(_margin>0 && iter->value-_margin>value)
		{ // the new result is outside the _margin given by the so-far best result value. Will not be added
		} 
		else // sort-in the new result, avoiding duplicates + possibly trim the tail of list that falls below _margin, if the new result becomes the new best
		{
			if(value>iter->value) {
				results.push_front(rr); added=true;
				while(_margin>0 && results.back().value<value-_margin) results.pop_back(); // trim the tail of results that fell below _margin
				while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
			} else { // sort the new result in, trimming not necessary
				bool dupl=false;
				while(iter!=results.end() && (value<iter->value || (value==iter->value && sub->get_d()>iter->sub->get_d()))) iter++; // skip all results that rank better: higher value, or equal value with smaller cardinality
				while(!dupl && iter!=results.end() && (value==iter->value && sub->get_d()==iter->sub->get_d())) if(sub->equivalent(*(iter->sub))) dupl=true; else iter++; // within the block of equal value and equal cardinality: is an equivalent subset already held?
				if(!dupl) {
					results.insert(iter,rr); added=true;
					while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
				}
			}
		}
	} else {
            results.push_back(rr); added=true;
        }
	return added;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
std::size_t Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::size(const RETURNTYPE margin) const
{
	assert(margin>=0);
	if(results.empty()) return 0;
	CONSTRESULTSITER iter=results.begin();
	std::size_t count=0;
	RETURNTYPE threshold=iter->value-margin;
	while(iter!=results.end() && iter->value>=threshold) {count++; iter++;}
	return count;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::get_first(PResultRec &r)
{
	if(results.empty()) return false;
	getiter=results.begin();
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::get_next(PResultRec &r)
{
	if(getiter==results.end()) return false;
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
void Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::join(Result_Tracker<RETURNTYPE,SUBSET> &src)
{
	if(results.empty()) // simple copy
	{
		PResultRec pr;
		for(bool got=src.get_first(pr);got==true;got=src.get_next(pr)) results.push_back(*pr);
	}
	else // join
	{
		RESULTSITER iter=begin();
		PResultRec pr;
		bool got=src.get_first(pr);
		while(got)
		{
			assert(pr->sub);
			while(iter!=end() && (iter->value>pr->value || (iter->value==pr->value && iter->sub->get_d()<pr->sub->get_d()))) iter++;
			bool dupl=false;
			while(!dupl && iter!=end() && (iter->value==pr->value && iter->sub->get_d()==pr->sub->get_d())) if(pr->sub->equivalent(*(iter->sub))) dupl=true; else iter++;
			if(!dupl) results.insert(iter,*pr);
			got=src.get_next(pr);
		}
	}
	if(!results.empty()) { // make joined list coherent with _capacity_limit and _margin restrictions
		RETURNTYPE threshold=results.front().value-_margin;
		while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back();
		while(_margin>0 && results.back().value<threshold) results.pop_back();
	}
}

} // namespace
#endif // FSTRESULTTRACKERDUPLESS_H ///:~
