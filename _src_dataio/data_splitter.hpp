#ifndef FSTDATASPLITTER_H
#define FSTDATASPLITTER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter.hpp
   \brief   Defines interface for data splitting implementations (for use in data accessors)
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
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"

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

template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter;

template<class INTERVALCONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds);

/*! \brief Abstract class, defines interface for data splitting implementations (for use in data accessors)

    A splitter enumerates a sequence of train/test partitions. It does not touch data: it works
    purely with sample INDEXES, and writes each partition as two lists of index intervals, per class.

    Driving a splitter. The owner first calls assign() and set_n() once per class, then iterates
    `for(bool b=makeFirstSplit(); b; b=makeNextSplit())`. Each successful call overwrites the train
    and test interval lists registered through assign(); the lists are only valid until the next call.
    getNoOfSplits() says how many iterations to expect and requires set_n() to have been called.

    What an implementation must guarantee. Every split, for every class:
    \li indexes are 0-based and LOCAL TO THE CLASS -- index 0 is that class's first sample and the
        largest valid index is get_n(class)-1. A splitter never sees a global sample index;
    \li the train and test lists must not overlap, neither with each other nor internally;
    \li the intervals within each list must be sorted by ascending startidx. Nested splitting
        translates a deeper level's indexes through Data_Intervaller::reduce(), which walks both
        lists once in ascending order and silently produces wrong intervals for unsorted input;
    \li an interval with count==0 should not be emitted; the consumers treat empty intervals as a
        degenerate case and skip them;
    \li leaving the train part empty makes the split unusable for any criterion that trains on it.
        A splitter that can reach such a configuration should end the sequence instead.

    Implementing one. The template methods makeFirstSplit()/makeNextSplit() own the shared
    scaffolding -- the state assertions, the per-class loop and the error reporting -- so a concrete
    splitter implements only the prepare*, splitClass* and finishSplit hooks, i.e. its actual
    splitting logic. The call order is: prepareFirstSplit(), then splitClassFirst() once per class in
    class order, then finishSplit(); and for every later split prepareNextSplit(), then
    splitClassNext() once per class in class order, then finishSplit(). A prepare hook returning
    false ends the sequence, and neither splitClass nor finishSplit is called after it. */
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter : public Clonable {
	//! Streams a splitter's print() to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds);
public:
	typedef std::shared_ptr<INTERVALCONTAINER> PIntervaller;  //!< shared pointer to one class's list of train/test data intervals
	Data_Splitter() {csplit.clear(); notify("Data_Splitter constructor.");}
	//! Deep-copies every class's TClassSplitter (own train/test interval lists, not shared with \p dsp).
	Data_Splitter(const Data_Splitter& dsp);
	virtual ~Data_Splitter() {notify("Data_Splitter destructor.");}

	//! Registers class \p _clsidx's train/test interval lists (growing csplit as needed); the caller retains ownership via the shared pointers.
	void assign(const unsigned int _clsidx, const PIntervaller _train, const PIntervaller _test);
	//! Sets class \p _clsidx's pattern count, which getNoOfSplits() and the concrete prepare*/splitClass* hooks size their splits against (growing csplit as needed).
	void set_n(const unsigned int _clsidx, const IDXTYPE _n);
	IDXTYPE get_n(const unsigned int _clsidx) const; //!< the pattern count previously set for class \p _clsidx via set_n()

	virtual IDXTYPE getNoOfSplits() const =0;	//!< number of splits this splitter yields (requires class sizes to have been set via set_n())
	//! \brief Produces the first split: prepareFirstSplit(), then splitClassFirst() per class, then finishSplit().
	//! \return false if no split can be made at all; any exception a hook throws is rethrown as an fst_error
	//! naming the splitter and this method -- the original exception's own message is not preserved.
	bool makeFirstSplit();
	//! \brief Produces the next split: prepareNextSplit(), then splitClassNext() per class, then finishSplit().
	//! \return false once the split sequence is exhausted; exceptions are reported as in makeFirstSplit().
	bool makeNextSplit();

	Data_Splitter* clone() const override {throw fst_error("Data_Splitter::clone() not supported, use Data_Splitter::stateless_clone() instead.");}
	Data_Splitter* sharing_clone() const override {throw fst_error("Data_Splitter::sharing_clone() not supported, use Data_Splitter::stateless_clone() instead.");}
	Data_Splitter* stateless_clone() const override =0;

	virtual std::ostream& print(std::ostream& os) const override {return os;} //!< writes a human-readable identification of the splitter and its configuration; base default writes nothing

protected:
	//! Data splitting support structure; holds one set of intervals (train, test) per each (data)class
	class TClassSplitter {
	public:
		TClassSplitter() {n=0; train.reset(); test.reset(); notify("TClassSplitter empty constructor.");}
		//! Deep-copies \p cs's train/test interval lists (own INTERVALCONTAINER instances, not shared).
		TClassSplitter(const TClassSplitter& cs) : n(cs.n) {
			notify("TClassSplitter strong copy constructor begin.");
			if(cs.train) train.reset(new INTERVALCONTAINER(*cs.train));
			if(cs.test) test.reset(new INTERVALCONTAINER(*cs.test));
			notify("TClassSplitter strong copy constructor end.");
		}
		~TClassSplitter() {notify("TClassSplitter destructor.");}
		IDXTYPE n;	//!< n=total no. of patterns
		PIntervaller train;	//!< pointer to external list of intervals that will be overwritten from here
		PIntervaller test;	//!< pointer to external list of intervals that will be overwritten from here
	};
	typedef std::shared_ptr<TClassSplitter> PClassSplitter;   //!< shared pointer to one class's TClassSplitter
	std::vector<PClassSplitter> csplit; //!< one TClassSplitter per data class, indexed by class index; grown on demand by assign()/set_n()
	typedef typename std::vector<PClassSplitter>::const_iterator ClassIterator; //!< iterator over csplit, as passed to the per-class splitClassFirst()/splitClassNext() hooks

	void assert_csplit() const; //!< DEBUG-only: asserts csplit is non-empty and every class has n>0 and non-null train/test lists

	// hooks a concrete splitter implements; see the class description for the call order and
	// for the contract the written intervals must satisfy
	virtual const char* splitter_name() const =0; //!< class name, used in error messages
	virtual bool prepareFirstSplit() {return true;} //!< set up split-sequence state; false = no split can be made
	virtual bool prepareNextSplit() =0; //!< advance split-sequence state; false = no more splits
	virtual void splitClassFirst(ClassIterator c, const unsigned int cidx) =0; //!< write first-split train/test intervals of class \p cidx (\p c is that class's entry in csplit)
	virtual void splitClassNext(ClassIterator c, const unsigned int cidx) {} //!< write next-split train/test intervals of class \p cidx (a single-split splitter never gets here, since its prepareNextSplit() returns false)
	virtual void finishSplit() {} //!< post-loop state update shared by first/next split
};

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter<INTERVALCONTAINER,IDXTYPE>::makeFirstSplit()
{
	assert_csplit();
	try {
		if(!prepareFirstSplit()) return false;
		unsigned int cidx=0;
		for(ClassIterator c=csplit.begin();c!=csplit.end();c++,cidx++) splitClassFirst(c,cidx);
		finishSplit();
	} catch(...) {
		throw fst_error(std::string(splitter_name())+"::makeFirstSplit() error.");
	}
	return true;
}

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter<INTERVALCONTAINER,IDXTYPE>::makeNextSplit()
{
	try {
		if(!prepareNextSplit()) return false;
		unsigned int cidx=0;
		for(ClassIterator c=csplit.begin();c!=csplit.end();c++,cidx++) splitClassNext(c,cidx);
		finishSplit();
	} catch(...) {
		throw fst_error(std::string(splitter_name())+"::makeNextSplit() error.");
	}
	return true;
}


template<class INTERVALCONTAINER, typename IDXTYPE>
Data_Splitter<INTERVALCONTAINER,IDXTYPE>::Data_Splitter(const Data_Splitter& dsp)
{
	notify("Data_Splitter strong copy constructor begin.");
	csplit.clear();
	for(typename std::vector<PClassSplitter>::const_iterator c=dsp.csplit.begin();c!=dsp.csplit.end();c++) 
		{PClassSplitter pcs(new TClassSplitter(*(*c))); this->csplit.push_back(pcs);}
	notify("Data_Splitter strong copy constructor end.");
}

template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::assign(const unsigned int _clsidx, const PIntervaller _train, const PIntervaller _test)
{
	assert(_train); 
	assert(_test);
	try {	
		while(csplit.size()<=_clsidx) {PClassSplitter pcs(new TClassSplitter); csplit.push_back(pcs);}
		csplit[_clsidx]->train=_train; 
		csplit[_clsidx]->test=_test;
	} catch(...) {
		throw fst_error("Data_Splitter::set_n() out-of-index? error.");
	}
}
	
template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::set_n(const unsigned int _clsidx, const IDXTYPE _n)
{
	assert(_n>0); 
	try {	
		while(csplit.size()<=_clsidx) {PClassSplitter pcs(new TClassSplitter); csplit.push_back(pcs);}
		csplit[_clsidx]->n=_n;
	} catch(...) {
		throw fst_error("Data_Splitter::set_n() out-of-index? error.");
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE>
IDXTYPE Data_Splitter<INTERVALCONTAINER,IDXTYPE>::get_n(const unsigned int _clsidx) const
{
	assert(_clsidx<csplit.size());
	try {
		return csplit[_clsidx]->n;
	} catch(...) {
		throw fst_error("Data_Splitter::get_n() out-of-index? error.");
	}
}

//! Streams a splitter's print() to \p os.
template<class INTERVALCONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds) {
	return ds.print(os);
}

template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::assert_csplit() const
{
#ifdef DEBUG
	assert(csplit.size()>0);
	for(ClassIterator c=csplit.begin();c!=csplit.end();c++) {
		assert((*c)->n>0);
		assert((*c)->train);
		assert((*c)->test);
	}
#endif
}


} // namespace
#endif // FSTDATASPLITTER_H ///:~
