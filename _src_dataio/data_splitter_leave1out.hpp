#ifndef FSTDATASPLITTERLEAVE1OUT_H
#define FSTDATASPLITTERLEAVE1OUT_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter_leave1out.hpp
   \brief   Implements train/test data splitting: by means of leave-one-out
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
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "data_splitter.hpp"

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

// LEAVE-ONE-OUT

/*! \brief Implements train/test data splitting: by means of leave-one-out

    Yields one split per sample of the WHOLE data set, not per class: the splits walk class 0's
    samples one by one, then class 1's, and so on. In each split exactly one sample is the test
    part and everything else -- including all of every other class -- is the train part. The
    non-pivot classes therefore get an empty test part in that split, which is what a per-class
    accuracy accumulation has to expect.

    The total number of splits equals the total number of samples, so leave-one-out is the most
    expensive validation protocol available: each of them retrains the model. Classes with no
    samples are skipped. */
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter_Leave1Out : public Clonable_Impl<Data_Splitter_Leave1Out<INTERVALCONTAINER,IDXTYPE>,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Clonable_Impl<Data_Splitter_Leave1Out,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone/sharing_clone throw in Data_Splitter
	Data_Splitter_Leave1Out() : parent_impl() {notify("Data_Splitter_Leave1Out (empty) constructor.");}
	//! pivotclass deliberately NOT copied: it is an iterator into the SOURCE's csplit list, which
	//! would be invalid here — split-loop state does not transfer, and prepareFirstSplit() must be
	//! called on the copy to (re)initialize it
	Data_Splitter_Leave1Out(const Data_Splitter_Leave1Out& dsp) : parent_impl(dsp), pivot(0) {notify("Data_Splitter_Leave1Out copy constructor.");}
	virtual ~Data_Splitter_Leave1Out() {notify("Data_Splitter_Leave1Out destructor.");}
	virtual IDXTYPE getNoOfSplits() const override;

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Splitter_Leave1Out()"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC; //!< alias for the Data_Splitter base, used by the per-class hooks
	// Template-method hooks: makeFirstSplit()/makeNextSplit() skeletons live in Data_Splitter
	virtual const char* splitter_name() const override {return "Data_Splitter_Leave1Out";}
	virtual bool prepareFirstSplit() override {
		pivotclass=TCC::csplit.begin();
		while(pivotclass!=TCC::csplit.end() && (*pivotclass)->n==0) pivotclass++;
		if(pivotclass==TCC::csplit.end()) return false;
		pivot=0;
		return true;
	}
	virtual bool prepareNextSplit() override {
		if(pivotclass==TCC::csplit.end()) return false;
		if(pivot>=(*pivotclass)->n) {
			pivot=0;
			pivotclass++;
			while(pivotclass!=TCC::csplit.end() && (*pivotclass)->n==0) pivotclass++;
			if(pivotclass==TCC::csplit.end()) return false;
		}
		return true;
	}
	virtual void splitClassFirst(typename TCC::ClassIterator c, const unsigned int) override {
		if(c==pivotclass) {
			Data_Interval<IDXTYPE> tri={pivot+1,(*c)->n-1};	(*c)->train->clear();	(*c)->train->push_back(tri);
			Data_Interval<IDXTYPE> tei={pivot,1};	(*c)->test->clear();	(*c)->test->push_back(tei);
		} else {
			Data_Interval<IDXTYPE> tri={0,(*c)->n};	(*c)->train->clear();	(*c)->train->push_back(tri); // all
			(*c)->test->clear();
		}
	}
	virtual void splitClassNext(typename TCC::ClassIterator c, const unsigned int) override {
		if(c==pivotclass) {
			(*c)->train->clear();
			if(pivot>0) {Data_Interval<IDXTYPE> tri={0,pivot};	(*c)->train->push_back(tri);}
			if(pivot<(*pivotclass)->n-1) {Data_Interval<IDXTYPE> tri2={pivot+1,(*c)->n-(pivot+1)};	(*c)->train->push_back(tri2);}
			Data_Interval<IDXTYPE> tei={pivot,1};	(*c)->test->clear();	(*c)->test->push_back(tei);
		} else {
			Data_Interval<IDXTYPE> tri={0,(*c)->n};	(*c)->train->clear();	(*c)->train->push_back(tri); // all
			(*c)->test->clear();
		}
	}
	virtual void finishSplit() override {++pivot;}
private:
	IDXTYPE pivot; //!< which sample (index within pivotclass) to hold out as the current test pattern
	typename TCC::ClassIterator pivotclass; //!< which class the currently held-out sample belongs to
};

template<class INTERVALCONTAINER, typename IDXTYPE>
IDXTYPE Data_Splitter_Leave1Out<INTERVALCONTAINER,IDXTYPE>::getNoOfSplits() const
{
	IDXTYPE sum=0;
	for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) sum+=(*c)->n;
	return sum;
}


} // namespace
#endif // FSTDATASPLITTERLEAVE1OUT_H ///:~
