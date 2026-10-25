#ifndef FSTDATASPLITTER5050_H
#define FSTDATASPLITTER5050_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter_5050.hpp
   \brief   Implements train/test data splitting: first 50% of original data for training, second 50% for testing
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

// FIFTY-FIFTY

/*! \brief Implements train/test data splitting: first 50% of original data for training, second 50% for testing, each (data)class split separately

    Yields exactly one split. A class of odd size gives the extra sample to the TEST part (the train
    size is the size divided by two, rounded down).

    \warning The split is by position, not at random: it depends entirely on the order the samples
    happen to have in the file. Data sorted by anything that correlates with the class -- or with a
    feature -- produces two systematically different halves and a misleading accuracy estimate.
    Use Data_Splitter_RandomRandom or a cross-validation splitter when the ordering is not known to
    be arbitrary. */
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter_5050 : public Clonable_Impl<Data_Splitter_5050<INTERVALCONTAINER,IDXTYPE>,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Clonable_Impl<Data_Splitter_5050,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone/sharing_clone throw in Data_Splitter
	Data_Splitter_5050() : parent_impl() {notify("Data_Splitter_5050 constructor.");}
	Data_Splitter_5050(const Data_Splitter_5050& dsp) : parent_impl(dsp) {notify("Data_Splitter_5050 copy constructor.");} //!< copies configuration (this splitter is parameterless, so there is nothing else to copy)
	virtual ~Data_Splitter_5050() {notify("Data_Splitter_5050 destructor.");}
	virtual IDXTYPE getNoOfSplits() const override {return 1;}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Splitter_5050()"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC; //!< alias for the Data_Splitter base, used by the per-class hook
	// Template-method hooks: makeFirstSplit()/makeNextSplit() skeletons live in Data_Splitter
	virtual const char* splitter_name() const override {return "Data_Splitter_5050";}
	virtual bool prepareNextSplit() override {return false;} // single-split splitter
	virtual void splitClassFirst(typename TCC::ClassIterator c, const unsigned int) override {
		const IDXTYPE trsiz=(*c)->n/2;
		const IDXTYPE tesiz=(*c)->n-trsiz;
		Data_Interval<IDXTYPE> tri={0,trsiz},tei={trsiz,tesiz};
		(*c)->train->clear();	(*c)->train->push_back(tri);
		(*c)->test->clear();	(*c)->test->push_back(tei);
	}
};


} // namespace
#endif // FSTDATASPLITTER5050_H ///:~
