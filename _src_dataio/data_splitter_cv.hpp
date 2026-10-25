#ifndef FSTDATASPLITTERCV_H
#define FSTDATASPLITTERCV_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter_cv.hpp
   \brief   Implements train/test data splitting: by means of k-fold cross-validation
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

// CROSS-VALIDATION

/*! \brief Implements train/test data splitting: by means of k-fold cross-validation

    Yields k splits. In each, one fold of every class is the test part and the remaining folds are
    the train part; the folds are contiguous blocks of each class, taken in file order.

    Fold sizes are NOT exactly equal when the class size is not a multiple of k. Each fold takes
    (samples not yet used as a test fold)/(folds still to come), rounded down, so the earlier folds
    are the smaller ones and the last fold absorbs the remainder -- with 10 samples and k=3 the folds
    are 3, 3 and 4. The per-class test parts therefore differ slightly in size between splits.

    \warning A class with fewer samples than k produces empty test folds, silently: the fold size
    rounds down to 0 for as long as the remaining count is below the remaining fold count. Choose k
    no larger than the smallest class size.

    \warning The folds are taken by position, not at random. On data sorted by anything that
    correlates with a feature, the folds are systematically different from one another. */
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter_CV : public Clonable_Impl<Data_Splitter_CV<INTERVALCONTAINER,IDXTYPE>,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Clonable_Impl<Data_Splitter_CV,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone/sharing_clone throw in Data_Splitter
	Data_Splitter_CV(const IDXTYPE _kfold) : parent_impl(), kfold(_kfold), tmpod(0) {assert(_kfold>1); notify("Data_Splitter_CV constructor.");} //!< \p _kfold-fold cross-validation (\p _kfold must be > 1)
	Data_Splitter_CV(const Data_Splitter_CV& dsp) : parent_impl(dsp), kfold(dsp.kfold), tmpod(0) {notify("Data_Splitter_CV copy constructor.");} //!< copies configuration only; a split sequence in progress is not reproduced (prepareFirstSplit() sets the fold state up afresh), consistent with the other splitters
	virtual ~Data_Splitter_CV() {notify("Data_Splitter_CV destructor.");}
	virtual IDXTYPE getNoOfSplits() const override {return kfold;}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Splitter_CV(" << kfold << ")"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC; //!< alias for the Data_Splitter base, used by the per-class hooks
	// Template-method hooks: makeFirstSplit()/makeNextSplit() skeletons live in Data_Splitter
	virtual const char* splitter_name() const override {return "Data_Splitter_CV";}
	virtual bool prepareFirstSplit() override {
		tmpn.resize(TCC::csplit.size());
		tmpstartidx.resize(TCC::csplit.size());
		tmpod=kfold;
		return true;
	}
	virtual bool prepareNextSplit() override {return tmpod>0;}
	virtual void splitClassFirst(typename TCC::ClassIterator c, const unsigned int cidx) override {
		tmpstartidx[cidx]=0;
		tmpn[cidx]=(*c)->n;

		const IDXTYPE tcs=tmpn[cidx]/tmpod;
		Data_Interval<IDXTYPE> tri={tcs,(*c)->n-tcs};	(*c)->train->clear();	(*c)->train->push_back(tri);
		Data_Interval<IDXTYPE> tei={0,tcs};	(*c)->test->clear();	(*c)->test->push_back(tei);

		tmpstartidx[cidx]+=tcs;
		tmpn[cidx]-=tcs;
	}
	virtual void splitClassNext(typename TCC::ClassIterator c, const unsigned int cidx) override {
		const IDXTYPE tcs=tmpn[cidx]/tmpod;
		Data_Interval<IDXTYPE> tri={0,tmpstartidx[cidx]};	(*c)->train->clear();	(*c)->train->push_back(tri);
		if(tmpod>1) {Data_Interval<IDXTYPE> tri2={tmpstartidx[cidx]+tcs,(*c)->n-(tmpstartidx[cidx]+tcs)}; (*c)->train->push_back(tri2);}
		Data_Interval<IDXTYPE> tei={tmpstartidx[cidx],tcs};	(*c)->test->clear();	(*c)->test->push_back(tei);

		tmpstartidx[cidx]+=tcs;
		tmpn[cidx]-=tcs;
	}
	virtual void finishSplit() override {tmpod--;}
private:
	const IDXTYPE kfold; //!< number of folds (splits)

	IDXTYPE tmpod; //!< remaining folds still to be produced in the current split sequence (counts down from kfold, decremented by finishSplit())
	std::vector<IDXTYPE> tmpn; //!< per-class remaining sample count not yet used as a test fold
	std::vector<IDXTYPE> tmpstartidx; //!< per-class offset where the next test fold begins
};


} // namespace
#endif // FSTDATASPLITTERCV_H ///:~
