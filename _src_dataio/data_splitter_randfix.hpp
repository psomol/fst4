#ifndef FSTDATASPLITTERRANDFIX_H
#define FSTDATASPLITTERRANDFIX_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter_randfix.hpp
   \brief   Implements train/test data splitting: equal to Data_Splitter_RandomRandom except that the same test data is returned in each loop
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
#include "data_splitter_randrand.hpp"

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

// NOTE: this is a special-purpose modification of Data_Splitter_RandomRandom
//       that generates train data equally to Data_Splitter_RandomRandom
//       but selects always the same test data (consecutive interval at the end of each class)

/*! \brief Implements train/test data splitting: equal to Data_Splitter_RandomRandom except that the same test data is returned in each loop

    The train part is drawn at random from the leading part of each class, exactly as
    Data_Splitter_RandomRandom draws it, and varies from split to split. The test part is fixed: it
    is always the LAST perctest % of each class, in file order, and is therefore the same samples in
    every split. Use it when several splits should be compared against one common test set.

    \warning Because the test part is chosen by position rather than at random, it inherits whatever
    structure the sample order has: on data sorted by anything that correlates with a feature, the
    trailing samples are not a representative test set. */
template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
class Data_Splitter_TrainRandom_TestFixed : public Clonable_Impl<Data_Splitter_TrainRandom_TestFixed<INTERVALCONTAINER,IDXTYPE,BINTYPE>,Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE> parent; //!< the base class this splitter reuses fill_randomly()/fill()/translate() and _data/_n_max from
	typedef Clonable_Impl<Data_Splitter_TrainRandom_TestFixed,parent,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone/sharing_clone throw in Data_Splitter
	Data_Splitter_TrainRandom_TestFixed(const IDXTYPE _splits, const IDXTYPE _perctrain, const IDXTYPE _perctest, const bool randomize=false) : parent_impl(_splits, _perctrain, _perctest, randomize) {notify("Data_Splitter_TrainRandom_TestFixed constructor.");} //!< same parameters as Data_Splitter_RandomRandom; the test part is always the fixed trailing \p _perctest % interval of each class
	Data_Splitter_TrainRandom_TestFixed(const Data_Splitter_TrainRandom_TestFixed& dsp) : parent_impl(dsp) {notify("Data_Splitter_TrainRandom_TestFixed copy constructor.");} //!< copies configuration
	virtual ~Data_Splitter_TrainRandom_TestFixed() {notify("Data_Splitter_TrainRandom_TestFixed destructor.");}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Splitter_TrainRandom_TestFixed(splits=" << parent::splits << ", %train=" << parent::perctrain << ", %test=" << parent::perctest << ")"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC; //!< alias for the Data_Splitter base
	//! Draws the same random \p perctrain % train subset as the base class, but always takes the
	//! fixed trailing \p perctest % interval of each class as test data (identical across splits).
	virtual void makeRandomSplit(const IDXTYPE n, const std::shared_ptr<INTERVALCONTAINER> list_train, const std::shared_ptr<INTERVALCONTAINER> list_test) override;
};

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_TrainRandom_TestFixed<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeRandomSplit(const IDXTYPE n, const std::shared_ptr<INTERVALCONTAINER> list_train, const std::shared_ptr<INTERVALCONTAINER> list_test)
{
	try {
		assert(list_train);
		assert(list_test);
		assert(n>0);
		assert(parent::perctrain+parent::perctest<=100);
		if(!parent::_data || parent::_n_max<n) {parent::_data.reset(new BINTYPE[n]); parent::_n_max=n;}
		const BINTYPE id_empty = 0;
		const BINTYPE id_train = 1;
		const IDXTYPE tesiz=(n * parent::perctest)/100;
		const IDXTYPE trsiz= (parent::perctrain+parent::perctest==100) ? n-tesiz : (n * parent::perctrain)/100;
		assert(trsiz+tesiz<=n);
		IDXTYPE i;
		for(i=0;i<n;i++) parent::_data[i]=id_empty;

		// now fill randomly perctrain% for training; the test part is the fixed trailing interval
		// NOTE: acceleration - whichever of the train set and its complement is smaller is the one
		//       actually drawn, so the number of random draws is min{perctrain,100-perctrain-perctest}
		//       rather than always perctrain; the other side is produced by pre-filling _data
		if(parent::perctrain<=(100-parent::perctest)/2) {
			parent::fill_randomly(n,id_empty,id_train,trsiz/*count*/,0/*minidx*/,n-1-tesiz/*maxidx*/); // randomly choose train samples
		} else {
			parent::fill(n,id_empty,id_train,0/*minidx*/,n-1-tesiz/*maxidx*/);
			parent::fill_randomly(n,id_train,id_empty,n-tesiz-trsiz/*count*/,0/*minidx*/,n-1-tesiz/*maxidx*/); // randomly choose non-train samples
		}

		// now transform the binary representation to TCC::train list and create TCC::test list
		parent::translate(n,id_train,list_train);
		list_test->clear();
		if(tesiz>0) {Data_Interval<IDXTYPE> tin={n-tesiz,tesiz}; list_test->push_back(tin);}

	} catch(...) {
		throw fst_error("Data_Splitter_TrainRandom_TestFixed::makeRandomSplit() error.");
	}
}


} // namespace
#endif // FSTDATASPLITTERRANDFIX_H ///:~
