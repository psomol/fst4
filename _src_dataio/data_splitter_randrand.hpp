#ifndef FSTDATASPLITTERRANDRAND_H
#define FSTDATASPLITTERRANDRAND_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_splitter_randrand.hpp
   \brief   Implements train/test data splitting: use randomly chosen x% of data samples for training and another y% of data for testing, without overlaps
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
#include <cstdlib>
#include <ctime>
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
#include <random> // per-instance engine for the randomize==true path
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

// NOTE: generates non-overlapping TCC::train and TCC::test lists of randomly chosen samples
// NOTE: repeated traversal through split loops does not give equal output ! 
//       (i.e., generated random sample subsets are not remembered, thus two consecutive makeFirstSplit() calls give different output)

/*! \brief Implements train/test data splitting: use randomly chosen x% of data samples for training and another y% of data for testing, without overlaps, separately in each class

    Yields the configured number of splits, each drawing a fresh random selection. The train part
    holds (class size * perctrain)/100 samples, rounded down. When the two percentages add up to
    100 the test part is the whole complement of the train part (so it may hold one sample more
    than (class size * perctest)/100 would give); otherwise it holds (class size * perctest)/100
    samples, rounded down, and the remaining samples are used by neither part.

    Reproducibility. By default the draws come from the library PRNG (FST::fst_rand()), so a fixed
    seed reproduces the same splits on every platform and in every run.
    \warning Constructed with randomize==true, the splitter instead seeds its own engine from
    std::random_device: the splits then differ from run to run and results cannot be reproduced.
    Choose it only for experiments that deliberately want fresh randomness each time. */
template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
class Data_Splitter_RandomRandom : public Clonable_Impl<Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Clonable_Impl<Data_Splitter_RandomRandom,Data_Splitter<INTERVALCONTAINER,IDXTYPE>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone/sharing_clone throw in Data_Splitter
	//! \p _splits splits, each drawing \p _perctrain % of each class for training and \p _perctest % for testing (non-overlapping); \p randomize selects a per-instance engine seeded from std::random_device instead of the reproducible library PRNG (see the class warning).
	Data_Splitter_RandomRandom(const IDXTYPE _splits, const IDXTYPE _perctrain, const IDXTYPE _perctest, const bool randomize=false) : parent_impl() ,_n_max(0), perctrain(_perctrain), perctest(_perctest), splits(_splits), _randomize(randomize) {assert(splits>0); assert(perctrain+perctest<=100); current_split=0; notify("Data_Splitter_RandomRandom constructor.");}
	//! \brief Copy constructor. It deliberately does NOT inherit RNG state: in the randomize==true
	//! mode each copy seeds its own engine in prepareFirstSplit(), which is what keeps worker-thread
	//! clones from all drawing the same "random" splits. Configuration and the split counter transfer.
	Data_Splitter_RandomRandom(const Data_Splitter_RandomRandom& dsp) : parent_impl(dsp) ,_n_max(0), perctrain(dsp.perctrain), perctest(dsp.perctest), splits(dsp.splits), _randomize(dsp._randomize) {current_split=dsp.current_split; notify("Data_Splitter_RandomRandom copy constructor.");}
	virtual ~Data_Splitter_RandomRandom() {notify("Data_Splitter_RandomRandom destructor.");}
	virtual IDXTYPE getNoOfSplits() const override {assert(splits>0); return splits;}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Splitter_RandomRandom(splits=" << splits << ", %train=" << perctrain << ", %test=" << perctest << ")"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC; //!< alias for the Data_Splitter base, used by the per-class hooks
	//! Generates one class's non-overlapping random train/test split of \p n samples into \p list_train / \p list_test (perctrain %/perctest % via fill_randomly()/fill()/translate()).
	virtual void makeRandomSplit(const IDXTYPE n, const std::shared_ptr<INTERVALCONTAINER> list_train, const std::shared_ptr<INTERVALCONTAINER> list_test);

	// Template-method hooks: makeFirstSplit()/makeNextSplit() skeletons live in Data_Splitter.
	// (splitter_name() stays "Data_Splitter_RandomRandom" in the derived randfix splitter too,
	// which inherits these methods and their error messages)
	virtual const char* splitter_name() const override {return "Data_Splitter_RandomRandom";}
	virtual bool prepareFirstSplit() override {
		// Each instance seeds its own engine here, rather than sharing one process-wide generator:
		// a clone handed to a worker thread must not reproduce the parent's draws, and seeding from
		// std::random_device avoids the collisions a clock-based seed gives to instances created
		// within the same tick. The default (randomize==false) path does not come here at all --
		// it draws from the library PRNG, whose sequence is what makes a run reproducible.
		if(_randomize) _rng.seed(std::random_device{}());
		current_split=1; // (set before the per-class loop; nothing reads it until the next prepare)
		return true;
	}
	virtual bool prepareNextSplit() override {
		if(current_split<splits) {++current_split; return true;}
		current_split=0;
		return false;
	}
	virtual void splitClassFirst(typename TCC::ClassIterator c, const unsigned int) override {makeRandomSplit((*c)->n,(*c)->train,(*c)->test);}
	virtual void splitClassNext(typename TCC::ClassIterator c, const unsigned int) override {makeRandomSplit((*c)->n,(*c)->train,(*c)->test);}

	//! \brief Randomly relabels \p count of the still-\p id_empty slots of _data (within [\p minidx,\p maxidx]) to \p id_fill; draws from the per-instance engine when _randomize, else from the reproducible FST::fst_rand() stream.
	//! \pre at least \p count slots in [\p minidx,\p maxidx] still carry \p id_empty (asserted in debug builds).
	//! A draw that lands on an already-filled slot walks forward (wrapping at \p maxidx) to the next
	//! free one, so the selection is not exactly uniform over all subsets of the given size.
	void fill_randomly(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE count, const IDXTYPE minidx, const IDXTYPE maxidx);
	//! Relabels every still-\p id_empty slot of _data (within [\p minidx,\p maxidx]) to \p id_fill, unconditionally (no randomness).
	void fill(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE minidx, const IDXTYPE maxidx);
	//! Converts the binary _data labeling (id_fill vs. everything else) over indices [0,\p n) into \p lst's list of consecutive Data_Interval runs.
	void translate(const IDXTYPE n, const BINTYPE id_fill, const std::shared_ptr<INTERVALCONTAINER> lst);

	IDXTYPE _n_max; //!< actual _data size (capacity; may exceed the current class size n)
	std::unique_ptr<BINTYPE[]> _data; //!< pattern selection array: one BINTYPE label (id_empty/id_train/id_test) per sample index, reused/reallocated per class in makeRandomSplit()

	const IDXTYPE perctrain; //!< percentage of each class's samples drawn for training
	const IDXTYPE perctest; //!< percentage of each class's samples drawn for testing
	const IDXTYPE splits; //!< number of splits this splitter yields (getNoOfSplits())
	const bool _randomize; //!< true = seed a per-instance std::mt19937 engine (different splits every run); false = draw from the FST::fst_rand() stream, which makes the splits reproducible at a fixed seed
	std::mt19937 _rng; //!< per-instance RNG engine; used only when _randomize==true (see prepareFirstSplit())
	IDXTYPE current_split; //!< 1-based index of the split currently being produced; 0 outside an active split sequence
};

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::fill_randomly(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE count, const IDXTYPE minidx, const IDXTYPE maxidx)
{
	assert(_data);
	assert(_n_max>=n);
	assert(maxidx>=minidx);
	assert(maxidx<n);
	assert(count<=maxidx-minidx+1);
	// NOTE: assumes at least count id_empty slots exist in _data in interval <minidx,maxidx>
#ifdef DEBUG
	IDXTYPE dbg_cnt=0;
	for(IDXTYPE dbg_i=minidx;dbg_i<=maxidx;dbg_i++) if(_data[dbg_i]==id_empty) ++dbg_cnt; // the interval is closed, <minidx,maxidx>
	assert(count<=dbg_cnt);
#endif
	IDXTYPE piv;
	for(IDXTYPE i=0;i<count;i++) // randomly choose samples
	{
		if(_randomize) piv=minidx+(IDXTYPE)std::uniform_int_distribution<long>(0,(long)(maxidx-minidx))(_rng); // unbiased over the whole range
		else piv=minidx+(IDXTYPE)fst_rand_below((unsigned long)(maxidx-minidx+1)); // the reproducible library stream (slightly biased modulo; two composed draws once the range exceeds 32768 indexes, so every sample stays reachable)
		assert(minidx<=piv && piv<=maxidx);
		while(_data[piv]!=id_empty) {piv++; if(piv>maxidx) piv=minidx;}
		_data[piv]=id_fill;
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::fill(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE minidx, const IDXTYPE maxidx)
{
	assert(_data);
	assert(_n_max>=n);
	assert(maxidx>=minidx);
	assert(maxidx<n);
	for(IDXTYPE i=minidx;i<=maxidx;i++) if(_data[i]==id_empty) _data[i]=id_fill;
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::translate(const IDXTYPE n, const BINTYPE id_fill, const std::shared_ptr<INTERVALCONTAINER> lst)
{
	assert(lst);
	lst->clear();
	IDXTYPE piv=0, count=0;
	while(piv<n && _data[piv]!=id_fill) piv++;
	while(piv<n)
	{
		count=1;
		while(piv+count<n && _data[piv+count]==id_fill) ++count;
		Data_Interval<IDXTYPE> tin={piv,count};
		lst->push_back(tin);
		piv+=count;
		while(piv<n && _data[piv]!=id_fill) piv++;
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeRandomSplit(const IDXTYPE n, const std::shared_ptr<INTERVALCONTAINER> list_train, const std::shared_ptr<INTERVALCONTAINER> list_test)
{
	try {
		assert(list_train);
		assert(list_test);
		assert(n>0);
		// NOTE: a zero percentage is not rejected here; it yields a degenerate split with an empty
		//       train or test part. Validating the percentages is the caller's job.
		assert(perctrain+perctest<=100);
		if(!_data || _n_max<n) {_data.reset(new BINTYPE[n]); _n_max=n;}
		const BINTYPE id_empty = 0;
		const BINTYPE id_train = 1;
		const BINTYPE id_test = 2;
		const IDXTYPE trsiz=(n * perctrain)/100;
		const IDXTYPE tesiz=(n * perctest)/100;
		assert(trsiz+tesiz<=n);
		IDXTYPE i;
		for(i=0;i<n;i++) _data[i]=id_empty;

		// now fill randomly perctrain% for training and perctest% for testing
		// NOTE: acceleration - whichever of a part and its complement is smaller is the one actually
		//       drawn, so the number of random draws is min{perctrain,100-perctrain} and
		//       min{perctest,100-perctest} rather than perctrain and perctest; the other side is
		//       produced by pre-filling _data. And when perctrain+perctest==100 the test part is
		//       simply everything the train draw left over, so no second draw is needed at all.
		// NOTE: the SEQUENCE of fst_rand() calls this produces is part of what a fixed seed
		//       reproduces, so changing the branch structure changes the splits a given seed yields.
		if(perctrain<=50) {
			fill_randomly(n,id_empty,id_train,trsiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose train samples
		} else {
			fill(n,id_empty,id_train,0/*minidx*/,n-1/*maxidx*/);
			fill_randomly(n,id_train,id_empty,n-trsiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose non-train samples
		}
		if(perctrain+perctest==100) {
			fill(n,id_empty,id_test,0/*minidx*/,n-1/*maxidx*/);
		} else if(perctest<=(100-perctrain)/2) {
			fill_randomly(n,id_empty,id_test,tesiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose test samples
		} else {
			fill(n,id_empty,id_test,0/*minidx*/,n-1/*maxidx*/);
			fill_randomly(n,id_test,id_empty,n-trsiz-tesiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose non-test samples
		}

		// now transform the binary representation to TCC::train and TCC::test lists
		translate(n,id_train,list_train);
		translate(n,id_test,list_test);
		
	} catch(...) {
		throw fst_error("Data_Splitter_RandomRandom::makeRandomSplit() error.");
	}
}

} // namespace
#endif // FSTDATASPLITTERRANDRAND_H ///:~
