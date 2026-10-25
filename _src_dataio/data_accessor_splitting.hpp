#ifndef FSTDATAACCESSORSPLITTING_H
#define FSTDATAACCESSORSPLITTING_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_accessor_splitting.hpp
   \brief   Defines support for data splitting
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
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "data_accessor.hpp"
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

/*! \brief partly-abstract class, defines support for data splitting

\details Data structures in Data_Accessor_Splitting are directly used in
the splitting mechanism that enables structured access to data.
In order to keep the implementation of data splitters
(specializations of Data_Splitter) as simple as possible
for the user, we moved most of the technicalities here.

Splitting depth. The accessor is constructed with an ordered list of splitters, one per nested
splitting level; the length of that list is the splitting depth of the data access. Depth 0 is
the outermost level and splits the whole data set; depth 1 splits the TRAIN part of the current
depth-0 split, depth 2 the TRAIN part of the current depth-1 split, and so on. Only one level is
active at a time: setSplittingDepth() selects it, and getFirstSplit()/getNextSplit() and the
block-access methods then all refer to that level. A two-level stack is the usual arrangement for
selection experiments -- the criterion works inside the depth-1 splits while the depth-0 test part
is held back as an independent validation set.

Correct state of key data structures in Data_Accessor_Splitting is as follows:

- dsp represents a list of Data_Splitters while each splitter keeps
      for each class two pointers: a pointer to the list of train and test data (lists of data intervals).
      The splitters are, however, not the owners of the actual train and test lists,
      these are allocated within Data_Accessor_Splitting.
      
The splitting mechanism needs not just one train-test data structure pair,
but two - the second pair denoted _reduced_train and _reduced_test. The
"reduced" pair is constructed from the "base" pair separately in each
splitting level, so as to correctly represent the subset of data defined by
the respective splitter (each deeper level further reduces access to data
visible in the preceeding level). The actual access to data through
getFirstBlock() and getNextBlock() is commanded by data intervals stored in
the "reduced" pair of interval lists only.

The two pairs of interval lists exist separately for each splitting level
and each data class. In Data_Accessor_Splitting they are collected in the DataSplit
subclass, of which the required number of instances is kept in the "splits"
container. In correct representation "splits" must contain 
[number of classes]*[number of splitting levels] DataSplit instances.

The DataSplit that represent top splitting level differ from the deeper
level - the "reduced" pair of lists is actually just referencing the "base"
pair. This is because data indexes as produced by splitters are valid
indexes usable to access the data. In deeper splitting levels this is not so
because splitter indexes must be treated are relative to the data possibly restricted
in higher level. Transforming the relative indexes to absolute indexes is
achieved through the "reduce" method implemented in Data_Intervaller.
In non-top splitting levels before data can be accessed, the "base" indexes/intervals
are first transformed using the "reduce" method with the result stored
in "_reduced" train and test lists.

The "base" train and test lists allocated here in Data_Accessor_Splitting need to
be interlinked with respective data splitters. The splitters do not hold
any allocated structures, they re-direct their output to the "base"
train and test lists kept in Data_Accessor_Splitting, to enable data accessing
routines to transform the indexes by means of "reduce" whenever needed
and subsequently to access the correct subset of data.
*/
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting : public Data_Accessor<DATATYPE,IDXTYPE>
{
	// NOTE: the calling convention is the same for every nested data access loop (as in cross-validation):
	//   all loops call the same methods (just get*Split, get*Block, get*Pattern), and the loop level
	//   they refer to is selected beforehand with setSplittingDepth()
	// Access to data is defined using indexes, assuming data is accessible by random access
	// Support for arbitrary data splitting (train/test, cross-validation, hold-out, etc., by means
	// of arbitrary Data_Splitter objects) is implemented here
	public:
		typedef std::shared_ptr<Data_Splitter<INTERVALCONTAINER,IDXTYPE> > PSplitter;  //!< shared pointer to one Data_Splitter (one nested splitting level)
		typedef std::shared_ptr<std::vector<PSplitter> > PSplitters;  //!< shared pointer to the ordered list of splitters, one per nested splitting depth (depth 0 = outermost)
		typedef const DATATYPE* PPattern;  //!< one data sample: a raw pointer to its (read-only) feature vector
		//! \brief Constructs an (uninitialized) accessor bound to the splitter stack \p _dsp; concrete subclasses must still call initialize() once dimensionality/class sizes are known.
		Data_Accessor_Splitting(const PSplitters _dsp);
		virtual ~Data_Accessor_Splitting(){notify("Data_Accessor_Splitting destructor.");}
		
		virtual unsigned int getNoOfClasses() const override;          //!< returns number of classes
		virtual unsigned int getNoOfFeatures() const override;         //!< returns data dimensionality
		
		virtual IDXTYPE getClassSize(const unsigned int c) const override;    //!< returns size (number of samples in) of class c
		virtual IDXTYPE getClassSizeSum() const override;    //!< returns summed size (number of samples in) of all classes, i.e., no. of all patterns in data
		
		virtual void setClass(const int c) override {assert(c>=0); active_class = c;} //!< sets active class -> from now on only data from class c will be considered
		virtual int  getClass() const override {return active_class;}

		//! \brief Selects the nested splitting level (0 = outermost/depth-0 splitter, 1 = the next-nested level, etc.) that subsequent getFirstSplit()/getNextSplit()/getFirstBlock()/getNextBlock() calls operate on.
		//! \note DEBUG builds assert \p depth is within the splitter stack and that every shallower level already has an active split (i.e. getFirstSplit()/getNextSplit() was already called there).
		void setSplittingDepth(const unsigned int depth);
		unsigned int getSplittingDepth() const {return splitting_depth;} //!< the nested splitting level last set via setSplittingDepth()
		        
		virtual unsigned int getNoOfSplits() const override;  //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getFirstSplit() override;                //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getNextSplit() override;                 //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual unsigned int getSplitIndex() const override;  //!< data access iteration (to support, e.g., loops in cross-validation)

		// NOTE: in the block accessing methods below, use a different loopdepth whenever two or more loops
		//       (of any DataPart type) overlap in time, otherwise their cursor state collides

		//! \warning getNoOfBlocks is not to be relied upon due to possible limitations in some implementations
		virtual IDXTYPE getNoOfBlocks(const DataPart ofwhat) const override;
		//! returns pointer to first consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0) override=0;
		//! returns pointer to next consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0) override=0;
		//! returns index of the current consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getBlockIndex(const unsigned int loopdepth=0) const override;
		//! returns number of patterns in all consecutive blocks of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getNoOfPatterns(const DataPart ofwhat) const override;
		//! enables change of meaning of DataPart types, for use in specialized data access scenarios like in bias predicting wrappers
		virtual void substitute(const DataPart source, const DataPart target) override;
		//! resets standard DataPart types' meaning
		virtual void resubstitute() override;

		virtual std::ostream& print(std::ostream& os) const override; //!< writes the splitter stack's print() output (or "Data_Accessor_Splitting()" if the stack is empty) to \p os
	protected:
		//! \brief "Strong" (1:1, deep) copy constructor: stateless_clone()s every splitter and re-assign()s it to this instance's own (deep-copied via #splits) interval lists.
		//! \note To be used in clone() overrides of derived classes.
		Data_Accessor_Splitting(const Data_Accessor_Splitting &da);      // to be used in clone() in derived classes
	protected:
		typedef std::vector<unsigned int> CLASSSIZES;  //!< per-class pattern counts, indexed by class index; the format initialize() takes
		/*! sets-up memory structures needed in the splitting mechanism
		    data access is not needed here, the structures work with indexes
		    only - the only information needed is dimensionality and sizes of data classes */
		void initialize(const unsigned int _features, const CLASSSIZES &_classes);
		CLASSSIZES classes;  //!< pattern count per class, as passed to initialize()
		unsigned int features;  //!< data dimensionality, as passed to initialize()

		// DataPart mapping
		DataPart mappedTRAIN;      //!< current remapping target of TRAIN (see substitute()/resubstitute()); defaults to TRAIN
		DataPart mappedTEST;       //!< current remapping target of TEST (see substitute()/resubstitute()); defaults to TEST
		DataPart mappedTRAINTEST;  //!< current remapping target of TRAINTEST (see substitute()/resubstitute()); defaults to TRAINTEST
		DataPart mappedALL;        //!< current remapping target of ALL (see substitute()/resubstitute()); defaults to ALL
		DataPart mappedDataPart(const DataPart ofwhat) const;  //!< looks up \p ofwhat's current remapping target (mappedTRAIN/mappedTEST/mappedTRAINTEST/mappedALL); NONE maps to itself

		// splitting support
		PSplitters dsp;  //!< the splitter stack this accessor was constructed with, one Data_Splitter per nested splitting depth
		typedef const Data_Interval<IDXTYPE>* DATAINTERVAL;  //!< pointer to a read-only Data_Interval record (as yielded by the protected Data_Interval-returning getFirstBlock()/getNextBlock() overloads)
		//! returns Data_Interval record representing the first consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getFirstBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth=0);
		//! returns Data_Interval record representing the next consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getNextBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth=0);

		typedef std::shared_ptr<INTERVALCONTAINER> PIntervaller;  //!< shared pointer to one class's list of train/test data intervals at one splitting depth
		//! Data splitting support structure; holds one set of intervals (train, test) per each splitting depth and (data)class
		class DataSplit {
		public:
			DataSplit() {_train.reset(); _test.reset(); _reduced_train.reset(); _reduced_test.reset(); notify("DataSplit empty constructor.");} //!< all interval lists start unset
			DataSplit(const DataSplit& cs);      //!< "strong" (deep) copy constructor; see the out-of-line definition's doc (for use in clone())
			DataSplit(const DataSplit& cs, int); //!< "weak" (referencing) copy constructor; see the out-of-line definition's doc (for use in sharing_clone())
			~DataSplit() {notify("DataSplit destructor.");}
			PIntervaller _train;          //!< this depth/class's base (splitter-produced) train intervals
			PIntervaller _test;           //!< this depth/class's base (splitter-produced) test intervals
			PIntervaller _reduced_train;  //!< depth 0: same list as #_train; deeper: #_train's indexes reduced (translated) into the enclosing depth's absolute indexes
			PIntervaller _reduced_test;   //!< depth 0: same list as #_test; deeper: #_test's indexes reduced (translated) into the enclosing depth's absolute indexes
			Data_Interval<IDXTYPE> _all;   //!< to enable passing of block reference when DataPart ofwhat==ALL
		};
		std::vector<std::vector<DataSplit> > splits;    //!< one set of splitters per each splitting depth and class

		std::vector<IDXTYPE> enum_split;           //!< current split..  0~none
		std::vector<std::vector<IDXTYPE> > enum_block;  //!< current block loop..  0~none
		std::vector<std::vector<DataPart> > tt_phase;   //!< in current block loop.. for DataPart==TRAINTEST indicates which half is being delivered: TRAIN, then TEST once the train blocks run out, NONE when the loop is finished

		unsigned int splitting_depth;           //!< the nested splitting level the get*Split()/get*Block() calls currently refer to; set via setSplittingDepth()

		int active_class;                     //!< denotes from which class the get*Train* get*Validate* get*Test* methods return patterns, to be set using setClass()
		
		// for debugging
		bool _initialize_called;  //!< set true by initialize(); read by is_initialized()
		bool is_initialized() const {return _initialize_called;}  //!< whether initialize() has been called on this instance
		//! \brief DEBUG-only: asserts the internal structures (#dsp, #splits, #enum_split, #enum_block, #tt_phase, #classes) are non-empty, consistently sized, and every #splits entry has its train/test/reduced interval lists allocated; a no-op in release builds.
		//! \param splitting_check -1 (default) skips the per-level sample-count check; a value >=0 additionally asserts every splitting level 0..\p splitting_check has a positive sample count set (i.e. set_n() was already called there)
		void assert_splits(const int splitting_check=-1) const;
};


// ============================================ PUBLIC ===============================================

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting(const PSplitters _dsp) :
	mappedTRAIN(TRAIN), mappedTEST(TEST), mappedTRAINTEST(TRAINTEST), mappedALL(ALL)
{
	notify("Data_Accessor_Splitting constructor.");
	_initialize_called=false;
	features=0;
	assert(_dsp->size()>0);
	dsp=_dsp;
	active_class=-1; 
	splitting_depth=0;
	splits.assign(_dsp->size(),std::vector<DataSplit>()); 
	enum_split.assign(_dsp->size(),0);
	enum_block.assign(_dsp->size(),std::vector<IDXTYPE>()); 
	tt_phase.assign(_dsp->size(),std::vector<DataPart>()); 
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfClasses() const
{
	return classes.size();
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfFeatures() const
{
	return features;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getClassSize(const unsigned int c) const
{
	assert(c<classes.size());
	return classes[c];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getClassSizeSum() const
{
	IDXTYPE patterns=0;
	for(unsigned int c=0;c<classes.size();c++) patterns+=classes[c];
	return patterns;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::setSplittingDepth(const unsigned int depth)
{
#ifdef DEBUG
	assert(depth<dsp->size());
	assert(enum_split.size()==dsp->size());
	for(unsigned int n=0;n<depth;n++) assert(enum_split[n]>0);
#endif
	splitting_depth=depth;
} 

// NOTE: precondition: initialize() must have completed before this is meaningful — it assigns
//       the per-class interval containers and calls set_n() on the depth-0 splitters (deeper
//       splitters get set_n() in the course of split enumeration). The asserts below make a
//       violation fail fast in debug builds.
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfSplits() const
{
	assert(dsp); // consider also checking the vector contents
	assert(splitting_depth<dsp->size());
	assert(splitting_depth<splits.size() && !splits[splitting_depth].empty()); // initialize() has run
	return (*dsp)[splitting_depth]->getNoOfSplits();
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstSplit()
{
#ifdef DEBUG
	assert(dsp); // consider also checking the vector contents
	assert(is_initialized());
	assert(splitting_depth<(unsigned int)dsp->size());
	for(unsigned int n=0;n<splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(/*n not set in splitters yet*/);
#endif
	bool result=true;
	if(splitting_depth>0)
		for(unsigned int c=0; c<classes.size(); c++)
			{splits[splitting_depth][c]._all.startidx=0; (*dsp)[splitting_depth]->set_n(c, splits[splitting_depth][c]._all.count=splits[splitting_depth-1][c]._train->sum() );}
	else
		for(unsigned int c=0; c<classes.size(); c++)
			{splits[splitting_depth][c]._all.startidx=0; (*dsp)[splitting_depth]->set_n(c, splits[splitting_depth][c]._all.count=classes[c] );}

	if(!(*dsp)[splitting_depth]->makeFirstSplit()) result=false;
	else 
	if(splitting_depth>0) for(unsigned int c=0; c<classes.size(); c++) {
		// translate the new level's intervals (relative to the enclosing level's TRAIN part) into
		// absolute indexes through the enclosing level's ABSOLUTE train list (_reduced_train): at
		// depth 1 that list is the depth-0 _train itself, at depth 2 and deeper _train would be
		// relative to depth-2's train part and the translation would land on the wrong samples
		splits[splitting_depth-1][c]._reduced_train->reduce(splits[splitting_depth][c]._train,splits[splitting_depth][c]._reduced_train);
		splits[splitting_depth-1][c]._reduced_train->reduce(splits[splitting_depth][c]._test,splits[splitting_depth][c]._reduced_test);
	}
	if(result) enum_split[splitting_depth]=1; else enum_split[splitting_depth]=0;
	for(unsigned int n=splitting_depth+1;n<dsp->size();n++) enum_split[n]=0; // reset/cancel all deeper splits
	
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextSplit()
{
#ifdef DEBUG
	assert(dsp); // consider also checking the vector contents
	assert(is_initialized());
	assert(splitting_depth<dsp->size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(splitting_depth);
#endif
	bool result=true;
	if(!(*dsp)[splitting_depth]->makeNextSplit()) result=false;
	else 
	if(splitting_depth>0) for(unsigned int c=0; c<classes.size(); c++) {
		splits[splitting_depth-1][c]._reduced_train->reduce(splits[splitting_depth][c]._train,splits[splitting_depth][c]._reduced_train);
		splits[splitting_depth-1][c]._reduced_train->reduce(splits[splitting_depth][c]._test,splits[splitting_depth][c]._reduced_test);
	}
	if(result) enum_split[splitting_depth]++; else enum_split[splitting_depth]=0;
	for(unsigned int n=splitting_depth+1;n<dsp->size();n++) enum_split[n]=0; // reset/cancel all deeper splits
		
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getSplitIndex() const
{
	assert(splitting_depth<dsp->size());
	return enum_split[splitting_depth];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfBlocks(const DataPart ofwhat) const
{
	assert(dsp); // consider also checking the vector contents
	assert(is_initialized());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert(splitting_depth<dsp->size());
	assert_splits(splitting_depth);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {return splits[splitting_depth][active_class]._reduced_train->size();}
		case TEST: {return splits[splitting_depth][active_class]._reduced_test->size();}
		case TRAINTEST: {return splits[splitting_depth][active_class]._reduced_train->size()+splits[splitting_depth][active_class]._reduced_test->size();}
		case ALL: {if(splitting_depth>0) return splits[splitting_depth-1][active_class]._reduced_train->size(); else return 1;}
		case NONE: {return 0;}
	} 
	return 0;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getBlockIndex(const unsigned int loopdepth) const
{
	assert(dsp); // consider also checking the vector contents
	assert(splitting_depth<dsp->size());
	assert(loopdepth<enum_block[splitting_depth].size());
	return enum_block[splitting_depth][loopdepth];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfPatterns(const DataPart ofwhat) const
{
	assert(is_initialized());
	assert(splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert_splits(splitting_depth);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {return splits[splitting_depth][active_class]._reduced_train->sum();}
		case TEST: {return splits[splitting_depth][active_class]._reduced_test->sum();}
		case TRAINTEST: {return splits[splitting_depth][active_class]._reduced_train->sum()+splits[splitting_depth][active_class]._reduced_test->sum();}
		case ALL: {if(splitting_depth>0) return splits[splitting_depth-1][active_class]._reduced_train->sum(); 
			else return classes[active_class];}
		case NONE: {return 0;}
	} 
	return 0;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::substitute(const DataPart source, const DataPart target)
{
	assert(source==TRAIN || source==TEST || source==TRAINTEST || source==ALL);
	assert(target==TRAIN || target==TEST || target==TRAINTEST || target==ALL);
	switch(source){
		case TRAIN:	mappedTRAIN=target; break;
		case TEST: mappedTEST=target; break;
		case TRAINTEST: mappedTRAINTEST=target; break;
		case ALL: mappedALL=target; break;
		case NONE: break;
	}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::resubstitute()
{
	mappedTRAIN=TRAIN;
	mappedTEST=TEST;
	mappedTRAINTEST=TRAINTEST;
	mappedALL=ALL;
}


// ============================================ PROTECTED ===============================================

// strong copy-constructor, creates 1:1 copy of all structures
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting(const Data_Accessor_Splitting &da) :
	classes(da.classes),
	features(da.features),
	mappedTRAIN(da.mappedTRAIN), 
	mappedTEST(da.mappedTEST), 
	mappedTRAINTEST(da.mappedTRAINTEST), 
	mappedALL(da.mappedALL),
	splits(da.splits),
	enum_split(da.enum_split),
	enum_block(da.enum_block),
	tt_phase(da.tt_phase),
	splitting_depth(da.splitting_depth),
	active_class(da.active_class),
	_initialize_called(da._initialize_called)
{
	notify("Data_Accessor_Splitting() copy constructor.");
	notify("DAS::copy-constructor, before da.assert_splits()");
	da.assert_splits();
	notify("DAS::copy-constructor, after da.assert_splits()");
	assert(da.dsp);
	assert(da.dsp->size()>0);
	dsp.reset(new std::vector<PSplitter>);
	for(unsigned int n=0; n<da.dsp->size();n++)
	{
		
		PSplitter ps(dynamic_cast<Data_Splitter<INTERVALCONTAINER,IDXTYPE>* >((*da.dsp)[n]->stateless_clone()));
		for(unsigned int c=0; c<classes.size(); c++)
			ps->assign(c,splits[n][c]._train,splits[n][c]._test);
		dsp->push_back(ps);
	}	
	notify("DAS::copy-constructor, before assert_splits()");
	assert_splits();
	notify("DAS::copy-constructor, after assert_splits()");
}


template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initialize(const unsigned int _features, const CLASSSIZES &_classes)
{
	notify("Data_Accessor_Splitting()::initialize.");
	if(is_initialized()) notify("Re-initializing Data_Accessor_Splitting.");
	classes.resize( _classes.size() );
	copy(_classes.begin(), _classes.end(), classes.begin() );
	features=_features;
	
	for(unsigned int n=0;n<dsp->size();n++) // number of nested splitting levels
	{
		splits[n].clear();
		notify("Data_Accessor_Splitting()::initialize to resize splits.");
		splits[n].resize(classes.size());
		notify("Data_Accessor_Splitting()::initialize splits resized.");
		for(unsigned int c=0;c<classes.size();c++) 
		{
			DataSplit &cs=splits[n][c];
			cs._train.reset(new INTERVALCONTAINER());
			cs._test.reset(new INTERVALCONTAINER());
			(*dsp)[n]->assign(c,cs._train,cs._test);
			if(n==0) {
				(*dsp)[0]->set_n(c,classes[c]);
				cs._reduced_train=cs._train;
				cs._reduced_test=cs._test;
			} else {
				cs._reduced_train.reset(new INTERVALCONTAINER());
				cs._reduced_test.reset(new INTERVALCONTAINER());
			}
			// NOTE: set_n() is done here for depth 0 only, where the sample count per class is
			//       simply the class size. At deeper levels the count depends on the enclosing
			//       level's current TRAIN part, so getFirstSplit() re-issues set_n() on the
			//       splitter of that level before every split enumeration.
		}
	}
	_initialize_called=true;
	notify("Data_Accessor_Splitting()::initialize finished.");
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
DataPart Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::mappedDataPart(const DataPart ofwhat) const
{
	switch(ofwhat){
		case TRAIN:	return mappedTRAIN;
		case TEST: return mappedTEST;
		case TRAINTEST: return mappedTRAINTEST;
		case ALL: return mappedALL;
		case NONE: return NONE;
	}
	return ofwhat;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth)
{
#ifdef DEBUG
	assert(dsp); // consider also checking the vector contents
	assert(is_initialized());
	assert(splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(splitting_depth);
#endif
	if(enum_block[splitting_depth].size()<=loopdepth) enum_block[splitting_depth].resize(loopdepth+1);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {tmp=splits[splitting_depth][active_class]._reduced_train->getFirstBlock(loopdepth); break;}
		case TEST: {tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth); break;}
		case TRAINTEST: {
			if(tt_phase[splitting_depth].size()<=loopdepth) tt_phase[splitting_depth].resize(loopdepth+1);
			tt_phase[splitting_depth][loopdepth]=TRAIN; 
			tmp=splits[splitting_depth][active_class]._reduced_train->getFirstBlock(loopdepth); 
			if(tmp==nullptr) {
				tt_phase[splitting_depth][loopdepth]=TEST; 
				tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth); 
			}
			if(tmp==nullptr) tt_phase[splitting_depth][loopdepth]=NONE;
			break;}
		case ALL: {
			if(splitting_depth>0) tmp=splits[splitting_depth-1][active_class]._reduced_train->getFirstBlock(loopdepth);
			else tmp=&(splits[0][active_class]._all);
			break;}
		case NONE: {tmp=nullptr;}
	}
	if(tmp!=nullptr) enum_block[splitting_depth][loopdepth]=1; else {enum_block[splitting_depth][loopdepth]=0; return false;}
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth)
{
#ifdef DEBUG
	assert(dsp); // consider also checking the vector contents
	assert(is_initialized());
	assert(splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert(loopdepth<enum_block[splitting_depth].size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert(enum_block[splitting_depth][loopdepth]>0);
	assert_splits(splitting_depth);
#endif
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {tmp=splits[splitting_depth][active_class]._reduced_train->getNextBlock(loopdepth); break;}
		case TEST: {tmp=splits[splitting_depth][active_class]._reduced_test->getNextBlock(loopdepth); break;}
		case TRAINTEST: {
			assert(loopdepth<tt_phase[splitting_depth].size());
			assert(tt_phase[splitting_depth][loopdepth]==TRAIN || tt_phase[splitting_depth][loopdepth]==TEST);
			if(tt_phase[splitting_depth][loopdepth]==TRAIN) {
				tmp=splits[splitting_depth][active_class]._reduced_train->getNextBlock(loopdepth); 
				if(tmp==nullptr) {
					tt_phase[splitting_depth][loopdepth]=TEST;
					tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth);
				}
			} else {
				tmp=splits[splitting_depth][active_class]._reduced_test->getNextBlock(loopdepth);
			}
			if(tmp==nullptr) tt_phase[splitting_depth][loopdepth]=NONE;
			break;}
		case ALL: {if(splitting_depth>0) tmp=splits[splitting_depth-1][active_class]._reduced_train->getNextBlock(loopdepth); else tmp=nullptr; break;}
		case NONE: {tmp=nullptr;}
	}
	if(tmp!=nullptr) enum_block[splitting_depth][loopdepth]++; else {enum_block[splitting_depth][loopdepth]=0; return false;}
	return true;
}

//! \brief "Strong" copy constructor: deep-copies \p cs's interval lists into freshly allocated INTERVALCONTAINER instances.
//! \note Used for clone() (1:1 independent copy); be careful when used for cloning in concurrent contexts.
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit::DataSplit(const DataSplit& cs)
{
	notify("DataSplit strong copy constructor.");
	if(cs._train) _train.reset(new INTERVALCONTAINER(*cs._train));
	if(cs._test) _test.reset(new INTERVALCONTAINER(*cs._test));
	if(cs._reduced_train == cs._train) // && cs._reduced_test == cs._test
	{
		notify("DataSplit _reduced_* to be linked");
		_reduced_train=_train;
		_reduced_test=_test;
	} else {
		notify("DataSplit _reduced_* to be re-allocated");
		if(cs._reduced_train) _reduced_train.reset(new INTERVALCONTAINER(*cs._reduced_train));
		if(cs._reduced_test) _reduced_test.reset(new INTERVALCONTAINER(*cs._reduced_test));
	}
	_all.startidx=cs._all.startidx; _all.count=cs._all.count;
}

//! \brief "Weak" copy constructor: copies only the shared_ptr references to \p cs's interval lists, not the pointed-to data.
//! \note Used for sharing_clone() (referencing, not owning, the source's structures); be careful when used for cloning in concurrent contexts.
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit::DataSplit(const DataSplit& cs, int)
{
	notify("DataSplit weak copy constructor.");
	_train=cs._train;
	_test=cs._test;
	_reduced_train=cs._reduced_train;
	_reduced_test=cs._reduced_test;
	_all.startidx=cs._all.startidx; _all.count=cs._all.count;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	if(dsp) {
		if(dsp->size()>0) {
			os << "Data_Accessor_Splitting() splitters: ";
			typename std::vector<PSplitter>::const_iterator iter;
			for(iter=dsp->begin();iter!=dsp->end();iter++) {
				if(*iter) os << **iter << " "; else os << "nullptr ";
			}
		} else os << "Data_Accessor_Splitting()"; 
	}
	return os;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::assert_splits(const int splitting_check) const
{
	// assumes Data_Accessor_Splitting() has been initialized; all data structures are checked to be valid and non-empty
#ifdef DEBUG
	assert(features>0); // No. of features
	assert(classes.size()>0); // No. of classes
	assert(dsp); // consider also checking the vector contents
	assert(dsp->size()>0); // No. of splitting levels
	// the following not to be checked here
	
	assert(splitting_check<(const int)dsp->size()); 
	// splitting_check == -1 by default
	// splitting_check > -1 checks that in each level 0..splitting_check the number of samples is set > 0

	// check correctness of data strucutre sizes
	assert(enum_split.size()==dsp->size());
	assert(splits.size()==dsp->size());
	assert(enum_block.size()==dsp->size());
	assert(tt_phase.size()==dsp->size());
	for(unsigned int n=0;n<dsp->size();n++) assert(splits[n].size()==classes.size());
	
	for(unsigned int n=0;n<dsp->size();n++)
	{
		for(unsigned int c=0;c<classes.size();c++) 
		{
			if((int)n<=splitting_check) assert((*dsp)[n]->get_n(c)>0);
			assert(splits[n][c]._train);
			assert(splits[n][c]._test);
			assert(splits[n][c]._reduced_train);
			assert(splits[n][c]._reduced_test);
			if(n==0) {
				assert(splits[n][c]._reduced_train==splits[n][c]._train);
				assert(splits[n][c]._reduced_test==splits[n][c]._test);
			} else {
				assert(splits[n][c]._reduced_train!=splits[n][c]._train);
				assert(splits[n][c]._reduced_test!=splits[n][c]._test);
			}
		}
	}
#endif
}


} // namespace
#endif // FSTDATAACCESSORSPLITTING_H ///:~
