#ifndef FSTDATAACCESSOR_H
#define FSTDATAACCESSOR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_accessor.hpp
   \brief   Defines interface for data access implementations
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

#include <vector>
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

//! Selects which part of a split's data a block-access call reads.
enum DataPart {
  NONE=0,      //!< no part (used internally, e.g. for traintestphase state before a phase is chosen)
  TRAIN=1,     //!< the split's training part
  TEST=2,      //!< the split's test part
  TRAINTEST=4, //!< both parts concatenated (train followed by test)
  ALL=8        //!< every pattern regardless of split (the whole class)
};

/*! \brief abstract class, defines interface for data access implementations

    A Data_Accessor presents the data set as a set of classes, each split into a training and a
    test part, and hands the data out in consecutive blocks of patterns.

    Usage protocol. Reading always follows the same three nested levels, in this order:
    \li select the class with setClass(c), c in 0..getNoOfClasses()-1;
    \li iterate over splits with getFirstSplit() / getNextSplit() (one iteration for a plain
        hold-out, k for k-fold cross-validation, and so on) -- both return false once the splits
        are exhausted;
    \li for the current class and split, read the data in blocks: either
        `for(const auto &b : da->blocks(FST::TRAIN)) {...}`, or the deprecated
        getFirstBlock()/getNextBlock() cursor pair.

    The split iteration is global to the accessor, not per class: setClass() must be re-issued
    for each class INSIDE the split loop, and a split loop must not be restarted while blocks of
    that split are still being read.

    Memory layout. A block is a pointer to the first pattern plus a pattern count. Patterns are
    stored consecutively, and within a pattern the getNoOfFeatures() feature values are
    consecutive too. So for a block (p, n): pattern i starts at p + i*getNoOfFeatures(), feature
    j of that pattern is p[i*getNoOfFeatures()+j], and walking one feature across the block means
    striding by getNoOfFeatures(). The data is valid only until the next split change or the next
    block call on the same loop slot, and the accessor keeps ownership of it -- never free it.

    Thread safety. A single instance must not be driven from several threads; each thread takes
    its own sharing_clone() (see Clonable), which shares the read-only sample storage but keeps a
    private iteration state. */
template<typename DATATYPE, typename IDXTYPE>
class Data_Accessor : public Clonable // abstract class
{
	public:
		typedef const DATATYPE*	PPattern; //!< one data sample: a raw pointer to its (read-only) feature vector, getNoOfFeatures() entries long
		Data_Accessor() {notify("Data_Accessor constructor");}
		virtual ~Data_Accessor() {notify("Data_Accessor destructor");}	
		
		virtual unsigned int getNoOfClasses() const =0;              //!< returns number of classes
		virtual unsigned int getNoOfFeatures() const =0;             //!< returns data dimensionality
		virtual IDXTYPE getClassSize(const unsigned int c) const =0; //!< returns size (number of samples in) of class c
		virtual IDXTYPE getClassSizeSum() const =0;                  //!< returns summed size (number of samples in) of all classes, i.e., no. of all patterns in data

		virtual void setClass(const int c) =0;   //!< sets active class from which get*Block() will fetch data
		virtual int  getClass() const =0;        //!< returns active class

		virtual unsigned int getNoOfSplits() const =0;  //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getFirstSplit()=0;                 //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getNextSplit()=0;                  //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual unsigned int getSplitIndex() const =0;  //!< data access iteration (to support, e.g., loops in cross-validation)

		//! \brief returns the number of consecutive blocks the requested DataPart is made of in the current split
		//! \warning getNoOfBlocks is not to be relied upon due to possible limitations in some implementations
		virtual IDXTYPE getNoOfBlocks(const DataPart ofwhat) const =0;
		//! \brief returns pointer to first consecutive block of data of requested DataPart type in the current split (access iteration)
		//! \param ofwhat which part of the split to read
		//! \param firstpattern receives a pointer to the block's first pattern (owned by the accessor)
		//! \param patterns receives the number of patterns in the block
		//! \param loopdepth the cursor slot to use; two block loops that overlap in time MUST use different
		//! slots or their state collides and the behaviour is undefined. Slots 0..3 are the ones available here.
		//! \return false if the requested part is empty in the current split
		//! \deprecated Deprecated since FST4 4.0; prefer the RAII blocks() range, which acquires a private
		//! cursor slot automatically instead of leaving the choice to the caller. Removal planned for the
		//! next major release.
		virtual bool getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0)=0;
		//! \brief returns pointer to next consecutive block of data of requested DataPart type in the current split (access iteration)
		//! Must be preceded by getFirstBlock() on the same \p loopdepth; returns false once the blocks are exhausted.
		//! \deprecated Deprecated since FST4 4.0; prefer the RAII blocks() range. Removal planned for the
		//! next major release.
		virtual bool getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0)=0;

		//! one consecutive block of patterns as yielded by Block_Range/blocks()
		struct Block {
			PPattern data;  //!< pointer to the first pattern of the block
			IDXTYPE size;   //!< number of patterns in the block
		};
		/*! \brief RAII range over consecutive data blocks of one DataPart in the current split

		    Owns a private loopdepth slot acquired from the accessor for the range's lifetime,
		    so overlapping block loops cannot collide the way manually chosen loopdepth
		    constants can. Depths 0..3 stay reserved for the deprecated getFirstBlock()/getNextBlock()
		    cursor API; acquired slots start above them.
		    Usage: for(const auto &b : da->blocks(FST::TRAIN)) {... b.data, b.size ...} */
		class Block_Range {
		public:
			//! Forward iterator over Block_Range; drives the wrapped getFirstBlock()/getNextBlock() cursor at the range's owned loopdepth.
			class iterator {
			public:
				//! Constructs an end iterator when \p done is true; otherwise fetches the first block via \p r, becoming an end iterator itself if none exists.
				iterator(Block_Range *r, const bool done) : _r(r), _done(done) {_cur.data=nullptr; _cur.size=0; if(!_done) _done=!_r->_da->getFirstBlock(_r->_part,_cur.data,_cur.size,_r->_depth);}
				const Block& operator*() const {return _cur;} //!< the current block (data pointer + size)
				iterator& operator++() {_done=!_r->_da->getNextBlock(_r->_part,_cur.data,_cur.size,_r->_depth); return *this;} //!< advances to the next block; becomes an end iterator once the loop is exhausted
				bool operator!=(const iterator& o) const {return _done!=o._done;} //!< range-for only ever compares against end(); true until exhaustion
			private:
				Block_Range *_r; //!< the owning range
				bool _done;       //!< whether iteration is exhausted
				Block _cur;       //!< the current block
			};
			//! Acquires a private loopdepth slot from \p da for the range's lifetime; \p part selects which DataPart the range iterates.
			Block_Range(Data_Accessor *da, const DataPart part) : _da(da), _part(part), _depth(da->acquire_block_loop_slot()) {}
			~Block_Range() {_da->release_block_loop_slot(_depth);}
			iterator begin() {return iterator(this,false);} //!< an iterator positioned at the first block, or already at end() if there are none
			iterator end() {return iterator(this,true);} //!< the sentinel end iterator
			unsigned int loopdepth() const {return _depth;} //!< the owned slot (e.g. for getBlockIndex())
			// neither copyable nor movable: exactly one owner of the loopdepth slot. blocks()
			// can still return a range by value because C++17 guaranteed copy elision
			// materializes the return value in place, with no copy or move involved.
			Block_Range(const Block_Range&) = delete;
			Block_Range& operator=(const Block_Range&) = delete;
		private:
			Data_Accessor *_da;       //!< the accessor iterated
			const DataPart _part;      //!< the DataPart selected
			const unsigned int _depth; //!< this range's owned loopdepth slot
		};
		//! RAII block range over the current split - prefer over the deprecated getFirstBlock()/getNextBlock() cursor
		Block_Range blocks(const DataPart ofwhat) {return Block_Range(this,ofwhat);}
		//! returns index of the current consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getBlockIndex(const unsigned int loopdepth=0) const =0;
		//! returns number of patterns in all consecutive blocks of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getNoOfPatterns(const DataPart ofwhat) const =0;
		//! \brief enables change of meaning of DataPart types, for use in specialized data access scenarios like in bias predicting wrappers
		//! Installs a remapping so that every later request for \p source is served with \p target's data instead.
		//! The mapping is a property of the accessor, applies to all classes and splits, and stays in force until
		//! resubstitute() -- so a component that substitutes must restore the mapping before handing the accessor on.
		//! Mappings do not chain: substituting TRAIN->TEST and then TEST->ALL makes TRAIN yield TEST's data, not ALL's.
		//! \param source the DataPart the caller will keep asking for
		//! \param target the DataPart actually delivered
		virtual void substitute(const DataPart source, const DataPart target)=0;
		//! resets standard DataPart types' meaning, i.e. undoes every substitute() call
		virtual void resubstitute()=0;

		Data_Accessor* clone() const override =0;
		Data_Accessor* sharing_clone() const override =0;
		Data_Accessor* stateless_clone() const override =0;
	private:
		//! loopdepth slot pool for Block_Range; depths 0.._first_acquired_loopdepth-1 stay reserved for the
		//! deprecated hard-coded cursor loops. The pool is plain, unsynchronized bookkeeping: it serves
		//! overlapping ranges within ONE accessor instance, not concurrent threads, which use their own
		//! sharing_clone() and therefore their own pool.
		static const unsigned int _first_acquired_loopdepth = 4;
		std::vector<bool> _block_slot_used; //!< one flag per acquired slot (index 0 = loopdepth _first_acquired_loopdepth); grows as concurrent Block_Range instances need more slots
		//! Reserves and returns a free loopdepth (reusing a released one if available, else growing the pool by one).
		unsigned int acquire_block_loop_slot() {
			for(unsigned int i=0;i<(unsigned int)_block_slot_used.size();++i) if(!_block_slot_used[i]) {_block_slot_used[i]=true; return _first_acquired_loopdepth+i;}
			_block_slot_used.push_back(true);
			return _first_acquired_loopdepth+(unsigned int)(_block_slot_used.size()-1);
		}
		//! Returns the loopdepth slot acquired by acquire_block_loop_slot() so a future Block_Range can reuse it.
		void release_block_loop_slot(const unsigned int depth) {
			assert(depth>=_first_acquired_loopdepth);
			const unsigned int i=depth-_first_acquired_loopdepth;
			if(i<(unsigned int)_block_slot_used.size()) _block_slot_used[i]=false;
		}
};


} // namespace
#endif // FSTDATAACCESSOR_H ///:~
