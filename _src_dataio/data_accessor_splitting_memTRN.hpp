#ifndef FSTDATAACCESSORSPLITTINGMEMTRN_H
#define FSTDATAACCESSORSPLITTINGMEMTRN_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_accessor_splitting_memTRN.hpp
   \brief   Implements data access to data cached entirely in memory, read once from TRN files
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    November 2010 (last updated July 2026)
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
#include <sstream>
#include <vector>

#include "error.hpp"
#include "global.hpp"
#include "data_accessor_splitting_mem.hpp"
#include "data_file_TRN.hpp"

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

//! Implements data access to data cached entirely in memory, read once from a TRN file
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_MemTRN : public Clonable_Impl<Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALCONTAINER>,Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT>
{
	// NOTE: the calling convention is the same for the inner and the outer CV loop -- both call just
	//   get*Split, get*Block, get*Pattern -- and which of them a call refers to is selected beforehand
	//   with the inherited setSplittingDepth()
	public:
		typedef Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER> DASM; //!< the immediate base class, aliased for brevity
		/*! parent_impl implements sharing_clone() via the weak copy-constructor; clone/stateless_clone throw in DASM.
		    \warning A sharing_clone() object shares 'data' with the original object. Although the sharing_clone
		             object can not modify 'data', there is no synchronization implemented to avoid
		             concurrent modification of 'data' by the original object while reading it from the sharing_clone! */
		typedef Clonable_Impl<Data_Accessor_Splitting_MemTRN,DASM,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT> parent_impl;
		typedef std::shared_ptr<Data_Scaler<DATATYPE> > PScaler; //!< shared pointer to the scaler applied to loaded data
		typedef typename DASM::PSplitters PSplitters; //!< inherited: shared pointer to the per-depth splitter stack
		//! Constructs the accessor for TRN file \p _filename, splitter stack \p _dsp and scaler \p _dsc; the file is opened and read by initialize().
		Data_Accessor_Splitting_MemTRN(const std::string _filename, const PSplitters _dsp, const PScaler _dsc) : parent_impl(_filename, _dsp, _dsc) {notify("Data_Accessor_Splitting_MemTRN constructor.");}
		virtual ~Data_Accessor_Splitting_MemTRN(){notify("Data_Accessor_Splitting_MemTRN destructor.");}

		virtual std::ostream& print(std::ostream& os) const override;
	protected:
		friend parent_impl; //!< grants Clonable_Impl access to the weak copy-constructor
		//! weak (referencing) copy-constructor to be used in sharing_clone(); the clone must keep a
		//! reference to #trnFile, because that object is what the file_get*() queries read their counters
		//! from -- a clone without it would dereference a null pointer
		Data_Accessor_Splitting_MemTRN(const Data_Accessor_Splitting_MemTRN &damt, int x) : parent_impl(damt, x), trnFile(damt.trnFile) {};
	protected:
		//! the TRN file reader; shared (read-only by convention) across sharing_clone()s so clones'
		//! file_get*() calls stay valid. The underlying file is closed at the end of initial_data_read(),
		//! but the object survives and keeps answering the dimensionality/class-size queries from memory.
		std::shared_ptr<Data_File_TRN<DATATYPE,IDXTYPE> > trnFile;

		virtual void initial_data_read() override ;    //!< reads the whole file into the base class's data array, then closes the file; \note off-limits in shared_clone
		virtual void initial_file_prepare() override; //!< opens trnFile on this->filename; \note off-limits in shared_clone

    public:
		virtual unsigned int file_getNoOfClasses() const override {assert(trnFile); return trnFile->getNoOfClasses(); } //!< forwards to the open TRN file
		virtual unsigned int file_getNoOfFeatures() const override {assert(trnFile); return trnFile->getNoOfFeatures(); } //!< forwards to the open TRN file
		virtual IDXTYPE file_getClassSize(unsigned int cls) const override {assert(trnFile); return trnFile->getClassSize(cls); } //!< forwards to the open TRN file
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_file_prepare()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemTRN()::initial_file_prepare() called from shared_clone instance.");
	trnFile.reset(new Data_File_TRN<DATATYPE,IDXTYPE>());
	trnFile->open(this->filename);
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_read()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemTRN()::initial_data_read() called from shared_clone instance.");
	{std::ostringstream sos; sos << "Reading data to memory..."; syncout::print(std::cout,sos);}
	IDXTYPE idx=0;
	for(unsigned int c=0;c<trnFile->getNoOfClasses();c++) 
	{
		for(IDXTYPE p=0;p<trnFile->getClassSize(c);p++) {
			for(unsigned int f=0;f<trnFile->getNoOfFeatures();f++) {
				this->data.get()[idx++]=trnFile->read_val();
			}
		}
	}
	trnFile->close();
	{std::ostringstream sos; sos << "done."<<std::endl; syncout::print(std::cout,sos);}
}



template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	DASM::print(os); 
	os << std::endl << "Data_Accessor_Splitting_MemTRN()";
	return os;
}
	
} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEMTRN_H ///:~
