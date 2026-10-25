#ifndef FSTDATAACCESSORSPLITTINGMEMARFF_H
#define FSTDATAACCESSORSPLITTINGMEMARFF_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_accessor_splitting_memARFF.hpp
   \brief   Implements data access to data cached entirely in memory, read once from ARFF files
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
#include <iostream>
#include <sstream>
#include <vector>

#include "error.hpp"
#include "global.hpp"
#include "data_accessor_splitting_mem.hpp"
#include "data_file_ARFF.hpp"

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

//! Implements data access to data cached entirely in memory, read once from a ARFF file
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_MemARFF : public Clonable_Impl<Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>,Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT>
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
		typedef Clonable_Impl<Data_Accessor_Splitting_MemARFF,DASM,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT> parent_impl;
		typedef std::shared_ptr<Data_Scaler<DATATYPE> > PScaler; //!< shared pointer to the scaler applied to loaded data
		typedef typename DASM::PSplitters PSplitters; //!< inherited: shared pointer to the per-depth splitter stack
		//! Constructs the accessor for the ARFF file \p _filename, the splitter stack \p _dsp and the scaler \p _dsc; actual file reading happens in initialize() (inherited from Data_Accessor_Splitting_Mem).
		Data_Accessor_Splitting_MemARFF(const std::string _filename, const PSplitters _dsp, const PScaler _dsc) : parent_impl(_filename, _dsp, _dsc) {notify("Data_Accessor_Splitting_MemARFF constructor.");}
		virtual ~Data_Accessor_Splitting_MemARFF(){notify("Data_Accessor_Splitting_MemARFF destructor.");}

		virtual std::ostream& print(std::ostream& os) const override;
	protected:
		friend parent_impl; //!< grants Clonable_Impl access to the weak copy-constructor
		//! \brief Weak (referencing) copy-constructor used by sharing_clone(); transfers the cached header counters #clsCount/#featureCount/#clsSize
		//! -- the clone's file_get*() queries read them, so a clone without them would report garbage.
		//! #ARFFFile itself stays null: initial_data_read() is off-limits in sharing clones.
		Data_Accessor_Splitting_MemARFF(const Data_Accessor_Splitting_MemARFF &damt, int x) : parent_impl(damt, x), clsCount(damt.clsCount), featureCount(damt.featureCount), clsSize(damt.clsSize) {};
	protected:
		std::unique_ptr<Data_File_ARFF> ARFFFile;  //!< the ARFF reader, opened by initial_file_prepare() and reset() once initial_data_read() has copied its data out; null in sharing clones

		unsigned int clsCount;      //!< number of classes, cached from ARFFFile->getNoOfClasses() in initial_file_prepare()
		unsigned int featureCount;  //!< data dimensionality, cached from ARFFFile->getNoOfFeatures() in initial_file_prepare()
		std::shared_ptr<IDXTYPE> clsSize;  //!< per-class sample counts (size #clsCount), cached from ARFFFile->getClassSize() in initial_file_prepare()

		virtual void initial_data_read() override ;    //!< copies the reader's values into the base class's data array, then releases the reader; \note off-limits in shared_clone
		virtual void initial_file_prepare() override; //!< opens #ARFFFile, reads and class-sorts it, and caches #clsCount/#featureCount/#clsSize; throws fst_error if called on a sharing clone

    public:
		virtual unsigned int file_getNoOfClasses() const override {return clsCount; } //!< #clsCount, as required by Data_Accessor_Splitting_Mem
		virtual unsigned int file_getNoOfFeatures() const override {return featureCount; } //!< #featureCount, as required by Data_Accessor_Splitting_Mem
		virtual IDXTYPE file_getClassSize(unsigned int cls) const override {return clsSize.get()[cls]; } //!< #clsSize[cls], as required by Data_Accessor_Splitting_Mem
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_file_prepare()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemARFF()::initial_file_prepare() called from shared_clone instance.");
	ARFFFile.reset(new Data_File_ARFF(this->filename));
	if(!ARFFFile->prepareFile()) // the reader has already printed the offending line/reason
		throw fst_error("ARFF file '"+this->filename+"' could not be loaded (see the reader's message above for the offending definition or data row)");
	ARFFFile->sortRecordsByClass();
	
	clsCount = ARFFFile->getNoOfClasses();
	featureCount = ARFFFile->getNoOfFeatures();
	clsSize.reset(new IDXTYPE[clsCount], std::default_delete<IDXTYPE[]>());
	for (unsigned int i=0;i<clsCount; i++) clsSize.get()[i] = ARFFFile->getClassSize(i);
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_read()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemARFF()::initial_data_read() called from shared_clone instance.");
	{std::ostringstream sos; sos << "Reading data to memory..."; syncout::print(std::cout,sos);}
	IDXTYPE idx=0;
	for(unsigned int p=0;p<ARFFFile->getTotalSize();p++) 
	{
		for(unsigned int f=0;f<ARFFFile->getNoOfFeatures();f++) {
			this->data.get()[idx++]=(DATATYPE) ARFFFile->getFeatureValue(p,f);
		}
	}
	ARFFFile.reset();
	{std::ostringstream sos; sos << "done."<<std::endl; syncout::print(std::cout,sos);}
}



template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	DASM::print(os); 
	os << std::endl << "Data_Accessor_Splitting_MemARFF()";
	return os;
}
	
} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEMARFF_H ///:~
