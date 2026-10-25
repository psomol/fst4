#ifndef FSTDATAACCESSORSPLITTINGMEMCSV_H
#define FSTDATAACCESSORSPLITTINGMEMCSV_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    data_accessor_splitting_memCSV.hpp
   \brief   Implements data access to data cached entirely in memory, read once from CSV files
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0
   \note    FST4 was developed using MinGW-W64 g++ 15.2 and requires
   \note    \li a C++17 conforming compiler and standard library (no Boost, unlike FST3),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/,
                version 3.37; not distributed with FST4 — drop svm.cpp/svm.h into ./libsvm/, see libsvm/readme.txt)
   \note    Note that LibSVM is required for SVM related tools only.

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
#include "data_file_CSV.hpp"

namespace FST {

//! Implements data access to data cached entirely in memory, read once from a CSV (delimited text) file; see Data_File_CSV for the accepted dialects and options
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_MemCSV : public Clonable_Impl<Data_Accessor_Splitting_MemCSV<DATATYPE,IDXTYPE,INTERVALCONTAINER>,Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT>
{
	public:
		typedef Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER> DASM; //!< the immediate base class, aliased for brevity
		/*! parent_impl implements sharing_clone() via the weak copy-constructor; clone/stateless_clone throw in DASM.
		    \warning A sharing_clone() object shares 'data' with the original object. Although the sharing_clone
		             object can not modify 'data', there is no synchronization implemented to avoid
		             concurrent modification of 'data' by the original object while reading it from the sharing_clone! */
		typedef Clonable_Impl<Data_Accessor_Splitting_MemCSV,DASM,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT> parent_impl;
		typedef std::shared_ptr<Data_Scaler<DATATYPE> > PScaler; //!< shared pointer to the scaler applied to loaded data
		typedef typename DASM::PSplitters PSplitters; //!< inherited: shared pointer to the per-depth splitter stack
		//! Constructs the accessor for the CSV/TSV file \p _filename under dialect/class-column/missing-value options \p _opts, the splitter stack \p _dsp and the scaler \p _dsc; actual file reading happens in initialize() (inherited from Data_Accessor_Splitting_Mem).
		Data_Accessor_Splitting_MemCSV(const std::string _filename, const Data_File_CSV_Options &_opts, const PSplitters _dsp, const PScaler _dsc) : parent_impl(_filename, _dsp, _dsc), opts(_opts) {notify("Data_Accessor_Splitting_MemCSV constructor.");}
		virtual ~Data_Accessor_Splitting_MemCSV(){notify("Data_Accessor_Splitting_MemCSV destructor.");}

		virtual std::ostream& print(std::ostream& os) const override;
	protected:
		friend parent_impl; //!< grants Clonable_Impl access to the weak copy-constructor
		//! \brief Weak (referencing) copy-constructor used by sharing_clone(); transfers #opts and the cached header counters #clsCount/#featureCount/#clsSize
		//! -- the clone's file_get*() queries read them, so a clone without them would report garbage.
		//! #CSVFile itself stays null: initial_data_read() is off-limits in sharing clones.
		Data_Accessor_Splitting_MemCSV(const Data_Accessor_Splitting_MemCSV &damt, int x) : parent_impl(damt, x), opts(damt.opts), clsCount(damt.clsCount), featureCount(damt.featureCount), clsSize(damt.clsSize) {};
	protected:
		std::unique_ptr<Data_File_CSV> CSVFile;  //!< the CSV reader, opened by initial_file_prepare() and reset() once initial_data_read() has copied its data out; null in sharing clones
		Data_File_CSV_Options opts;  //!< dialect/class-column/missing-value options, as passed to the constructor

		unsigned int clsCount;      //!< number of classes, cached from CSVFile->getNoOfClasses() in initial_file_prepare()
		unsigned int featureCount;  //!< data dimensionality, cached from CSVFile->getNoOfFeatures() in initial_file_prepare()
		std::shared_ptr<IDXTYPE> clsSize;  //!< per-class sample counts (size #clsCount), cached from CSVFile->getClassSize() in initial_file_prepare()

		virtual void initial_data_read() override ;    //!< copies the reader's values into the base class's data array, then releases the reader; \note off-limits in shared_clone
		virtual void initial_file_prepare() override; //!< opens #CSVFile, reads and class-sorts it, and caches #clsCount/#featureCount/#clsSize; throws fst_error if called on a sharing clone

    public:
		virtual unsigned int file_getNoOfClasses() const override {return clsCount; } //!< #clsCount, as required by Data_Accessor_Splitting_Mem
		virtual unsigned int file_getNoOfFeatures() const override {return featureCount; } //!< #featureCount, as required by Data_Accessor_Splitting_Mem
		virtual IDXTYPE file_getClassSize(unsigned int cls) const override {return clsSize.get()[cls]; } //!< #clsSize[cls], as required by Data_Accessor_Splitting_Mem
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemCSV<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_file_prepare()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemCSV()::initial_file_prepare() called from shared_clone instance.");
	CSVFile.reset(new Data_File_CSV(this->filename, opts));
	CSVFile->prepareFile();
	CSVFile->sortRecordsByClass();

	clsCount = CSVFile->getNoOfClasses();
	featureCount = CSVFile->getNoOfFeatures();
	clsSize.reset(new IDXTYPE[clsCount], std::default_delete<IDXTYPE[]>());
	for (unsigned int i=0;i<clsCount; i++) clsSize.get()[i] = CSVFile->getClassSize(i);
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemCSV<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_read()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemCSV()::initial_data_read() called from shared_clone instance.");
	{std::ostringstream sos; sos << "Reading data to memory..."; syncout::print(std::cout,sos);}
	IDXTYPE idx=0;
	for(unsigned int p=0;p<CSVFile->getTotalSize();p++)
	{
		for(unsigned int f=0;f<CSVFile->getNoOfFeatures();f++) {
			this->data.get()[idx++]=(DATATYPE) CSVFile->getFeatureValue(p,f);
		}
	}
	CSVFile.reset();
	{std::ostringstream sos; sos << "done."<<std::endl; syncout::print(std::cout,sos);}
}



template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_MemCSV<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const
{
	DASM::print(os);
	os << std::endl << "Data_Accessor_Splitting_MemCSV()";
	return os;
}

} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEMCSV_H ///:~
