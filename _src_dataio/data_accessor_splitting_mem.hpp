#ifndef FSTDATAACCESSORSPLITTINGMEM_H
#define FSTDATAACCESSORSPLITTINGMEM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_accessor_splitting_mem.hpp
   \brief   Implements data access to data cached entirely in memory, concrete file type support is delegated to derived classes
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
#include "data_accessor_splitting.hpp"
#include "data_scaler.hpp"

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

/*! \brief Implements data access to data cached entirely in memory, concrete file type support is delegated to derived classes

    The whole data set is read once into one contiguous DATATYPE array. Classes are stored one after
    another (class c starts at #_class_offset[c]), each class holds its patterns consecutively, and
    each pattern holds its getNoOfFeatures() values consecutively. A block handed out by
    getFirstBlock()/getNextBlock() is therefore just a pointer into that array plus a pattern count;
    the array stays alive for the accessor's lifetime and is never handed over to the caller.

    A derived class supplies the file format: it implements initial_file_prepare() (open and read the
    header), initial_data_read() (fill the array) and the three file_get*() queries that report
    dimensionality and class sizes. initialize() drives all of them, in that order, and must be called
    exactly once before any data access. */
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_Mem : public Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>
{
	// NOTE: the calling convention is the same for the inner and the outer CV loop -- both call just
	//   get*Split, get*Block, get*Pattern -- and which of them a call refers to is selected beforehand
	//   with the inherited setSplittingDepth()
	public:
		typedef Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER> DAS; //!< the immediate base class, aliased for brevity in derived classes
		typedef std::shared_ptr<Data_Scaler<DATATYPE> > PScaler; //!< shared pointer to the scaler applied to loaded data
		typedef typename DAS::PPattern PPattern; //!< inherited from Data_Accessor: raw pointer to one read-only feature vector
		typedef typename DAS::PSplitters PSplitters; //!< inherited: shared pointer to the per-depth splitter stack
		//! Constructs the accessor for \p _filename (opened by a derived class's initial_file_prepare()),
		//! the splitter stack \p _dsp and the scaler \p _dsc; actual file reading happens in initialize().
		Data_Accessor_Splitting_Mem(const std::string _filename, const PSplitters _dsp, const PScaler _dsc) : Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>(_dsp), dsc(_dsc), filename(_filename), datasize(0), scaler_fit_train(false) {notify("Data_Accessor_Splitting_Mem constructor.");}
		virtual ~Data_Accessor_Splitting_Mem(){notify("Data_Accessor_Splitting_Mem destructor.");}

		/*! \brief to be called after object creation (for initialization of data file access etc.)

		    Opens and reads the file through the derived class's hooks, allocates the in-memory data
		    array and sets up the splitting structures. Call it exactly once, on the owning instance,
		    before any other access; it throws fst_error (or an I/O or allocation exception) on failure.
		    It also reports the file name, dimensionality, class sizes and splitting depth on standard
		    output through syncout::print() -- there is no quiet mode, so a caller that needs clean
		    standard output must redirect it around this call.
		    \note off-limits in shared_clone */
		virtual void initialize();

		/*! \brief Switches on per-split scaling: the scaler learns its transform on the current depth-0
		    split's TRAIN part only and applies it to ALL data, refreshed on every depth-0 split change.

		    This avoids the normalization leakage inherent in scaling at load time, where the test part
		    contributes to the min/max or mean/stddev the transform is built from and so biases every
		    later estimate of generalization performance. Must be called BEFORE initialize().
		    The default, false, scales once over the whole data set at load time.
		    \note each instance (incl. every sharing_clone) then keeps a private working buffer and
		          a private scaler instance, so threaded searches stay race-free and produce the same
		          result as a sequential run */
		void set_scaler_fit_train(const bool b) {scaler_fit_train=b;}
		bool get_scaler_fit_train() const {return scaler_fit_train;} //!< see set_scaler_fit_train()

		/*! \brief Applies the learned per-feature transform to \p sample, a caller-owned vector of
		    getNoOfFeatures() values, in place.

		    Value i receives feature i's transform -- min/max (to01), mean/stddev (white) or the
		    missing-value substitution -- exactly as the loaded data went through it, so a vector that
		    is to be classified by a model trained on this accessor's data must pass through here first.
		    In per-split mode (set_scaler_fit_train()) the transform is the one fitted on the CURRENT
		    depth-0 split, i.e. the one the working buffer carries at the moment of the call.
		    Throws fst_error before initialize(), or when the scaler does not implement duplicate(). */
		void scale_sample_inplace(DATATYPE *sample) const;

		virtual bool getFirstSplit() override;
		virtual bool getNextSplit() override;

		virtual bool getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0) override;
		virtual bool getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0) override;

		Data_Accessor_Splitting_Mem* clone() const override {throw fst_error("Data_Accessor_Splitting_Mem::clone() not supported, use Data_Accessor_Splitting_Mem::sharing_clone() instead.");}
		Data_Accessor_Splitting_Mem* sharing_clone() const override = 0;
		Data_Accessor_Splitting_Mem* stateless_clone() const override {throw fst_error("Data_Accessor_Splitting_Mem::stateless_clone() not supported, use Data_Accessor_Splitting_Mem::sharing_clone() instead.");}
		
		virtual std::ostream& print(std::ostream& os) const override;
	protected:
		//! weak (referencing) copy-constructor to be used in sharing_clone()
		Data_Accessor_Splitting_Mem(const Data_Accessor_Splitting_Mem &damt, int);
	protected:
		// Contract for a derived (file-format) class. initialize() calls these in this order:
		// initial_file_prepare(), then the three file_get*() queries, then initial_data_read().
		// A hook that cannot do its job must throw fst_error; returning silently would leave the
		// accessor half-built. None of them may be reached on a sharing_clone.
		//! \brief fills the already-allocated #data array with the file's samples; called once by initialize() after the array is sized
		//! Must write exactly #datasize values, laid out class by class, pattern by pattern, feature by
		//! feature (class c starts at #_class_offset[c]), matching the counts the file_get*() queries report.
		//! \note off-limits in shared_clone
		virtual void initial_data_read() = 0;
		//! \brief opens/prepares the underlying file for reading; called once by initialize() before initial_data_read()
		//! After it returns, the three file_get*() queries below must answer correctly.
		virtual void initial_file_prepare() = 0;
		virtual unsigned int file_getNoOfClasses() const = 0; //!< number of classes as reported by the underlying file (valid only after initial_file_prepare())
		virtual unsigned int file_getNoOfFeatures() const = 0; //!< data dimensionality as reported by the underlying file (valid only after initial_file_prepare())
		virtual IDXTYPE file_getClassSize(unsigned int cls) const = 0; //!< number of samples of class \p cls as reported by the underlying file (valid only after initial_file_prepare())

		void initial_data_scaling(); //!< \note off-limits in shared_clone

		//! re-learns the scaler transform on the current depth-0 TRAIN part and re-scales the whole working buffer from the pristine data (scaler_fit_train mode only)
		void per_split_data_scaling();

		PScaler dsc; //!< \note in scaler_fit_train mode every instance holds a private duplicate(); otherwise off-limits in shared_clone
		//! the learned per-feature transforms: one duplicate() of #dsc per feature, taken right after that
		//! feature was learned, for scale_sample_inplace() -- the single #dsc object is re-learned feature
		//! after feature and so ends up holding only the LAST feature's statistics. Filled by
		//! initial_data_scaling() (load-time fit), or by per_split_data_scaling() in scaler_fit_train mode
		//! (then the current depth-0 split's transform); empty until then, or when the scaler does not
		//! implement duplicate(). The duplicates are never mutated, so sharing clones may share them.
		std::vector<PScaler> feature_scalers;

		std::string filename; //!< \note off-limits in shared_clone

		IDXTYPE datasize; //!< total number of DATATYPE values held in data/data_raw (sum over classes of class size * features)
		std::shared_ptr<DATATYPE> data; //!< \note read-only in shared_clone — EXCEPT in scaler_fit_train mode, where each instance owns a private working buffer
		//! pristine (unscaled) data as loaded; allocated in scaler_fit_train mode only; shared across clones, never mutated after load
		std::shared_ptr<DATATYPE> data_raw;
		//! per-split scaling mode switch (see set_scaler_fit_train())
		bool scaler_fit_train;
		//! class start indexes (offsets) in whatever random-access data representation
		std::vector<IDXTYPE> _class_offset; //!< \note read-only in shared_clone
};

// learns the scaler transform over the whole data set, per feature, and applies it in place;
// off-limits in shared_clone (it mutates the shared data buffer)
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_scaling()
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_Mem()::initial_data_scaling() called from shared_clone instance.");
	// SCALING over values per-feature, regardless classes
	assert(dsc);
	{std::ostringstream sos; sos << std::endl << "Scaling data (per-feature, over all classes) using " << *dsc << "..."; syncout::print(std::cout,sos);}
	IDXTYPE idx=0;
	IDXTYPE _features=file_getNoOfFeatures();
	feature_scalers.clear(); feature_scalers.reserve(_features);
	bool keep_transforms=true;
	for(IDXTYPE f=0;f<_features;f++) {
		for(bool b=dsc->startFirstLoop(); b==true; b=dsc->startNextLoop())
		{
			idx=f;
			for(unsigned int c=0;c<file_getNoOfClasses();c++) for(IDXTYPE p=0;p<file_getClassSize(c);p++)
			{
				dsc->learn(data.get()[idx]);
				idx+=_features;
			}
		}
		idx=f;
		for(unsigned int c=0;c<file_getNoOfClasses();c++) for(IDXTYPE p=0;p<file_getClassSize(c);p++)
		{
			dsc->scale_inplace(data.get()[idx]);
			idx+=_features;
		}
		if(keep_transforms) { // keep this feature's learned transform for scale_sample_inplace()
			try {feature_scalers.push_back(PScaler(dsc->duplicate()));}
			catch(fst_error&) {keep_transforms=false; feature_scalers.clear();} // a scaler without duplicate(): user vectors cannot be transformed, the data itself is unaffected
		}
	}
	{std::ostringstream sos; sos << "done." << std::endl << std::endl; syncout::print(std::cout,sos);}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::per_split_data_scaling()
{
	// scaler_fit_train mode: learn the transform on the current depth-0 TRAIN part only,
	// then re-derive the whole working buffer from the pristine data (leakage-free scaling;
	// deliberately silent — this runs on every depth-0 split change).
	// NOTE: fitting reads the physical splits[0][*]._train intervals (absolute at depth 0),
	//       so DataPart substitution (substitute()) does not affect what the scaler learns on
	assert(scaler_fit_train);
	assert(dsc);
	assert(data && data_raw);
	const unsigned int _features=DAS::features;
	const unsigned int _classes=(unsigned int)DAS::classes.size();
	feature_scalers.clear(); feature_scalers.reserve(_features);
	bool keep_transforms=true;
	for(unsigned int f=0;f<_features;f++)
	{
		for(bool b=dsc->startFirstLoop(); b==true; b=dsc->startNextLoop())
		{
			for(unsigned int c=0;c<_classes;c++)
			{
				const INTERVALCONTAINER &train=*(DAS::splits[0][c]._train);
				for(typename INTERVALCONTAINER::const_iterator iter=train.begin();iter!=train.end();++iter)
				{
					IDXTYPE idx=_class_offset[c]+((IDXTYPE)iter->startidx)*_features+f;
					for(IDXTYPE p=0;p<(IDXTYPE)iter->count;p++,idx+=_features) dsc->learn(data_raw.get()[idx]);
				}
			}
		}
		// apply to ALL patterns (train and test parts alike), always from the pristine values
		IDXTYPE idx=f;
		for(unsigned int c=0;c<_classes;c++) for(IDXTYPE p=0;p<(IDXTYPE)DAS::classes[c];p++,idx+=_features)
			data.get()[idx]=dsc->scale(data_raw.get()[idx]);
		if(keep_transforms) { // this split's transform of feature f, for scale_sample_inplace()
			try {feature_scalers.push_back(PScaler(dsc->duplicate()));}
			catch(fst_error&) {keep_transforms=false; feature_scalers.clear();}
		}
	}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::scale_sample_inplace(DATATYPE *sample) const
{
	assert(sample);
	if(feature_scalers.size()!=(std::size_t)DAS::features)
		throw fst_error("Data_Accessor_Splitting_Mem::scale_sample_inplace(): no per-feature scaler transform available (initialize() not called yet, or the scaler does not implement duplicate())");
	for(std::size_t f=0;f<feature_scalers.size();f++) feature_scalers[f]->scale_inplace(sample[f]);
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstSplit()
{
	const bool result=DAS::getFirstSplit();
	if(result && scaler_fit_train && DAS::getSplittingDepth()==0) per_split_data_scaling();
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextSplit()
{
	const bool result=DAS::getNextSplit();
	if(result && scaler_fit_train && DAS::getSplittingDepth()==0) per_split_data_scaling();
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initialize()
{
	// NOTE: throws I/O and memory related exceptions; off-limits in shared_clone
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_Mem()::initialize() called from shared_clone instance.");
	
	assert(DAS::dsp->size()>0);
	try {
		{std::ostringstream sos; sos << "Opening file: " << filename << std::endl; syncout::print(std::cout,sos);}
		initial_file_prepare();
		{
			std::ostringstream sos;
			sos << "Features: " << file_getNoOfFeatures() << std::endl << "Classes: " << file_getNoOfClasses() << std::endl;
			syncout::print(std::cout,sos);
		}
			
		typename DAS::CLASSSIZES csizes;
		for(unsigned int c=0;c<file_getNoOfClasses();c++) 
		{
			{std::ostringstream sos; sos << "  Class "<<c<<" size: "<<file_getClassSize(c) << std::endl; syncout::print(std::cout,sos);} // 0-based, like every class index FST4 prints or accepts (--f1-positive, --class-weights, confusion rows) and like the feature indices
			csizes.push_back(file_getClassSize(c));
		}
		{
			std::ostringstream sos;
			sos << "Splitting depth: " << DAS::dsp->size() << std::endl;
			syncout::print(std::cout,sos);
		}
		
		DAS::initialize(file_getNoOfFeatures(), csizes); // set-up memory structures to support data splitting
		
		datasize=0;
		_class_offset.clear();
		for(unsigned int c=0;c<file_getNoOfClasses();c++) {
			_class_offset.push_back(datasize); // to be set only once
			datasize+=file_getClassSize(c)*file_getNoOfFeatures();
		}

		data.reset(new DATATYPE[datasize], std::default_delete<DATATYPE[]>());
		initial_data_read();
		if(!scaler_fit_train) initial_data_scaling();
		else {
			// per-split scaling: keep the loaded data pristine (shared, read-only), work on a private
			// copy; the transform is learned per depth-0 split in per_split_data_scaling() — before the
			// first split is made no transform is defined, so pre-split access sees unscaled data
			assert(dsc);
			data_raw=data;
			data.reset(new DATATYPE[datasize], std::default_delete<DATATYPE[]>());
			std::copy(data_raw.get(),data_raw.get()+datasize,data.get());
			{std::ostringstream sos; sos << std::endl << "Per-split scaling enabled (fit on each depth-0 TRAIN part) using " << *dsc << "." << std::endl; syncout::print(std::cout,sos);}
		}

	} catch (...) {
		notify("Data_Accessor_Splitting_Mem::initialize() failed.");
		throw;
	}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth)
{
	assert(data);
	typename DAS::DATAINTERVAL tmp;
	if(!DAS::getFirstBlock(ofwhat,tmp,loopdepth)) return false;
	firstpattern=&data.get()[_class_offset[DAS::active_class]+tmp->startidx*DAS::features];
	patterns=tmp->count;
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth)
{
	assert(data);
	typename DAS::DATAINTERVAL tmp;
	if(!DAS::getNextBlock(ofwhat,tmp,loopdepth)) return false;
	firstpattern=&data.get()[_class_offset[DAS::active_class]+tmp->startidx*DAS::features];
	patterns=tmp->count;
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting_Mem(const Data_Accessor_Splitting_Mem& damt, int) : Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>(damt),
	feature_scalers(damt.feature_scalers),
	datasize(damt.datasize),
	data(damt.data),
	data_raw(damt.data_raw),
	scaler_fit_train(damt.scaler_fit_train),
	_class_offset(damt._class_offset)
{
	if(scaler_fit_train) {
		// per-split scaling: the clone re-learns the transform whenever it advances depth-0 splits,
		// so it needs a private working buffer (initialized to the parent's current state — correct
		// for any access that does not advance splits) and a private scaler instance
		if(!damt.dsc) throw fst_error("Data_Accessor_Splitting_Mem weak cc: scaler_fit_train mode requires a scaler instance in the source.");
		data.reset(new DATATYPE[datasize], std::default_delete<DATATYPE[]>());
		std::copy(damt.data.get(),damt.data.get()+datasize,data.get());
		dsc.reset(damt.dsc->duplicate());
	} else dsc.reset();
	filename.clear();
	notify("Data_Accessor_Splitting_Mem weak copy-constructor.");
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	DAS::print(os); 
	os << std::endl << "Data_Accessor_Splitting_Mem(file=" << filename << ")";
	return os;
}
	
} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEM_H ///:~
