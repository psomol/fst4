#ifndef FSTDATAACCESSORUNIVERSE_H
#define FSTDATAACCESSORUNIVERSE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    data_accessor_universe.hpp
   \brief   Data-free accessor over an abstract feature universe (`--features` N; pure set-function optimization)
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

   \warning Please note that this code plays only supportive role
            and does not represent coding standards followed in core
            FST4 codes.
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

#include <sstream>
#include "error.hpp"
#include "global.hpp"
#include "data_accessor_splitting_mem.hpp"

namespace FST {

/*! \brief Data-free accessor over an abstract feature universe of N features
    (the fst4 CLI's `--features` N entry; the companion of criterion_extern_value.hpp
    in its data-free mode)

    There is no data file, no samples and no classes — the accessor's ONLY
    substance is the universe size N, which defines the subset dimensionality
    the searches optimize over. Everything else is the minimal honest
    degenerate case: 0 classes, 0 samples, and a trivial split protocol that
    yields exactly ONE (empty) split so the pipeline's split-iteration
    contract holds. Intended for `--criterion` extern-value `--extern-no-data`,
    where the external tool answers index-only VALUE requests from its own
    logic and never reads data; every data-touching component (classifiers,
    scalers, real splitters, validation) is refused upstream by the CLI's
    validity rules — this class deliberately implements NO way to read
    patterns, so a component that slipped through fails loudly rather than
    computing on fabricated data.

    Derives from Data_Accessor_Splitting_Mem so the CLI's concrete-side
    plumbing (initialize()/setSplittingDepth()) is unchanged, but overrides
    initialize() and the split protocol wholesale: the base machinery is
    built around per-class sample intervals, and riding it with zero classes
    would trip its (correct) debug assertions. The constructor still takes
    the splitter/scaler arguments for structural compatibility — they are
    never exercised. */
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Universe : public Clonable_Impl<Data_Accessor_Universe<DATATYPE,IDXTYPE,INTERVALCONTAINER>,Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT>
{
	public:
		typedef Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER> DASM; //!< the immediate base class, aliased for brevity
		typedef typename DASM::DAS DAS; //!< inherited: the Data_Accessor_Splitting base, used for its CLASSSIZES/enum_split/getSplittingDepth() machinery
		/*! parent_impl implements sharing_clone() via the weak copy-constructor; clone/stateless_clone throw in DASM. */
		typedef Clonable_Impl<Data_Accessor_Universe,DASM,CLONE_INHERIT,CLONE_WEAK_SHARING,CLONE_INHERIT> parent_impl;
		typedef std::shared_ptr<Data_Scaler<DATATYPE> > PScaler; //!< accepted for constructor-signature compatibility only; this accessor is data-free and never uses a scaler
		typedef typename DASM::PSplitters PSplitters; //!< accepted for constructor-signature compatibility only; this accessor is data-free and never uses a splitter
		//! Constructs the accessor over an abstract universe of \p n_features features; \p _dsp and \p _dsc are accepted for structural compatibility with the other Data_Accessor_Splitting_Mem descendants but are never exercised.
		Data_Accessor_Universe(const unsigned int n_features, const PSplitters _dsp, const PScaler _dsc) : parent_impl("",_dsp,_dsc), _n(n_features) {notify("Data_Accessor_Universe constructor.");}
		virtual ~Data_Accessor_Universe(){notify("Data_Accessor_Universe destructor.");}

		//! \brief no file to read: announces the universe and sets up the degenerate shape (N features, 0 classes)
		//! Writes a one-line summary of the universe on standard output through syncout::print(), and throws
		//! fst_error if the universe is empty or if it is called on a sharing clone.
		virtual void initialize() override
		{
			if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Universe::initialize() called from shared_clone instance.");
			if(_n==0) throw fst_error("Data_Accessor_Universe: the feature universe must have at least 1 feature.");
			{std::ostringstream sos; sos << "Feature universe: " << _n << " features, no data (--features)" << std::endl; syncout::print(std::cout,sos);}
			DAS::initialize(_n, typename DAS::CLASSSIZES()); // features=N, zero classes; no buffer, no scaling, no splitter set_n
		}

		// The three overrides below implement a trivial split protocol: exactly one (empty) split per
		// enumeration round. There is nothing to split, but callers iterate splits unconditionally. The
		// base machinery is bypassed on purpose: it would run the configured splitter over zero classes,
		// tripping its (correct) debug assertions in Debug builds.
		virtual bool getFirstSplit() override {DAS::enum_split[DAS::getSplittingDepth()]=1; return true;} //!< enters the single split; always true
		virtual bool getNextSplit() override {DAS::enum_split[DAS::getSplittingDepth()]=0; return false;} //!< always false: the single split is the only one
		virtual unsigned int getNoOfSplits() const override {return 1;} //!< always 1

		virtual std::ostream& print(std::ostream& os) const override {os << "Data_Accessor_Universe(features=" << _n << ", data-free)"; return os;}
	protected:
		friend parent_impl; //!< grants Clonable_Impl access to the weak copy-constructor
		//! weak (referencing) copy-constructor to be used in sharing_clone(); there is no data to
		//! share — only the universe size transfers (the criterion clones assert its shape)
		Data_Accessor_Universe(const Data_Accessor_Universe &dau, int x) : parent_impl(dau, x), _n(dau._n) {};
	protected:
		unsigned int _n; //!< the universe size — the accessor's only substance

		// the file hooks exist to satisfy the DASM pure-virtual contract; initialize() is
		// overridden above so the read hooks never run, and the shape hooks report the
		// honest degenerate values should diagnostics ever query them
		virtual void initial_data_read() override {} //!< no-op: there is no data to read (the overridden initialize() never calls it)
		virtual void initial_file_prepare() override {} //!< no-op: there is no file to open
		virtual unsigned int file_getNoOfClasses() const override {return 0;} //!< the degenerate shape: 0 classes (no data)
		virtual unsigned int file_getNoOfFeatures() const override {return _n;} //!< the universe size \p _n
		virtual IDXTYPE file_getClassSize(unsigned int) const override {return 0;} //!< the degenerate shape: no class has any samples
};

} // namespace
#endif // FSTDATAACCESSORUNIVERSE_H ///:~
