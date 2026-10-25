#ifndef FSTRESULTTRACKER_H
#define FSTRESULTTRACKER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    result_tracker.hpp
   \brief   Defines interface for classes that enable collecting multiple results
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
#include <cstddef>
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

/*! \brief Abstract class, defines interface for classes that enable collecting multiple results. 

    The reason to do this is to enable a) feature preference fragility
    workaround, i.e., employing secondary criterion to select a result that
    may prove better than the seemingly best one, b) evaluation of feature
    selection stability, c) additional feature selection schemes, etc..
*/ 
template<class RETURNTYPE, class SUBSET>
class Result_Tracker : public Clonable { // abstract class
public:
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a tracked feature subset
	Result_Tracker(): _detail(NORMAL) {notify("Result_Tracker constructor.");}
	Result_Tracker(const Result_Tracker& rt): _detail(rt._detail) {notify("Result_Tracker copy-constructor.");} //!< copies the output-detail setting only -- concrete trackers copy their own collected results separately
	virtual ~Result_Tracker() {}

	//! Structure to hold [subset,criterion value] temporary solutions in the course of search
	typedef struct {
		RETURNTYPE value; //!< the criterion value of the recorded subset
		PSubset sub;      //!< the recorded subset itself
	} ResultRec;
	typedef ResultRec* PResultRec; //!< raw pointer to a ResultRec, as used by get_first()/get_next() iteration

	virtual bool add(const RETURNTYPE value, const PSubset sub)=0; //!< records one (value, subset) result; returns false if the tracker declines to keep it (concrete trackers may restrict by subset size, a value margin from the best known result, or a capacity limit -- see the concrete tracker for its actual inclusion rule)
	virtual void clear() =0; //!< discards all recorded results
	virtual std::size_t size() const =0; //!< number of results currently held (std::size_t rather than 'long': on LLP64 platforms 'long' is 32 bits and would narrow the container's own size type)
	virtual bool get_first(PResultRec &r) =0; //!< starts an iteration over the held results, best first; false if the tracker holds nothing. The record \p r points into the tracker's own storage and is invalidated by any add(), join() or clear()
	virtual bool get_next(PResultRec &r) =0; //!< hands out the next held result; false once the iteration is exhausted. A tracker has a single shared iteration cursor, so two iterations of the same object cannot be interleaved
	virtual void join(Result_Tracker &src); //!< adds the contents of \p src to this tracker, one add() per record, so this tracker's own inclusion rules apply; \p src is only read, but its iteration cursor is consumed

	Result_Tracker* clone() const override {throw fst_error("Result_Tracker::clone() not supported, use Result_Tracker::stateless_clone() instead.");}
	Result_Tracker* sharing_clone() const override {throw fst_error("Result_Tracker::sharing_clone() not supported, use Result_Tracker::stateless_clone() instead.");}
	Result_Tracker* stateless_clone() const override =0;

	void set_output_detail(OutputDetail detail) {_detail=detail;} //!< sets the amount of information that is logged to output stream throughout the course of search
	OutputDetail get_output_detail() const {return _detail;} //!< the currently configured output detail level
	bool output_normal() const {return _detail==NORMAL || _detail==DETAILED;} //!< true unless output detail is SILENT
	bool output_detailed() const {return _detail==DETAILED;} //!< true only at the DETAILED output level
protected:
	OutputDetail _detail; //!< the output detail level set via set_output_detail(); read by output_normal()/output_detailed()
};

template<class RETURNTYPE, class SUBSET>
void Result_Tracker<RETURNTYPE,SUBSET>::join(Result_Tracker &src)
{
	PResultRec prr;
	bool found=src.get_first(prr);
	while(found) {
		add(prr->value,prr->sub);
		found=src.get_next(prr);
	}
}

} // namespace
#endif // FSTRESULTTRACKER_H ///:~
