#ifndef FSTSEARCHSEQSTEP_H
#define FSTSEARCHSEQSTEP_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    seq_step.hpp
   \brief   Defines interface for implementations of one selection step in sequential search type of methods
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
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"

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

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Sequential_Step;

//! Streams a selection step's print() to \p os.
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& ss);

/*! \brief Abstract class, defines interface for implementations of one selection step in sequential search type of methods

	\note To prevent misunderstanding: _generalization_level here does not relate to classification performance on unknown data,
	      but to "generalization" of sequential steps in the course of search, i.e., the number of features added/removed while 
	      generating candidate subsets (higher generalization=more candidates)
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Sequential_Step { // abstract class
	//! Streams a selection step's print() to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& ss);
public:
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the criterion this step evaluates candidates against
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a feature subset
	typedef std::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker; //!< shared pointer to the result tracker enabled via enable_result_tracking()
	Sequential_Step():_detail(NORMAL) {}
	virtual ~Sequential_Step() {}

	// NOTE: crit is expected to be already initialize()d before use here
	/*! \brief Performs one selection step: evaluates the candidate subsets reachable from \p sub and leaves the best of them in \p sub.

		This is the extension point of the whole sequential search family - a
		different candidate-selection strategy (prefiltering, ensemble voting,
		budgeted sampling, ...) is introduced by implementing this one method.

		The candidates are the subsets offered by sub->getFirstCandidateSubset() and
		sub->getNextCandidateSubset(): those differing from \p sub in
		\p _generalization_level features, added in forward mode and removed in
		backward mode. An implementation may evaluate all of them or any subset of
		them, but must leave the chosen candidate in \p sub and its criterion value
		in \p result.

		\param result receives the criterion value of the chosen candidate; meaningful only if true is returned
		\param sub in: the subset the candidates are generated from; out: the chosen candidate. \p sub doubles as the traversal state, so its contents are modified even when the call returns false
		\param crit the criterion the candidates are evaluated by; must be non-null and already initialize()d
		\param _generalization_level number of features added or removed at once (>=1, and at most the number of features)
		\param os stream the step's progress messages are written to
		\return false if no candidate subset could be generated at all (nothing left to add or remove), or if a criterion evaluation failed; the caller must then treat the step as not performed */
	virtual bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os) =0;

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the step and its configuration; base default writes nothing
public:
	void enable_result_tracking(PResultTracker tracker) {_tracker=tracker;} //!< assigns a result tracker to this step, i.e., enables logging of all evaluated candidate subsets for future reuse (re-evaluation, etc.)
	void disable_result_tracking() {_tracker.reset();} //!< detaches any result tracker (result_tracker_active() then returns false)
	bool result_tracker_active() const {return _tracker.get()!=0;} //!< true if a result tracker was assigned via enable_result_tracking() and not since disabled

	void set_output_detail(OutputDetail detail) {_detail=detail;} //!< sets the amount of information that is logged to output stream throughout the course of search
	OutputDetail get_output_detail() const {return _detail;} //!< the currently configured output detail level
	bool output_normal() const {return _detail==NORMAL || _detail==DETAILED;} //!< true unless output detail is SILENT
	bool output_detailed() const {return _detail==DETAILED;} //!< true only at the DETAILED output level
protected:
	PResultTracker _tracker; //!< the assigned result tracker, or null (see enable_result_tracking()/result_tracker_active())
	OutputDetail _detail;    //!< the output detail level set via set_output_detail(); read by output_normal()/output_detailed()
};

//! Streams a selection step's print() to \p os.
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& ss) {
	return ss.print(os);
}

} // namespace
#endif // FSTSEARCHSEQSTEP_H ///:~
