#ifndef FSTSEARCH_H
#define FSTSEARCH_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    search.hpp
   \brief   Defines interface for search method implementations
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
#include <sstream>
#include <string>
#include <ctime>
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

//! Message accumulator for Search::report() - collects one progress message exactly as an std::ostringstream would, for delivery either to an attached Search_Observer or to the search's output stream
class msgbuf {
public:
	template<class T> msgbuf& operator<<(const T& t) {_s<<t; return *this;} //!< appends \p t via std::ostringstream's own operator<<, like building any ostream message
	msgbuf& operator<<(std::ostream& (*manip)(std::ostream&)) {_s<<manip; return *this;} //!< applies a stream manipulator (std::endl, std::flush) to the accumulator
	std::ostream& stream() {return _s;} //!< for legacy print(std::ostream&)-style helpers appending into the message
	std::string str() const {return _s.str();} //!< the accumulated message text, as passed to Search_Observer::on_progress()/on_solution()
private:
	std::ostringstream _s; //!< the underlying accumulator every operator<< overload writes into
};

/*! \brief Observer interface for search progress reporting

    Attach via Search::set_observer() to take over delivery of search progress:
    on_progress() receives each pre-formatted progress message, on_solution()
    additionally fires whenever a search announces a new (sub-)solution.
    With no observer attached the search prints the same messages as syncout
    text to its os stream.
    A default-constructed Search_Observer swallows everything (quiet mode). */
template<class RETURNTYPE, class SUBSET>
class Search_Observer {
public:
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to a feature subset (not used by the interface itself; kept for observer implementations' convenience)
	virtual ~Search_Observer() {}
	virtual void on_progress(const std::string &msg) {} //!< one pre-formatted progress message
	virtual void on_solution(const RETURNTYPE value, const SUBSET &sub, const std::string &msg) {} //!< a new (sub-)solution has been reached (msg = its pre-formatted text form); copy sub if kept beyond the call
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search;

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr);

//! abstract class, defines interface for search method implementations
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search { // abstract class
	//! Streams a search's print() to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr);
public:
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the criterion the search optimizes
	typedef std::shared_ptr<SUBSET> PSubset;                   //!< shared pointer to a feature subset
	typedef std::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker; //!< shared pointer to the result tracker enabled via enable_result_tracking()
	Search(): _detail(NORMAL) {}
	virtual ~Search() {}

	/*! \brief Runs the search; the central entry point of every search method.

		\param target_d requested subset size; 0 asks the method to optimize the
		       subset size as well (not every method supports this - see the concrete class)
		\param result receives the criterion value of the returned subset; its value
		       is meaningful only if the call returns true
		\param sub non-null subset object dimensioned for the full feature set
		       (its get_n() defines n); on success it holds the selected features.
		       Its incoming contents are ignored unless the concrete method has been
		       told to start from them (Search_Sequential::enable_initial_subset())
		\param crit the criterion to maximize; must have been initialize()d before this call
		\param os stream the progress messages are written to when no observer is attached
		\return false if no valid subset was produced - the criterion failed to evaluate,
		        the requested target_d is not reachable, or no candidate could be
		        generated; \p result and \p sub are then left meaningless

		While the search runs, an attached observer (set_observer()) receives
		on_solution() once per newly reached best (sub-)solution - typically many
		times per search() call - and on_progress() for all other messages. */
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) =0;

	void enable_result_tracking(PResultTracker tracker) {_tracker=tracker;} //!< assigns a result tracker to the search process, i.e., enables logging of all evaluated subsets for future reuse (re-evaluation, etc.)
	void disable_result_tracking() {_tracker.reset();} //!< detaches any result tracker (result_tracker_active() then returns false)
	bool result_tracker_active() const {return _tracker.get()!=0;} //!< true if a result tracker was assigned via enable_result_tracking() and not since disabled

	void set_output_detail(OutputDetail detail) {_detail=detail;} //!< sets the amount of information that is logged to output stream throughout the course of search
	OutputDetail get_output_detail() const {return _detail;} //!< the currently configured output detail level
	bool output_normal() const {return _detail==NORMAL || _detail==DETAILED;} //!< true unless output detail is SILENT
	bool output_detailed() const {return _detail==DETAILED;} //!< true only at the DETAILED output level

	typedef Search_Observer<RETURNTYPE,SUBSET> OBSERVER;      //!< the observer interface type this search reports through (see Search_Observer)
	typedef std::shared_ptr<OBSERVER> PObserver;              //!< shared pointer to an OBSERVER, as set via set_observer()
	void set_observer(PObserver observer) {_observer=observer;} //!< redirects search progress reporting away from the default syncout text output to \p observer

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the search and its configuration; base default writes nothing

protected:
	//! delivers one pre-formatted progress message: to the observer if attached, otherwise as syncout text to os
	void report(std::ostream& os, const msgbuf &m) const {
		if(_observer) _observer->on_progress(m.str());
		else {std::ostringstream sos; sos<<m.str(); syncout::print(os,sos);}
	}
	//! like report(), for sites announcing a new (sub-)solution; fires on_solution instead
	void report_solution(std::ostream& os, const RETURNTYPE value, const SUBSET &sub, const msgbuf &m) const {
		if(_observer) _observer->on_solution(value,sub,m.str());
		else {std::ostringstream sos; sos<<m.str(); syncout::print(os,sos);}
	}
	PResultTracker _tracker;   //!< the assigned result tracker, or null (see enable_result_tracking()/result_tracker_active())
	OutputDetail _detail;      //!< the output detail level set via set_output_detail(); read by output_normal()/output_detailed()
	PObserver _observer;       //!< the attached progress observer, or null (see set_observer(), report(), report_solution())
};

//! Streams a search's print() to \p os.
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr) {
	return sr.print(os);
}

} // namespace
#endif // FSTSEARCH_H ///:~
