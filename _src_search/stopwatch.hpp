#ifndef FSTSTOPWATCH_H
#define FSTSTOPWATCH_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    stopwatch.hpp
   \brief   Simple tool to measure run time
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

#include <iostream>
#include <ctime>
#ifdef FST_THREADED
#include <mutex>
#endif
#include "error.hpp"
#include "global.hpp"

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

/*! \brief Simple stopwatch to measure run time of search algorithms

    The clock starts when the object is constructed and there is no way to stop,
    pause or reset it - measure another interval by constructing another StopWatch.
    Resolution is one second (wall-clock time()), which is what the search
    algorithms' progress messages need; it is not a profiling instrument.
*/
class StopWatch {
	//! Streams a StopWatch's print() (formatted elapsed time) to \p os.
	friend std::ostream& operator<<(std::ostream& os, const StopWatch& sw);
public:
	StopWatch() {time(&first); notify("StopWatch constructor.");}
	~StopWatch() {notify("StopWatch destructor.");}
	//! \brief Returns the number of whole seconds elapsed since construction (wall-clock via time(), one-second resolution).
	double time_elapsed() const // in seconds
	{
#ifdef FST_THREADED
		std::unique_lock<std::mutex> locktimeelapsed(timeelapsed_mutex);
#endif
		time_t current;
		time(&current);
		return difftime(current,first);
	}
	//! \brief Writes elapsed time formatted as "StopWatch time: D:HH:MM:SS  (equals N seconds)" to \p os.
	//! \param os the stream to write to
	std::ostream& print(std::ostream& os) const
	{
		long dif=long(time_elapsed());
		long dif2=dif;
		int days=dif/(60*60*24); dif%=60*60*24;
		int hours=dif/(60*60); dif%=60*60;
		int minutes=dif/60; dif%=60;
		int seconds=dif;
		os << "StopWatch time: "<< days << ":"; 
		if(hours<10) os << "0";
		os << hours << ":";
		if(minutes<10) os << "0";
		os << minutes << ":";
		if(seconds<10) os << "0";
		os << seconds << "  (equals " << dif2 << " seconds)";
		return os;
	}
private:
	time_t first; //!< wall-clock time captured at construction, the time_elapsed() baseline
#ifdef FST_THREADED
	mutable std::mutex timeelapsed_mutex; //!< serializes concurrent time_elapsed() calls on one StopWatch (FST_THREADED only); \c first itself is written at construction and never changes afterwards
#endif
};

//! Streams a StopWatch's print() (formatted elapsed time) to \p os.
inline std::ostream& operator<<(std::ostream& os, const StopWatch& sw) {
	return sw.print(os);
}

} // namespace
#endif // FSTSTOPWATCH_H ///:~
