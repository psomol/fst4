#ifndef FSTGLOBAL_H
#define FSTGLOBAL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    global.hpp
   \brief   Global definitions
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

#include <string>
#include <ostream>
#include <iostream> // std::cout in the inline notify() bodies (DEBUG builds)
#include <sstream>
#ifdef FST_THREADED
#include <mutex>
#endif

/*! \namespace FST
    \brief The Feature Selection Toolbox 4 library: every library component -- data access,
    criteria and classifiers, search algorithms, result trackers and the global infrastructure --
    lives in this namespace.

    Application code refers to it by explicit qualification; the fst4 command-line tool keeps its
    own types apart in namespace fst4_cli. */
namespace FST {

//! search algorithms' output extent
enum OutputDetail {
  SILENT=0, //!< completely silent operation
  NORMAL=1, //!< informs about new found solutions and time elapsed etc.
  DETAILED=2//!< includes much detail including variable contents etc. (may render output unreadably cluttered)
};

/*! \brief Serializes textual reporting from concurrent threads.

    The class holds no state and is never instantiated (the constructor is private); it exists
    only as the scope of the static print() service and of the mutex that guards it. Every
    output produced from code that may run inside a threaded search must go through
    syncout::print() -- writing to std::cout directly from several threads interleaves
    characters and, in the worst case, stalls the reporting. */
class syncout
{
public:
	//! \brief Writes \p sos's buffered content to \p os, serializing concurrent writers under FST_THREADED via a shared mutex.
	//! \param os the output stream to write to
	//! \param sos the buffered message content to flush
	static void print(std::ostream& os, std::ostringstream &sos)
	{
#ifdef FST_THREADED
		std::unique_lock<std::mutex> lockio(get_io_mutex());
#endif
		os << sos.str() << std::flush;
	}
private:
#ifdef FST_THREADED
	//! \brief Returns the process-wide mutex guarding concurrent output via print() (FST_THREADED builds only).
	static std::mutex& get_io_mutex()
	{
		static std::mutex io_mutex;
		return io_mutex;
	}
#endif
	syncout();
};

// notify(): DEBUG-mode-only diagnostic output. The overloads are inline, so the library is
// header-only and needs no compiled translation unit; _src_global/global.cpp is an empty
// compatibility stub for build setups that still list it.
/*! \brief Prints \p msg followed by a newline via syncout::print(); a no-op unless the
    preprocessor macro DEBUG is defined at compile time.
    Used throughout FST4 for lightweight diagnostic tracing.
    \param msg the diagnostic message to print */
inline void notify(const std::string msg)
{
#ifdef DEBUG
	std::ostringstream ss;
	ss << msg << std::endl;
	syncout::print(std::cout,ss);
#else
	(void)msg;
#endif
}

/*! \brief Prints \p msg1, \p num and \p msg2 concatenated, DEBUG builds only (see notify(const std::string)).
    \param msg1 leading message text
    \param num a numeric value interleaved with the message text
    \param msg2 optional trailing message text (defaults to empty) */
inline void notify(const std::string msg1, const double num, const std::string msg2="")
{
#ifdef DEBUG
	std::ostringstream ss;
	ss << msg1 << num << msg2 << std::endl;
	syncout::print(std::cout,ss);
#else
	(void)msg1; (void)num; (void)msg2;
#endif
}

/*! \brief Prints \p msg1, \p num1, \p msg2, \p num2, \p msg3 concatenated, DEBUG builds only (see notify(const std::string)).
    \param msg1 leading message text
    \param num1 first numeric value
    \param msg2 message text between the two numeric values
    \param num2 second numeric value
    \param msg3 optional trailing message text (defaults to empty) */
inline void notify(const std::string msg1, const double num1, const std::string msg2, const double num2, const std::string msg3="")
{
#ifdef DEBUG
	std::ostringstream ss;
	ss << msg1 << num1 << msg2 << num2 << msg3 << std::endl;
	syncout::print(std::cout,ss);
#else
	(void)msg1; (void)num1; (void)msg2; (void)num2; (void)msg3;
#endif
}

/*! \brief Prints \p msg1, \p num1, \p msg2, \p num2, \p msg3, \p num3, \p msg4 concatenated, DEBUG builds only (see notify(const std::string)).
    \param msg1 leading message text
    \param num1 first numeric value
    \param msg2 message text between the first and second numeric values
    \param num2 second numeric value
    \param msg3 message text between the second and third numeric values
    \param num3 third numeric value
    \param msg4 optional trailing message text (defaults to empty) */
inline void notify(const std::string msg1, const double num1, const std::string msg2, const double num2, const std::string msg3, const double num3, const std::string msg4="")
{
#ifdef DEBUG
	std::ostringstream ss;
	ss << msg1 << num1 << msg2 << num2 << msg3 << num3 << msg4 << std::endl;
	syncout::print(std::cout,ss);
#else
	(void)msg1; (void)num1; (void)msg2; (void)num2; (void)msg3; (void)num3; (void)msg4;
#endif
}

//! \brief Returns the greater of \p x and \p y (strict '>' comparison; ties return \p y).
//! \param x first value
//! \param y second value
template<typename VALUE>
inline VALUE maxf(const VALUE &x, const VALUE &y)
{
	return (x>y)?x:y;
}

//! \brief Returns the smaller of \p x and \p y (strict '<' comparison; ties return \p y).
//! \param x first value
//! \param y second value
template<typename VALUE>
inline VALUE minf(const VALUE &x, const VALUE &y)
{
	return (x<y)?x:y;
}

} //namespace
#endif // FSTGLOBAL_H ///:~
