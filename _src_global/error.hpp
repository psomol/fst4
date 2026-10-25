#ifndef FSTERROR_H
#define FSTERROR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    error.hpp
   \brief   FST4 specific error passing class
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    October 2010 (last updated July 2026)
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

#include <stdexcept>
#include <string>

namespace FST {

/*! \brief FST4 specific error passing class

    Derives from std::runtime_error, so a caller that catches std::exception (or
    std::runtime_error) catches every error FST4 throws; what() returns the message
    supplied at construction, prefixed with "[FST_ERROR!] ". The optional numeric code is
    diagnostic only -- no FST4 component branches on it, and the same code may be reused
    by unrelated components, so recovery logic must not depend on it. */
class fst_error : public std::runtime_error {
	public:
		/*! \brief Constructs the error, prefixing \p _err with "[FST_ERROR!] " for the inherited std::runtime_error::what() message.
		    \param _err human-readable error message
		    \param _errcode optional numeric error code, retrievable via code(); diagnostic only */
		fst_error(const std::string& _err, const int _errcode=0) : std::runtime_error("[FST_ERROR!] "+_err), _code(_errcode) {}
		int code() const {return _code;} //!< returns the numeric error code supplied at construction (0 if none was given)
	private:
		const int _code; //!< numeric error code supplied at construction
};

} //namespace
#endif // FSTERROR_H ///:~
