#ifndef FSTSEARCHRANKINGBIFTHREADED_H
#define FSTSEARCHRANKINGBIFTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    search_ranking_bif_threaded.hpp
   \brief   DEPRECATED compatibility shim: threaded individual feature ranking is
            now the merged Search_Ranking_BIF with an injected
            Candidate_Evaluator_Threaded
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    March 2011, merged July 2026
   \version 4.2.0

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

#include <cassert>
#include "error.hpp"
#include "global.hpp"
#include "search_ranking_bif.hpp"
#include "candidate_evaluator_threaded.hpp"

namespace FST {

/*! \brief DEPRECATED shim: threaded Best Individual Features, i.e., individual feature ranking

 \deprecated Deprecated since FST4 4.0; scheduled for removal in the next major
 release — construct the merged class with a Candidate_Evaluator_Threaded engine instead.

 Kept for source compatibility with earlier FST4 sources, written when the threaded
 and the sequential variant were separate classes. This class is now nothing but a
 Search_Ranking_BIF constructed with a Candidate_Evaluator_Threaded engine;
 prefer doing that directly in new code. The final best-dimensionality scan in
 Search_Ranking::search() runs through the same engine.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Ranking_BIF_Threaded : public Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class: the merged BIF ranking search this shim configures with a threaded engine
	typedef typename parent::PEvaluatorEngine PEvaluatorEngine; //!< the candidate-evaluation engine type, inherited from parent
	//! \brief Constructs a Search_Ranking_BIF configured with a Candidate_Evaluator_Threaded engine of \p max_threads workers.
	//! \param max_threads pool capacity for the threaded engine (must be greater than zero, asserted)
	explicit Search_Ranking_BIF_Threaded(const unsigned int max_threads=2)
		:Search_Ranking_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(PEvaluatorEngine(new Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(max_threads)))
		{assert(max_threads>0); notify("Search_Ranking_BIF_Threaded constructor.");}
	virtual ~Search_Ranking_BIF_Threaded() {notify("Search_Ranking_BIF_Threaded destructor.");}
};

} // namespace
#endif // FSTSEARCHRANKINGBIFTHREADED_H ///:~
