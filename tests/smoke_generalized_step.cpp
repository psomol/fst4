/*!======================================================================
   Feature Selection Toolbox 4 - generalized-step probe
   ---------------------------------------------------------

   \file    smoke_generalized_step.cpp
   \brief   Probes Subset::get_combinations()/getNoOfCandidateSubsets() and the
            threaded generalized sequential step that depends on them.
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work

   get_combinations(n,k) must stay exact for every n and g a generalized
   sequential step can meet: an implementation computing full factorials in
   32-bit arithmetic would throw "numeric representation overflow" already for
   n>=13 with k>=2 (13! > 2^32), which would make getNoOfCandidateSubsets(g)
   unusable for generalization level g>=2 beyond toy n and thereby block
   threaded evaluation of generalized sequential steps (the threaded engine
   sizes its result storage by the candidate count). This probe verifies:
     1. candidate-subset counts for g>=2 at n>=13,
     2. the overflow throw still fires for genuinely too-large C(n,k),
     3. the threaded generalized step end-to-end: (G)SFS with G=2 on wdbc_30
        run both sequentially and threaded must yield the identical result
        (the threaded generalized-step path had no other coverage).

   Usage: smoke_generalized_step        (no arguments, deterministic output)
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
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "subset.hpp"
#include "data_intervaller.hpp"
#include "data_splitter.hpp"
#include "data_splitter_5050.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "criterion_normal_gmahalanobis.hpp"
#include "seq_step_straight.hpp"
#include "seq_step_straight_threaded.hpp"
#include "search_seq_sfs.hpp"

typedef double RETURNTYPE; typedef double DATATYPE; typedef double REALTYPE;
typedef unsigned int IDXTYPE; typedef unsigned int DIMTYPE; typedef short BINTYPE;
typedef FST::Subset<BINTYPE,DIMTYPE> SUBSET;
typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_TRN;

// abstract seams as in smoke_type_erasure: one Search_SFS instantiation serves
// both the sequential and the threaded evaluator
typedef FST::Data_Accessor<DATATYPE,IDXTYPE> DATAACCESSOR;
typedef FST::Criterion<RETURNTYPE,SUBSET> CRITERION;
typedef FST::Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATOR;
typedef FST::Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_GMAHA;
typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_SEQ;
typedef FST::Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_THREAD;

static int failures=0;

// a factorial-based implementation would throw the overflow error for every case below
void check_count(const DIMTYPE n, const DIMTYPE g, const DIMTYPE expected)
{
	SUBSET sub(n);
	sub.deselect_all();
	const DIMTYPE got=sub.getNoOfCandidateSubsets(g,false/*reverse*/);
	const bool ok=(got==expected);
	if(!ok) ++failures;
	std::cout << "count n=" << n << " g=" << g << ": got " << got << ", expected " << expected << (ok?" OK":" MISMATCH") << std::endl;
}

// the overflow throw must be preserved for genuinely too-large C(n,g)
void check_overflow_throw(const DIMTYPE n, const DIMTYPE g)
{
	SUBSET sub(n);
	sub.deselect_all();
	try {
		const DIMTYPE got=sub.getNoOfCandidateSubsets(g,false/*reverse*/);
		++failures;
		std::cout << "overflow n=" << n << " g=" << g << ": NOT THROWN (got " << got << ") MISMATCH" << std::endl;
	}
	catch(FST::fst_error &e) {
		std::cout << "overflow n=" << n << " g=" << g << ": correctly thrown (" << e.what() << ") OK" << std::endl;
	}
}

// one (G)SFS run; returns "criterion value / selected subset" signature for comparison
std::string run_gsfs(const std::string &eval_name, const std::shared_ptr<DATAACCESSOR> &da, const std::shared_ptr<ACCESSOR_TRN> &da_concrete)
{
	std::cout << "==== evaluator=" << eval_name << " ====" << std::endl;
	da_concrete->setSplittingDepth(0);
	if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");

	std::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
	sub->deselect_all();
	std::cout << "candidate g-tuples per first step: " << sub->getNoOfCandidateSubsets(2,false/*reverse*/) << std::endl;

	std::shared_ptr<CRIT_GMAHA> crit(new CRIT_GMAHA);
	crit->initialize(da);
	std::shared_ptr<EVALUATOR> eval;
	if(eval_name=="thread") eval.reset(new EVAL_THREAD(2)); else eval.reset(new EVAL_SEQ);
	FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> srch(eval);
	srch.set_search_direction(FST::FORWARD);
	srch.set_generalization_level(2);
	std::cout << "Feature selection setup:" << std::endl << srch << std::endl << *crit << std::endl << std::endl;

	RETURNTYPE critval_train;
	const DIMTYPE d=6;
	if(!srch.search(d,critval_train,sub,crit,std::cout)) throw FST::fst_error("Search not finished.");
	std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value=" << critval_train << std::endl << std::endl;

	std::ostringstream sig;
	sig << critval_train << " / " << *sub;
	return sig.str();
}

int main()
{
	try{
		std::cout << "Starting generalized-step probe (get_combinations / threaded generalized step)..." << std::endl << std::endl;

		std::cout << "==== candidate-subset counts for g>=2 ====" << std::endl;
		check_count(13,2,78);
		check_count(15,2,105);      // a 15-feature data set at g=2
		check_count(30,3,4060);
		check_count(64,2,2016);
		check_count(34,17,2333606220u); // largest-magnitude class: still fits 32-bit DIMTYPE
		check_overflow_throw(100,17);   // C(100,17)~6.6e18 exceeds 32-bit DIMTYPE
		check_overflow_throw(1000,500); // exceeds even 64-bit intermediates
		std::cout << std::endl;

		std::cout << "==== (G)SFS, G=2, d=6, gmahalanobis, wdbc_30: threaded vs sequential ====" << std::endl;
		PSPLITTER dsp(new SPLITTER5050());
		std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
		std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>);
		splitters->push_back(dsp);
		std::shared_ptr<ACCESSOR_TRN> da_concrete(new ACCESSOR_TRN("data/wdbc_30.trn",splitters,dsc));
		da_concrete->initialize();
		std::shared_ptr<DATAACCESSOR> da(da_concrete);

		const std::string sig_seq=run_gsfs("seq",da,da_concrete);
		const std::string sig_thr=run_gsfs("thread",da,da_concrete);
		const bool match=(sig_seq==sig_thr);
		if(!match) ++failures;
		std::cout << "threaded result matches sequential: " << (match?"yes OK":"NO MISMATCH") << std::endl << std::endl;

		if(failures>0) {std::cerr << "Generalized-step probe FAILED (" << failures << " check(s))." << std::endl; return 1;}
		std::cout << "Generalized-step probe finished successfully." << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl; return 1;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; return 1;}
	return 0;
}
