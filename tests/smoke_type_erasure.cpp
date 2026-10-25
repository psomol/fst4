/*!======================================================================
   Feature Selection Toolbox 4 - type-erasure smoke test
   ---------------------------------------------------------

   \file    smoke_type_erasure.cpp
   \brief   Proves that search/criterion/classifier/accessor components can be
            instantiated against their ABSTRACT base classes and selected at
            runtime.
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work

   Unlike the demos, which bind every template parameter to a concrete class,
   this program instantiates each component once with the abstract base as the
   template argument and picks concrete implementations at runtime, factory
   style - the exact pattern the fst4 command-line tool relies on.

   Type-erased seams exercised:
     - CRITERION    = FST::Criterion<...>            (abstract)
     - CLASSIFIER   = FST::Classifier<...>           (abstract, inside wrapper)
     - DATAACCESSOR = FST::Data_Accessor<...>        (abstract)
     - EVALUATOR    = FST::Sequential_Step<...>      (abstract)
   The threaded evaluator additionally exercises covariant clone() through the
   abstract Criterion/Classifier/Data_Accessor bases (worker-thread cloning).

   The few calls that still need the concrete accessor handle (initialize() and
   setSplittingDepth() live below the root base) are marked CONCRETE-ONLY.

   Usage: smoke_type_erasure [gmaha|wknn|all] [seq|thread|all]
          (no arguments = all four combinations, deterministic output)
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
#include <cstdlib>
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
#include "criterion_wrapper.hpp"
#include "distance_euclid.hpp"
#include "classifier_knn.hpp"
#include "seq_step_straight.hpp"
#include "seq_step_straight_threaded.hpp"
#include "search_seq_sfs.hpp"
#include "search_branch_and_bound_improved_threaded.hpp" // compile coverage for the deprecated shims: no demo or
#include "search_exhaustive_threaded.hpp"    // CLI translation unit includes them, so without these lines
#include "search_monte_carlo_threaded.hpp"   // nothing would compile them at all
#include "search_ranking_bif_threaded.hpp"

typedef double RETURNTYPE; typedef double DATATYPE; typedef double REALTYPE;
typedef unsigned int IDXTYPE; typedef unsigned int DIMTYPE; typedef short BINTYPE;
typedef FST::Subset<BINTYPE,DIMTYPE> SUBSET;
typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;

// ---- the type-erased seams: ABSTRACT bases as template arguments ----
typedef FST::Data_Accessor<DATATYPE,IDXTYPE> DATAACCESSOR;
typedef FST::Criterion<RETURNTYPE,SUBSET> CRITERION;
typedef FST::Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CLASSIFIER;
typedef FST::Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVALUATOR;

// ---- concrete components, each instantiated ONCE against the abstract seams ----
typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> ACCESSOR_TRN;
typedef FST::Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CRIT_GMAHA;
typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DIST_EUCLID; // CONCRETE-ONLY: Classifier_kNN does `new DISTANCE`, so the distance stays a compile-time strategy (one kNN instantiation per distance in the factory)
typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DIST_EUCLID> CLASSIF_KNN;
typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> CRIT_WRAPPER;
typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_SEQ;
typedef FST::Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> EVAL_THREAD;

// runtime string -> constructed criterion, initialize()d at factory scope where types are known
std::shared_ptr<CRITERION> make_criterion(const std::string &name, const std::shared_ptr<DATAACCESSOR> &da)
{
	if(name=="gmaha") {
		std::shared_ptr<CRIT_GMAHA> c(new CRIT_GMAHA);
		c->initialize(da);
		return c;
	}
	if(name=="wknn") {
		std::shared_ptr<CLASSIF_KNN> knn(new CLASSIF_KNN);
		knn->set_k(3);
		std::shared_ptr<CRIT_WRAPPER> c(new CRIT_WRAPPER);
		c->initialize(knn,da);
		return c;
	}
	throw FST::fst_error("unknown criterion '"+name+"' (expected gmaha|wknn)");
}

std::shared_ptr<EVALUATOR> make_evaluator(const std::string &name)
{
	if(name=="seq")    return std::shared_ptr<EVALUATOR>(new EVAL_SEQ);
	if(name=="thread") return std::shared_ptr<EVALUATOR>(new EVAL_THREAD(2)); // the thread count is a runtime argument
	throw FST::fst_error("unknown evaluator '"+name+"' (expected seq|thread)");
}

void run_combo(const std::string &crit_name, const std::string &eval_name,
               const std::shared_ptr<DATAACCESSOR> &da,
               const std::shared_ptr<ACCESSOR_TRN> &da_concrete)
{
	std::cout << "==== criterion=" << crit_name << " evaluator=" << eval_name << " ====" << std::endl;
	// CONCRETE-ONLY: setSplittingDepth() lives on Data_Accessor_Splitting, not on the root base
	da_concrete->setSplittingDepth(0);
	if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");

	std::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
	sub->deselect_all();
	std::shared_ptr<CRITERION> crit=make_criterion(crit_name,da);
	std::shared_ptr<EVALUATOR> eval=make_evaluator(eval_name);
	FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> srch(eval);
	srch.set_search_direction(FST::FORWARD);
	std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *crit << std::endl << std::endl;

	RETURNTYPE critval_train, critval_test;
	const DIMTYPE d=7;
	if(!srch.search(d,critval_train,sub,crit,std::cout)) throw FST::fst_error("Search not finished.");
	std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value=" << critval_train << std::endl << std::endl;

	// validate through the abstract Classifier seam
	std::shared_ptr<CLASSIF_KNN> knn(new CLASSIF_KNN); knn->set_k(3);
	std::shared_ptr<CLASSIFIER> cls(knn);
	da_concrete->setSplittingDepth(0); // CONCRETE-ONLY: as above
	cls->train(da,sub);
	cls->test(critval_test,da);
	std::cout << "Validated 3-NN accuracy=" << critval_test << std::endl << std::endl;
}

int main(int argc, char *argv[])
{
	try{
		{ // compile+construct coverage for the deprecated threaded shims no other TU includes;
		  // deliberately not run - the merged classes + engine injection are the supported path,
		  // the shims exist for compatibility with earlier FST4 sources only
			FST::Search_Branch_And_Bound_Improved_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> bbit_shim(2);
			(void)bbit_shim;
			FST::Search_Exhaustive_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> ext_shim(2);
			(void)ext_shim;
			FST::Search_Monte_Carlo_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> mct_shim(2);
			(void)mct_shim;
			FST::Search_Ranking_BIF_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> bift_shim(2);
			(void)bift_shim;
		}
		const std::string crit_arg=(argc>1?argv[1]:"all");
		const std::string eval_arg=(argc>2?argv[2]:"all");
		std::vector<std::string> crits, evals;
		if(crit_arg=="all") {crits.push_back("gmaha"); crits.push_back("wknn");} else crits.push_back(crit_arg);
		if(eval_arg=="all") {evals.push_back("seq"); evals.push_back("thread");} else evals.push_back(eval_arg);

		std::cout << "Starting type-erasure smoke test (runtime component selection)..." << std::endl << std::endl;

		PSPLITTER dsp(new SPLITTER5050());
		std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
		std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>);
		splitters->push_back(dsp);
		std::shared_ptr<ACCESSOR_TRN> da_concrete(new ACCESSOR_TRN("data/wdbc_30.trn",splitters,dsc));
		da_concrete->initialize(); // CONCRETE-ONLY: initialize() lives below the root base; called at factory scope where the concrete type is known
		std::shared_ptr<DATAACCESSOR> da(da_concrete); // the type-erased handle every component sees

		for(std::vector<std::string>::const_iterator c=crits.begin();c!=crits.end();++c)
			for(std::vector<std::string>::const_iterator e=evals.begin();e!=evals.end();++e)
				run_combo(*c,*e,da,da_concrete);

		std::cout << "Type-erasure smoke test finished successfully." << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl; return 1;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; return 1;}
	return 0;
}
