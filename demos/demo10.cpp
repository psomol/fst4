/* =========================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
*/ /*! 

   \file    demo10.cpp 
   \brief   Example 10: Basic Filter-based feature selection
   \example demo10.cpp 
            Implements \ref example10, see also 
            <a href="https://fst.utia.cz/doc/demo10_cpp.html">Example 10 source code</a>
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see 
            <a href="https://fst.utia.cz/contacts.html">Contacts</a>
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
//#include "data_splitter_cv.hpp"
//#include "data_splitter_holdout.hpp"
//#include "data_splitter_leave1out.hpp"
//#include "data_splitter_resub.hpp"
//#include "data_splitter_randrand.hpp"
//#include "data_splitter_randfix.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
//#include "data_scaler_to01.hpp"
//#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

//#include "criterion_normal_bhattacharyya.hpp"
#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
//#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
//#include "classifier_multinom_naivebayes.hpp"
//#include "classifier_svm.hpp"

//#include "search_ranking_bif.hpp"
//#include "search_monte_carlo.hpp"
//#include "search_exhaustive.hpp"
//#include "branch_and_bound_predictor_averaging.hpp"
//#include "search_branch_and_bound_basic.hpp"
//#include "search_branch_and_bound_improved.hpp"
//#include "search_branch_and_bound_partial_prediction.hpp"
//#include "search_branch_and_bound_fast.hpp"
#include "seq_step_straight.hpp"
//#include "candidate_evaluator_threaded.hpp" // thread-pooled candidate evaluation: pass its engine to any search/step constructor
//#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example10 Example 10: Basic Filter-based feature selection.

This is a simple example of filter-based feature selection, as described in the seminal book by Devijver
and Kittler from 1982 - Pattern Recognition: A Statistical Approach. Features are selected here
using the Sequential Forward Selection (SFS) procedure so as to maximize the Generalized Mahalanobis
probabilistic class distance based on the assumption of normality of the data. 
Generalized Mahalanobis is evaluated on the first 50% of data samples.
The selected subset is eventually verified by means of 3-NN classifier accuracy estimation
on the second 50% (independent test) part of the data. SFS is called here in d-parametrized setting, 
invoked by nonzero parameter d in search(d,...). In this scenario the user has to decide about the target subset size.

\note With arbitrary data the assumption of normality may not be fulfilled what would negatively affect the 
feature selection results based on Mahalanobis, Bhattacharyya or Divergence.
*/
int main()
{
	try{
	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR;
	typedef FST::Criterion_Normal_GMahalanobis<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> FILTERCRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,FILTERCRIT> EVALUATOR;
	typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;

		std::cout << "Starting Example 10: Basic Filter-based feature selection..." << std::endl;
	// in the course of search use the first half of data for feature selection and the second half for testing using 3-NN classifier
		PSPLITTER dsp(new SPLITTER5050());
	// do not scale data
		std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp);
		std::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/wdbc_30.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");
	// initiate the storage for subset to-be-selected
		std::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up the normal Generalized Mahalanobis criterion 
		std::shared_ptr<FILTERCRIT> crit(new FILTERCRIT);
		crit->initialize(da); // initialization = normal model parameter estimation on training data part
	// set-up the standard sequential search step object (options: hybrid, ensemble, etc.)
		std::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Sequential Forward Selection search procedure
		FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,FILTERCRIT,EVALUATOR> srch(eval);
		srch.set_search_direction(FST::FORWARD); // try FST::BACKWARD
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *crit << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		const DIMTYPE d=7; // request subset of size d; if set to 0, cardinality will be decided in the course of search
		if(!srch.search(d,critval_train,sub,crit,std::cout)) throw FST::fst_error("Search not finished.");
	// (optionally) the following line is included here just for illustration because srch.search() reports results in itself
		std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value=" << critval_train << std::endl << std::endl;
	// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
		std::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
		da->setSplittingDepth(0);
		cknn->train(da,sub);
		cknn->test(critval_test,da);
		std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl; return 1;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; return 1;}
	return 0;
}
