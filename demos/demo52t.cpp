/* =========================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
*/ /*! 

   \file    demo52t.cpp 
   \brief   Example 52t: (Threaded SFRS) Result regularization using secondary criterion
   \example demo52t.cpp 
            Implements \ref example52t, see also 
            <a href="https://fst.utia.cz/doc/demo52t_cpp.html">Example 52t source code</a>
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
#include "data_splitter_cv.hpp"
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
//#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
#include "criterion_negative.hpp"

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
#include "candidate_evaluator_threaded.hpp" // thread-pooled candidate evaluation: pass its engine to the step constructor
//#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
//#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example52t Example 52t: (Threaded SFRS) Result regularization using secondary criterion.

It is known that feature selection may over-fit. As in the case of over-trained classifiers,
over-selected feature subsets may generalize poorly. This unwanted effect can lead to
serious degradation of generalization ability, i.e., model or decision-rule behavior 
on previously unknown data. It has been suggested (Raudys: Feature Over-Selection, LNCS 4109, 2006, 
or Somol et al., ICPR 2010) that preferring a subset with slightly-worse-than-maximal criterion
value can actually improve generalization. FST4 makes this possible through result tracking
and subsequent selection of alternative solution by means of secondary criterion maximization.
In this example we show a 3-Nearest Neighbor Wrapper based feature selection process, where
the final result is eventually chosen among a group of solutions close enough to the achieved
maximum, so as to optimize the secondary criterion. The group of solutions to select from is defined 
by means of a user-selected margin value (permitted primary criterion value difference from the known
maximum). In this case we show that even the simplest secondary criterion (mere preference of 
smaller subsets) can improve classification accuracy on previously unknown data.
*/
int main()
{
	try{
	const unsigned int max_threads=2; // worker threads of the candidate-evaluation engine (results are identical for any count)
	typedef double RETURNTYPE; 	typedef double DATATYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
	typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
	typedef FST::Criterion_Subset_Size<RETURNTYPE,SUBSET> CRITSUBSIZE;
	typedef FST::Criterion_Negative<CRITSUBSIZE,RETURNTYPE,SUBSET> NEGATIVECRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN> EVALUATOR;
	typedef FST::Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN> ENGINE; // the thread-pooled candidate-evaluation engine injected into the step
	typedef FST::Result_Tracker_Regularizer<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,NEGATIVECRIT> TRACKER;

		std::cout << "Starting Example 52t: (Threaded SFRS) Result regularization using secondary criterion..." << std::endl;
	// keep second half of data for independent testing of final classification performance
		PSPLITTER dsp_outer(new SPLITTER5050());
	// in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		std::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/waveform_40.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		std::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up 3-Nearest Neighbor classifier based on Euclidean distances
		std::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		std::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN);
		wknn->initialize(cknn,da);
	// set-up the standard sequential search step object with a thread-pooled evaluation engine of max_threads workers (options: hybrid, ensemble)
		std::shared_ptr<EVALUATOR> eval(new EVALUATOR(EVALUATOR::PEvaluatorEngine(new ENGINE(max_threads))));
	// set-up Sequential Forward Retreating Search procedure
		FST::Search_SFRS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval);
		srch.set_search_direction(FST::FORWARD);
	// set-up the regularizing result tracker
		std::shared_ptr<TRACKER> tracker(new TRACKER);
	// register the result tracker with the used search step object
		eval->enable_result_tracking(tracker);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wknn << std::endl << *tracker << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		srch.set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw FST::fst_error("Search not finished.");
	// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
		da->setSplittingDepth(0);
		cknn->train(da,sub);
		cknn->test(critval_test,da);
		std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
	// set-up the secondary criterion (regularization criterion); in this case to minimize subset size
		std::shared_ptr<CRITSUBSIZE> critsubsiz(new CRITSUBSIZE); //Criterion_Subset_Size does not need to be initialized
		std::shared_ptr<NEGATIVECRIT> regulcrit(new NEGATIVECRIT(critsubsiz)); //Criterion_Negative does not need to be initialized
	// select final solution among those recorded by tracker (show more alternatives for various margins)
		tracker->set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		for(unsigned int i=1; i<10; i++) 
		{
			RETURNTYPE margin=(double)i*0.001;
			da->setSplittingDepth(1); // necessary with criteria that need access to training data
			if(!tracker->optimize_within_margin(margin,critval_train,critval_test,sub,regulcrit)) throw FST::fst_error("tracker->optimize_within_margin() failed.");
			std::cout << std::endl << "Regularized (margin="<<margin<<") result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
		// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
			da->setSplittingDepth(0);
			cknn->train(da,sub);
			cknn->test(critval_test,da);
			std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
		}
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl; return 1;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; return 1;}
	return 0;
}
