/* =========================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
*/ /*! 

   \file    demo55.cpp 
   \brief   Example 55: Evaluating Similarity of Two Feature Selection Processes
   \example demo55.cpp 
            Implements \ref example55, see also 
            <a href="https://fst.utia.cz/doc/demo55_cpp.html">Example 55 source code</a>
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
//#include "data_splitter_5050.hpp"
#include "data_splitter_cv.hpp"
//#include "data_splitter_holdout.hpp"
//#include "data_splitter_leave1out.hpp"
//#include "data_splitter_resub.hpp"
#include "data_splitter_randrand.hpp"
//#include "data_splitter_randfix.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
//#include "data_scaler_to01.hpp"
//#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

#include "criterion_normal_bhattacharyya.hpp"
//#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

//#include "distance_euclid.hpp"
#include "distance_L1.hpp"
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
//#include "search_seq_sfs.hpp"
#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
#include "result_tracker_stabileval.hpp"


/*! \section example55 Example 55: Evaluating Similarity of Two Feature Selection Processes.

To study the difference in feature preferences among principally different feature selection methods
or among differently parametrized instances of the same method FST4 provides measures
capable of evaluating the level of similarity between two sets of trials (Somol Novovicova, IEEE, TPAMI, 2010). 
In analogy to stability evaluation (see the demo54 protocol, `demos/demo54.cfg`) for each of the two feature selection scenarios
a series of trials is conducted on various samplings of the same data. In this example ten feature selection 
trials are performed per scenario, each on randomly sampled 95% of the data. In the first scenario 
in each trial the resulting subset is obtained using DOS procedure, optimizing the 3-Nearest
Neighbour accuracy estimated by means of 3-fold cross-validation. In the second scenario
in each trial the resulting subset is obtained using SFFS procedure, maximizing
the Bhattacharyya distance based on normal model. A selection of standard stability measures
is evaluated separately for each of the two scenarios. Eventually the similarity of the two
scenarios is evaluated using analogously founded similarity measures. All measures yield values from
[0,1], where values close to 0 denote low stability/similarity and values close to 1 denote 
high stability/similarity. Note that in this experiment the inter-measures (IATI, ICW, IANHI) 
yield markedly lower values than the corresponding stability measures (ATI, CW, ANHI). 
This illustrates well that considerably different results can be expected from differently founded
feature selection methods.
*/
int main()
{
	try{
	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef std::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
	typedef FST::Data_Splitter_RandomRandom<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRANDRAND;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Distance_L1<DATATYPE,DIMTYPE,SUBSET> DISTANCEL1;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCEL1> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPER;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPER> EVALUATOR1;
	typedef FST::Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> BHATTCRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT> EVALUATOR2;
	typedef FST::Result_Tracker_Stability_Evaluator<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER;

		std::cout << "Starting Example 55: Evaluating Similarity of Two Feature Selection Processes..." << std::endl;
	// set-up ten trials where in each 95% of data is randomly sampled
		PSPLITTER dsp_outer(new SPLITTERRANDRAND(10/*splits=trials*/,95,5));
	// in the course of wrapper based feature subset search (in one trial) use 3-fold cross-validation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		std::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		std::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		std::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/wdbc_30.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("RandRand data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		std::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up result trackers to collect results of each trial in both scenarios
		std::shared_ptr<TRACKER> tracker1(new TRACKER);
		std::shared_ptr<TRACKER> tracker2(new TRACKER);
	// FEATURE SELECTION SCENARIO A (wrapper)
	// set-up 3-Nearest Neighbor classifier based on L1 distances
		std::shared_ptr<CLASSIFIERKNN> cknn1(new CLASSIFIERKNN);
		cknn1->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		std::shared_ptr<WRAPPER> wknn1(new WRAPPER);
		wknn1->initialize(cknn1,da);
	// set-up the standard sequential search step object (option: hybrid, ensemble, threaded)
		std::shared_ptr<EVALUATOR1> eval1(new EVALUATOR1);
	// set-up Dynamic Oscillating Search procedure
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPER,EVALUATOR1> srch1(eval1);
		srch1.set_delta(10);
		sub->deselect_all();
		// Technical remark: with a threaded evaluator (Sequential_Step_Straight constructed with a Candidate_Evaluator_Threaded
		// engine) both the evaluator and the search procedure set-up would have to move inside the trial loop. The threaded engine
		// caches criterion clones - including their data accessor state - for as long as the same criterion object is passed to it,
		// so a splitting-level or split change made outside would not reach the clones. A renewed evaluator set-up regenerates the
		// clones and thus ensures correct threaded criterion evaluation after a split change (see the staleness warning in the
		// class documentation of Candidate_Evaluator_Threaded).
	// run the trials
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << *wknn1 << std::endl << *tracker1 << std::endl << std::endl;
		RETURNTYPE critval_train;
		da->setSplittingDepth(0);
		unsigned int trial=0;
		bool run=da->getFirstSplit(); if(!run) throw FST::fst_error("RandRand data split failed.");
		while(run)
		{
			trial++; std::cout << std::endl<<"TRIAL A"<<trial<< " ---------------------------------------------------------------------"<<std::endl;
			da->setSplittingDepth(1);
			if(!srch1.search(0,critval_train,sub,wknn1,std::cout)) throw FST::fst_error("Search not finished.");
			tracker1->add(critval_train,sub);
			std::cout << std::endl << "(TRIAL A"<<trial<<") Search result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
			da->setSplittingDepth(0);
			run=da->getNextSplit();
		}
	// FEATURE SELECTION SCENARIO B (filter)
	// set-up normal Bhattacharyya distance criterion 
		std::shared_ptr<BHATTCRIT> cb(new BHATTCRIT);
	// set-up the standard sequential search step object (option: hybrid, ensemble, threaded)
		std::shared_ptr<EVALUATOR2> eval2(new EVALUATOR2);
	// set-up Sequential Forward Floating Selection search procedure
		FST::Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT,EVALUATOR2> srch2(eval2);
		srch2.set_search_direction(FST::FORWARD);
	// target subset size must be set because Bhattacharyya is monotonous with respect to subset size (i.e., evaluates full set as the best)
		const DIMTYPE target_size=7;
	// run the trials
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch2 << std::endl << *cb << std::endl << *tracker2 << std::endl << std::endl;
		trial=0;
		da->setSplittingDepth(0);
		run=da->getFirstSplit(); if(!run) throw FST::fst_error("RandRand data split failed.");
		while(run)
		{
			trial++; std::cout << std::endl<<"TRIAL B"<<trial<< " ---------------------------------------------------------------------"<<std::endl;
			cb->initialize(da); // (note that cb initialization = normal model parameter estimation on training data, therefore it must be repeated for each split)
			da->setSplittingDepth(1);
			if(!srch2.search(target_size,critval_train,sub,cb,std::cout)) throw FST::fst_error("Search not finished.");
			tracker2->add(critval_train,sub);
			std::cout << std::endl << "(TRIAL B"<<trial<<") Search result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
			da->setSplittingDepth(0);
			run=da->getNextSplit();
		}
	// evaluate stability of each scenario and similarity of the two scenarios using results collected by trackers
		std::cout<<std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "Scenario A resulting criterion values' mean: " << tracker1->value_mean() << ", std. dev.: " << tracker1->value_stddev() << std::endl;
		std::cout << "Scenario A subset sizes' mean: " << tracker1->size_mean() << ", std. dev.: " << tracker1->size_stddev() << std::endl;
		std::cout << std::endl;
		std::cout << "Scenario A stability_ATI()=" << tracker1->stability_ATI() << std::endl;
		std::cout << "Scenario A stability_CW()=" << tracker1->stability_CW() << std::endl;
		std::cout << "Scenario A stability_ANHI("<<da->getNoOfFeatures()<<")=" << tracker1->stability_ANHI(da->getNoOfFeatures()) << std::endl;
		std::cout<<std::endl;
		std::cout << "Scenario B resulting criterion values' mean: " << tracker2->value_mean() << ", std. dev.: " << tracker2->value_stddev() << std::endl;
		std::cout << "Scenario B subset sizes' mean: " << tracker2->size_mean() << ", std. dev.: " << tracker2->size_stddev() << std::endl;
		std::cout << std::endl;
		std::cout << "Scenario B stability_ATI()=" << tracker2->stability_ATI() << std::endl;
		std::cout << "Scenario B stability_CW()=" << tracker2->stability_CW() << std::endl;
		std::cout << "Scenario B stability_ANHI("<<da->getNoOfFeatures()<<")=" << tracker2->stability_ANHI(da->getNoOfFeatures()) << std::endl;
		std::cout<<std::endl;
		std::cout << "Evaluating similarity between scenario A and scenario B:"<< std::endl;
		std::cout << "similarity measure IATI()=" << tracker1->similarity_IATI(*tracker2) << std::endl;
		std::cout << "similarity measure ICW()=" << tracker1->similarity_ICW(*tracker2) << std::endl;
		std::cout << "similarity measure IANHI("<<da->getNoOfFeatures()<<")=" << tracker1->similarity_IANHI(da->getNoOfFeatures(), *tracker2) << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl; return 1;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl; return 1;}
	return 0;
}
