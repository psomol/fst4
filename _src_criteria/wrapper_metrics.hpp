#ifndef FSTWRAPPERMETRICS_H
#define FSTWRAPPERMETRICS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    wrapper_metrics.hpp
   \brief   Wrapper criterion metrics beyond accuracy (balanced accuracy, F1, AUC, class weights, regression metrics)
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0
   \note    FST4 was developed using MinGW-W64 g++ 15.2 and requires
   \note    \li a C++17 conforming compiler and standard library (no Boost, unlike FST3),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/,
                version 3.37; not distributed with FST4 — drop svm.cpp/svm.h into ./libsvm/, see libsvm/readme.txt)
   \note    Note that LibSVM is required for SVM related tools only.

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

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>

namespace FST {

/*! Metric selector for wrapper criteria and validation reporting.
    All metrics are higher-is-better (search maximizes).
    accuracy (the default, with empty class_weights) is served directly by
    Classifier::test()/Regressor::test() — the metric machinery below is
    bypassed entirely on that path. */
enum class Wrapper_Metric {
	accuracy,          //!< classification: fraction correct (cost-weighted if class_weights set)
	balanced_accuracy, //!< classification: macro-averaged recall (classes absent from the test part are skipped)
	f1,                //!< classification, two-class data: F1 of the positive class, which is Metric_Spec::f1_positive (class index 1 unless set otherwise)
	f1_macro,          //!< classification: unweighted mean of per-class one-vs-rest F1 (0/0 contributes 0)
	auc,               //!< classification: two-class data = area under the ROC curve (exact rank statistic, midrank ties); K>2 = one-vs-rest macro (mean of per-class binary AUCs; classes absent from the test part are skipped)
	r2,                //!< regression: coefficient of determination (SS_tot around the test-part mean)
	neg_rmse,          //!< regression: negated root-mean-square prediction error
	neg_mae            //!< regression: negated mean absolute prediction error
};

//! configuration of the wrapper metric; immutable during search, copied by criterion clones (thread-safe, RNG-free)
template<typename RETURNTYPE>
struct Metric_Spec {
	Wrapper_Metric metric = Wrapper_Metric::accuracy; //!< which metric to compute; accuracy = the default path served by Classifier::test()
	/*! per-class cost weights, indexed by TRUE class; empty = unweighted. A weight scales the
	    whole confusion-matrix ROW of its class, i.e. it is the relative importance of the
	    samples of that class, not of the predictions made for it. Only the ratios matter -
	    every classification metric here is a ratio of weighted counts, so scaling all weights
	    by a common factor leaves the value unchanged. Example: weights (1,5) on two-class data
	    make each class-1 sample count like five class-0 samples, so errors on class 1 become
	    five times as expensive. Must hold at least \c classes entries when non-empty. */
	std::vector<RETURNTYPE> class_weights;
	unsigned int f1_positive = 1; //!< class index the f1 metric scores as positive (two-class data; `--f1-positive`). Affects f1 ONLY - f1-macro averages over all classes, and auc keeps class 1 positive by its own convention
	bool is_default() const {return metric==Wrapper_Metric::accuracy && class_weights.empty();} //!< true iff this spec is the plain unweighted accuracy path (f1_positive is irrelevant unless metric==f1, so it stays out of this test)
	bool is_regression_metric() const {return metric==Wrapper_Metric::r2 || metric==Wrapper_Metric::neg_rmse || metric==Wrapper_Metric::neg_mae;} //!< true for the three regression metrics (r2/neg_rmse/neg_mae), false for the classification metrics
};

//! Returns the canonical CLI name (`--wrapper-metric` value) for \p m, e.g. "balanced-accuracy"; the inverse of wrapper_metric_from_name().
inline const char* wrapper_metric_name(const Wrapper_Metric m)
{
	switch(m) {
		case Wrapper_Metric::accuracy: return "accuracy";
		case Wrapper_Metric::balanced_accuracy: return "balanced-accuracy";
		case Wrapper_Metric::f1: return "f1";
		case Wrapper_Metric::f1_macro: return "f1-macro";
		case Wrapper_Metric::auc: return "auc";
		case Wrapper_Metric::r2: return "r2";
		case Wrapper_Metric::neg_rmse: return "neg-rmse";
		case Wrapper_Metric::neg_mae: return "neg-mae";
	}
	return "accuracy";
}

//! \brief Parses a CLI metric name (`--wrapper-metric` value, e.g. "f1-macro") into \p m.
//! \return false and leaves \p m unchanged if \p name is not a recognized metric name.
inline bool wrapper_metric_from_name(const std::string &name, Wrapper_Metric &m)
{
	if(name=="accuracy") m=Wrapper_Metric::accuracy;
	else if(name=="balanced-accuracy") m=Wrapper_Metric::balanced_accuracy;
	else if(name=="f1") m=Wrapper_Metric::f1;
	else if(name=="f1-macro") m=Wrapper_Metric::f1_macro;
	else if(name=="auc") m=Wrapper_Metric::auc;
	else if(name=="r2") m=Wrapper_Metric::r2;
	else if(name=="neg-rmse") m=Wrapper_Metric::neg_rmse;
	else if(name=="neg-mae") m=Wrapper_Metric::neg_mae;
	else return false;
	return true;
}

/*! computes a classification metric from a classes*classes confusion matrix
    (row-major, row = true class, column = predicted class).
    Returns false when the metric is undefined on this matrix (e.g. no test
    samples, or balanced accuracy with every class absent) — callers treat
    that as "skip this split". */
template<typename RETURNTYPE, typename DIMTYPE>
bool metric_from_confusion(RETURNTYPE &result, const Wrapper_Metric metric, const std::vector<RETURNTYPE> &confusion, const DIMTYPE classes, const std::vector<RETURNTYPE> &class_weights, const DIMTYPE f1_positive=1)
{
	if(classes<2 || confusion.size()<(size_t)classes*(size_t)classes) return false;
	const bool weighted=!class_weights.empty();
	if(weighted && class_weights.size()<(size_t)classes) return false;
	// cost weights scale whole rows = true-class importance (see Metric_Spec::class_weights);
	// cell access below applies them on the fly, so every metric below sees weighted counts
	auto cell=[&](DIMTYPE t, DIMTYPE p)->RETURNTYPE {
		RETURNTYPE v=confusion[(size_t)t*(size_t)classes+(size_t)p];
		return weighted ? v*class_weights[t] : v;
	};
	switch(metric) {
	case Wrapper_Metric::accuracy: {
		RETURNTYPE diag=0.0, total=0.0;
		for(DIMTYPE t=0;t<classes;t++) for(DIMTYPE p=0;p<classes;p++) {RETURNTYPE v=cell(t,p); total+=v; if(t==p) diag+=v;}
		if(!(total>0.0)) return false;
		result=diag/total;
		return true;
	}
	case Wrapper_Metric::balanced_accuracy: {
		RETURNTYPE sum=0.0; DIMTYPE present=0;
		for(DIMTYPE t=0;t<classes;t++) {
			RETURNTYPE rowsum=0.0;
			for(DIMTYPE p=0;p<classes;p++) rowsum+=cell(t,p);
			if(rowsum>0.0) {sum+=cell(t,t)/rowsum; present++;} // classes absent from the test part are skipped
		}
		if(present==0) return false;
		result=sum/(RETURNTYPE)present;
		return true;
	}
	case Wrapper_Metric::f1: { // two-class data; positive class = f1_positive (index 1 unless `--f1-positive` says otherwise)
		if(classes!=2) return false;
		if(f1_positive>=classes) return false;
		const DIMTYPE pos=f1_positive, neg=(DIMTYPE)(1-f1_positive); // two-class only, so the negative class is simply the other index (0 <-> 1)
		const RETURNTYPE tp=cell(pos,pos), fp=cell(neg,pos), fn=cell(pos,neg);
		const RETURNTYPE denom=2.0*tp+fp+fn;
		result=(denom>0.0)?(2.0*tp/denom):(RETURNTYPE)0.0;
		return true;
	}
	case Wrapper_Metric::f1_macro: {
		RETURNTYPE sum=0.0;
		for(DIMTYPE c=0;c<classes;c++) {
			RETURNTYPE tp=cell(c,c), fp=0.0, fn=0.0;
			for(DIMTYPE t=0;t<classes;t++) if(t!=c) fp+=cell(t,c);
			for(DIMTYPE p=0;p<classes;p++) if(p!=c) fn+=cell(c,p);
			const RETURNTYPE denom=2.0*tp+fp+fn;
			sum+=(denom>0.0)?(2.0*tp/denom):(RETURNTYPE)0.0; // 0/0 contributes 0
		}
		result=sum/(RETURNTYPE)classes;
		return true;
	}
	default: return false; // auc needs scores, regression metrics need predictions
	}
}

/*! computes the exact two-class AUC (Mann-Whitney rank statistic with midrank
    tie handling) from per-sample (score, true class in {0,1}) pairs; class 1
    is the positive class and higher scores must indicate class 1.
    Deterministic: a stable sort on the score keeps the caller's (deterministic)
    sample enumeration order within tie groups, and tie groups share their
    midrank, so the value is a pure function of (scores, labels).
    Returns false when either class is absent (AUC undefined) — callers treat
    that as "skip this split". Reorders the input vector. */
template<typename RETURNTYPE, typename DIMTYPE>
bool auc_from_scores(RETURNTYPE &result, std::vector<std::pair<RETURNTYPE,DIMTYPE>> &scores)
{
	const size_t n=scores.size();
	if(n==0) return false;
	size_t n_pos=0;
	for(size_t i=0;i<n;i++) if(scores[i].second==1) n_pos++;
	const size_t n_neg=n-n_pos;
	if(n_pos==0 || n_neg==0) return false;
	std::stable_sort(scores.begin(),scores.end(),
		[](const std::pair<RETURNTYPE,DIMTYPE> &a, const std::pair<RETURNTYPE,DIMTYPE> &b){return a.first<b.first;});
	RETURNTYPE rank_sum_pos=0.0;
	size_t i=0;
	while(i<n) { // walk tie groups; group members share the midrank
		size_t j=i;
		while(j<n && !(scores[i].first<scores[j].first)) j++;
		const RETURNTYPE midrank=((RETURNTYPE)(i+1)+(RETURNTYPE)j)/2.0; // ranks are 1-based: (first+last)/2
		for(size_t k=i;k<j;k++) if(scores[k].second==1) rank_sum_pos+=midrank;
		i=j;
	}
	result=(rank_sum_pos-(RETURNTYPE)n_pos*((RETURNTYPE)n_pos+1.0)/2.0)/((RETURNTYPE)n_pos*(RETURNTYPE)n_neg);
	return true;
}

/*! computes the one-vs-rest macro AUC on multiclass data: the unweighted mean
    over classes of the exact two-class AUC (the auc_from_scores midrank rule)
    computed from the class's own score column against the rest.
    Classes absent from the test part have
    undefined AUC and are skipped — the balanced-accuracy convention; returns
    false when no class yields a defined AUC.
    scores = row-major samples x classes matrix (sample i, class c at
    scores[i*classes+c]); labels = true class indices, same sample order. */
template<typename RETURNTYPE, typename DIMTYPE>
bool auc_macro_from_scores(RETURNTYPE &result, const std::vector<RETURNTYPE> &scores, const std::vector<DIMTYPE> &labels, const DIMTYPE classes)
{
	const size_t n=labels.size();
	if(n==0 || classes<2 || scores.size()<(size_t)n*(size_t)classes) return false;
	RETURNTYPE sum=0.0;
	DIMTYPE defined=0;
	std::vector<std::pair<RETURNTYPE,DIMTYPE>> pairs;
	for(DIMTYPE c=0;c<classes;c++)
	{
		pairs.clear(); pairs.reserve(n);
		for(size_t i=0;i<n;i++) pairs.push_back(std::make_pair(scores[i*(size_t)classes+(size_t)c],(DIMTYPE)((labels[i]==c)?1:0)));
		RETURNTYPE a;
		if(auc_from_scores<RETURNTYPE,DIMTYPE>(a,pairs)) {sum+=a; defined++;}
	}
	if(defined==0) return false;
	result=sum/(RETURNTYPE)defined;
	return true;
}

/*! computes a regression metric from per-sample (true value, prediction) pairs.
    r2 uses SS_tot around the test-part mean; returns false when undefined
    (no samples, or r2 with zero total variance). */
template<typename RETURNTYPE>
bool metric_from_predictions(RETURNTYPE &result, const Wrapper_Metric metric, const std::vector<std::pair<RETURNTYPE,RETURNTYPE>> &pairs)
{
	const size_t n=pairs.size();
	if(n==0) return false;
	switch(metric) {
	case Wrapper_Metric::r2: {
		RETURNTYPE mean=0.0;
		for(size_t i=0;i<n;i++) mean+=pairs[i].first;
		mean/=(RETURNTYPE)n;
		RETURNTYPE ss_res=0.0, ss_tot=0.0;
		for(size_t i=0;i<n;i++) {
			const RETURNTYPE e=pairs[i].first-pairs[i].second, d=pairs[i].first-mean;
			ss_res+=e*e; ss_tot+=d*d;
		}
		if(!(ss_tot>0.0)) return false; // constant target: R^2 undefined
		result=1.0-ss_res/ss_tot;
		return true;
	}
	case Wrapper_Metric::neg_rmse: {
		RETURNTYPE ss=0.0;
		for(size_t i=0;i<n;i++) {const RETURNTYPE e=pairs[i].first-pairs[i].second; ss+=e*e;}
		result=-std::sqrt(ss/(RETURNTYPE)n);
		return true;
	}
	case Wrapper_Metric::neg_mae: {
		RETURNTYPE sa=0.0;
		for(size_t i=0;i<n;i++) sa+=std::fabs(pairs[i].first-pairs[i].second);
		result=-sa/(RETURNTYPE)n;
		return true;
	}
	default: return false;
	}
}

} // namespace
#endif // FSTWRAPPERMETRICS_H ///:~
