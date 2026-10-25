#ifndef FSTCLASSIFIERRBFSVM_H
#define FSTCLASSIFIERRBFSVM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_rbf_svm.hpp
   \brief   Defines Classifier_RBF_SVM — dependency-free RBF kernel SVM (kernel dual coordinate descent) for wrapper-based feature selection
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

#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
#include <sstream>
#include "error.hpp"
#include "global.hpp"
#include "dense_train_tools.hpp"
#include "clonable.hpp"
#include "classifier.hpp"
#include "fst_rand.hpp"

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

/*! \brief Dependency-free RBF kernel SVM trained by kernel dual coordinate
    descent (implemented from scratch, no LibSVM code; C-SVC with the RBF
    kernel only, deliberately NOT a LibSVM replacement: no nu-variants, no
    SVR, no one-class, no probability estimates, no shrinking).

    The solver is the Classifier_Linear_SVM structure kernelized via the
    PENALIZED-BIAS formulation: the kernel is offset by a constant,
    K'(x,z) = exp(-gamma*||x-z||^2) + 1, the kernel analogue of the appended
    constant coordinate (LIBLINEAR's -B 1 convention) - this removes the
    equality constraint that otherwise forces two-variable SMO, so the dual
    stays a box-constrained problem solved coordinate-wise in FIXED CYCLIC
    order: fully deterministic and RNG-free, so a fixed seed and any thread
    count give identical results (the fst_rand stream is never touched). The
    margin sums s_k = sum_j alpha_j y_j K'_jk are maintained incrementally,
    so one gradient check costs O(1) and one alpha update costs O(N) given the
    kernel row. The N x N kernel matrix is precomputed when it fits a fixed
    memory budget (shared by all one-vs-rest models); above the budget,
    kernel rows are recomputed on demand - slower, still exact and
    deterministic. L1-loss (hinge; the C-SVC correspondence, default) or
    L2-loss (squared hinge) via the diagonal shift, exactly as in the linear
    solver. Features are standardized internally per split like the
    logistic-regression classifier; gamma defaults to 1/d on the
    standardized features (0 = auto). Deliberately a SIBLING of
    Classifier_Linear_SVM, not a refactor - the linear solver keeps its
    kernel-free fast path.

    Multiclass = one-vs-rest with K dual solves over the SAME kernel matrix;
    two-class data trains ONE model (class index 0 = +1 side) and derives
    both scores from its single decision value. After training the model is
    compacted to the support vectors (patterns with a nonzero dual
    coefficient in ANY model); classify() takes the argmax of the per-class
    decision values (first maximum wins) and classify_scores() exposes them
    for the ranking metrics (AUC). The bias lives inside the +1 kernel
    offset - sum_i beta_i acts as the intercept, no separate term.

    Results are NOT bit-comparable to LibSVM at matched C/gamma (different
    solver, penalized vs. exact bias, internal standardization) - the
    correspondence bar is near-identical accuracy, by design. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_RBF_SVM : public Clonable_Impl<Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Classifier_RBF_SVM,parent> parent_impl; //!< implements clone() via the (implicit, member-wise) copy-constructor -- all members are value-semantic
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being trained/evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR
	Classifier_RBF_SVM() : _C(1.0), _gamma(0.0), _l2loss(false), _max_iter(1000), _tol(0.1),
		_classes(0), _features(0), _subfeatures(0), _gamma_eff(0.0), _sv_count(0) {notify("Classifier_RBF_SVM constructor.");}
	virtual ~Classifier_RBF_SVM() {notify("Classifier_RBF_SVM destructor.");}

	void set_parameter_C(const REALTYPE c) {assert(c>0.0); _C=c;} //!< sets the SVM regularization parameter C (box constraint upper bound U under L1-loss); default 1
	void set_parameter_gamma(const REALTYPE g) {assert(g>=0.0); _gamma=g;}  //!< RBF width on standardized features; 0 = auto (1/d), which is the default
	void set_l2_loss(const bool l2) {_l2loss=l2;}                //!< false = hinge (C-SVC correspondence, default), true = squared hinge
	void set_max_iter(const unsigned int mi) {assert(mi>0); _max_iter=mi;} //!< cap on full coordinate sweeps; default 1000
	void set_tolerance(const REALTYPE t) {assert(t>0.0); _tol=t;}          //!< max |projected gradient| sweep-stopping threshold; default 0.1
	REALTYPE get_parameter_C() const {return _C;} //!< returns the configured C
	REALTYPE get_parameter_gamma() const {return _gamma;} //!< the configured value (0 = auto 1/d until set explicitly)

	void initialize(const PDataAccessor da); //!< records data shape, allocates the index buffer

	/*! C/gamma random search mirroring Classifier_LIBSVM::optimize_parameters: space-filling
	    candidate choice - each of \p max_points points is the best of \p max_throws draws,
	    maximizing its minimum log-space distance to the already-evaluated points, distances
	    weighed by those points' accuracy to repel from bad regions - evaluated by mean
	    test accuracy over the accessor's split loop; the best tested pair is set on exit.
	    The lg* bounds are NATURAL logarithms (a candidate is exp(uniform(min,max))), as in
	    the LibSVM twin. Returns false as soon as a train()/test() fails.
	    All randomness comes from fst_rand on the CALLING (master) thread, so a fixed seed
	    reproduces the search exactly; training itself stays RNG-free.
	    \note This always leaves an EXPLICIT gamma set on the object, so the 0 = auto (1/d)
	          setting does not survive a call: re-set gamma to 0 to restore it.
	    \note The trained model on return is the one from the last candidate evaluated, not
	          from the best one - call train() again before classify()/test(). */
	bool optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points=100, const int max_throws=100,
		const double lgC_min=-5, const double lgC_max=9, const double lggamma_min=-15, const double lggamma_max=3, std::ostream& os=std::cout);
	bool mean_cv_accuracy(const PDataAccessor da, const PSubset sub, RETURNTYPE &mean); //!< train+test over the accessor's split loop (optimize_parameters' evaluation)

	//! argmax of the per-class decision values; the first maximum wins. Both this and
	//! classify_scores() require a successful train() first -- with asserts disabled an
	//! untrained model has no support vectors and silently scores every class 0, i.e.
	//! classifies everything as class 0.
	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;
	//! per-class decision values (the quantity classify() maximizes) for ranking metrics such
	//! as AUC; higher means more evidence for that class. For two-class data the two entries
	//! are the single model's decision value and its negation.
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override;
	/*! \brief Trains the RBF-kernel SVM: buffers the subset index, standardizes the design
	    (dense_train_tools.hpp), precomputes the N x N kernel matrix when it fits within
	    KERNEL_BUDGET_ENTRIES (otherwise kernel_row() recomputes rows on demand), solves one
	    dual coordinate-descent problem per one-vs-rest model (solve_dual(), FIXED cyclic
	    order), then compacts the trained state to the support vectors -- patterns with a
	    nonzero dual coefficient in ANY model -- into _sv/_beta. Returns false only when the
	    split has no training data (N==0); a subset/data shape mismatch is an assertion
	    failure, not a bool return.
	    \note Cost and memory jump at N*N > KERNEL_BUDGET_ENTRIES: below it the kernel matrix
	          is built once (N*N reals) and every sweep reads it; above it each alpha update
	          recomputes a kernel row in O(N*d), which is markedly slower but bounded in memory. */
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	// test() is the accuracy loop inherited from the Classifier base

	virtual std::ostream& print(std::ostream& os) const override
	{
		os << "Classifier_RBF_SVM(C=" << _C << ", gamma=";
		if(_gamma>0.0) os << _gamma; else os << "auto(1/d)";
		os << ", loss=" << (_l2loss?"l2":"l1") << ", max_iter=" << _max_iter << ", tol=" << _tol << ")";
		return os;
	}
protected:
	// configuration
	REALTYPE _C;                  //!< regularization parameter, see set_parameter_C()
	REALTYPE _gamma;               //!< configured RBF gamma; 0 = auto: 1/d per train() (the effective value actually used is _gamma_eff)
	bool _l2loss;                  //!< false = hinge loss (default), true = squared hinge
	unsigned int _max_iter;        //!< cap on full coordinate sweeps in solve_dual()
	REALTYPE _tol;                 //!< max |projected gradient| sweep-stopping threshold

	// data shape + trained state (support vectors only — compacted after training)
	DIMTYPE _classes;              //!< class count, set by initialize()
	DIMTYPE _features;             //!< raw (full, unnarrowed) feature count, set by initialize()
	DIMTYPE _subfeatures;         //!< d = current subset size
	std::vector<DIMTYPE> _index;  //!< subset position -> raw feature index (first d entries valid)
	std::vector<REALTYPE> _mean;  //!< standardization shift per subset position
	std::vector<REALTYPE> _scale; //!< standardization scale per subset position (1 for constant features)
	REALTYPE _gamma_eff;          //!< gamma actually used by the trained model (resolves _gamma's 0=auto to 1/d)
	IDXTYPE _sv_count;            //!< support vectors kept (nonzero dual coefficient in ANY model)
	std::vector<REALTYPE> _sv;    //!< sv_count x d standardized support-vector values
	std::vector<REALTYPE> _beta;  //!< models x sv_count dual coefficients alpha_i*y_i; two-class data holds ONE model

	// training scratch (copied by clones harmlessly)
	std::vector<REALTYPE> _Z;     //!< N x d standardized design (no bias coordinate — the +1 kernel offset carries it)
	std::vector<DIMTYPE> _y;      //!< N training labels
	std::vector<REALTYPE> _alpha; //!< N dual variables of the model currently being trained (solve_dual() scratch)
	std::vector<REALTYPE> _s;     //!< N maintained margin sums s_k = sum_j alpha_j y_j K'_jk
	std::vector<REALTYPE> _K;     //!< N x N kernel matrix K' (only when within budget, else empty -- see KERNEL_BUDGET_ENTRIES)
	std::vector<REALTYPE> _krow;  //!< kernel-row scratch for the above-budget mode (filled by kernel_row())
	std::vector<REALTYPE> _cs_x;  //!< classify_scores() per-pattern scratch: the standardized query pattern (kept as a member to avoid per-pattern allocation)
	std::vector<REALTYPE> _cs_kv; //!< classify_scores() per-pattern scratch: kernel value against each support vector
	std::vector<RETURNTYPE> _cs_sc;      //!< classify() score scratch
	std::vector<REALTYPE> _beta_full; //!< models x N pre-compaction dual coefficients (solve_dual() output before the support-vector compaction in train()); kept allocated as scratch, the trained model itself is _sv/_beta

	//! kernel-matrix memory budget in entries (512 MB of double); above it rows are recomputed on demand
	static constexpr std::size_t KERNEL_BUDGET_ENTRIES=((std::size_t)1<<29)/sizeof(REALTYPE);

	//! \brief The penalized-bias RBF kernel K'(a,b) = exp(-gamma_eff*||a-b||^2) + 1 over the
	//! current _subfeatures standardized coordinates. \p a and \p b must each point at (at
	//! least) _subfeatures contiguous standardized values in subset order, e.g. rows of _Z/_sv
	//! or the classify_scores() scratch -- never raw, unstandardized pattern data.
	REALTYPE kernel_value(const REALTYPE *a, const REALTYPE *b) const
	{
		REALTYPE dist2=0.0;
		for(std::size_t f=0;f<(std::size_t)_subfeatures;f++) {const REALTYPE dd=a[f]-b[f]; dist2+=dd*dd;}
		return std::exp(-_gamma_eff*dist2)+(REALTYPE)1.0; // +1 = the penalized-bias kernel offset
	}
	//! \brief Row \p i of the N x N kernel matrix over the N training patterns in _Z.
	//! When _K was precomputed (the within-budget path) returns a pointer straight into it;
	//! otherwise recomputes the row into the _krow scratch buffer and returns that instead --
	//! in that mode the returned pointer is only valid until the next kernel_row() call, since
	//! _krow is reused. \p N must match _Z's row count (the caller's training-set size).
	const REALTYPE* kernel_row(const std::size_t i, const std::size_t N)
	{
		if(!_K.empty()) return &_K[i*N];
		const REALTYPE *zi=&_Z[i*(std::size_t)_subfeatures];
		for(std::size_t k=0;k<N;k++) _krow[k]=kernel_value(zi,&_Z[k*(std::size_t)_subfeatures]);
		return _krow.data();
	}
	//! \brief One one-vs-rest kernel dual coordinate-descent run: patterns of class
	//! \p positive_class are the +1 side, all others the -1 side. Writes the N dual
	//! coefficients alpha_i*y_i into \p beta, which must have room for N = _y.size()
	//! entries. Requires _Z/_y/_gamma_eff and the kernel state (_K or _krow) to be set up
	//! by train(); uses _alpha and _s as scratch. Stops when the largest projected gradient
	//! of a sweep falls below _tol, or after _max_iter sweeps (no convergence failure is
	//! reported -- an unconverged model is simply less accurate).
	void solve_dual(REALTYPE *beta, const DIMTYPE positive_class);
	DIMTYPE model_count() const {return (_classes==2)?1:_classes;} //!< number of one-vs-rest models actually trained: 1 for two-class data (the single model scores both sides), _classes otherwise
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_RBF_SVM::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_RBF_SVM requires at least 2 classes");
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::mean_cv_accuracy(const PDataAccessor da, const PSubset sub, RETURNTYPE &mean)
{
	RETURNTYPE val, result=0.0, cnt=0.0;
	for(bool b=da->getFirstSplit();b==true;b=da->getNextSplit()) {
		if(!train(da,sub)) return false;
		if(!this->test(val,da)) return false; // this-> because test() lives in the dependent Classifier base
		result+=val;
		cnt+=1.0;
	}
	mean=result/(RETURNTYPE)cnt;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points, const int max_throws, const double lgC_min, const double lgC_max, const double lggamma_min, const double lggamma_max, std::ostream& os)
{
	// the Classifier_LIBSVM::optimize_parameters protocol, C/gamma only (no coef0 axis)
	assert(da);
	assert(sub);
	assert(sub->get_d_raw()>0);
	assert(max_points>0);
	assert(max_throws>0);
	assert(lgC_max>lgC_min);
	assert(lggamma_max>lggamma_min);
	const double lgCrange=lgC_max-lgC_min;
	const double lggammarange=lggamma_max-lggamma_min;
	struct PSet {RETURNTYPE val; double Cpar, gamma;};
	std::vector<PSet> paramlist;
	// evaluate the initial (current) parameter set; gamma 0 = auto resolves to 1/d on the first train()
	RETURNTYPE val, maxval, firstval;
	if(!mean_cv_accuracy(da,sub,val)) return false;
	firstval=maxval=val;
	{
		// NOTE: this line reports _gamma_eff (auto resolved to 1/d by the train() just done);
		//       the optimizer sets _gamma explicitly from here on, so the per-candidate and
		//       "after" lines below print concrete, directly comparable values as well
		std::ostringstream sos;
		sos << "RBF SVM before optimization: accuracy=" << val << ", C=" << _C << ", gamma=" << _gamma_eff << std::endl;
		syncout::print(os,sos);
	}
	paramlist.push_back(PSet{val,(double)_C,(double)_gamma_eff});
	PSet tmpset{0.0,0.0,0.0}, bestset{0.0,0.0,0.0};
	for(int i=0;i<max_points;i++)
	{ // find and evaluate next parameter set candidate (space-filling choice, as in the LibSVM twin)
		double maxdistmin=0.0;
		for(int j=0;j<max_throws;j++)
		{
			// the bounds are natural logarithms: a candidate is exp(uniform(min,max))
			tmpset.Cpar =std::exp(lgC_min+(((double)fst_rand()*(double)(lgC_max-lgC_min))/(double)FST_RAND_MAX));
			tmpset.gamma=std::exp(lggamma_min+(((double)fst_rand()*(double)(lggamma_max-lggamma_min))/(double)FST_RAND_MAX));
			double distmin=-1.0; // candidate (weighted) distance to already evaluated candidates
			for(const PSet &ps : paramlist)
			{
				const double dC=(std::log(tmpset.Cpar)-std::log(ps.Cpar))/lgCrange;
				const double dG=(std::log(tmpset.gamma)-std::log(ps.gamma))/lggammarange;
				const double dist=std::sqrt(dC*dC+dG*dG);
				// candidates more distant from all others are preferred; distances are
				// weighed to penalize closeness to worse-performing candidates
				assert(ps.val>0.0);
				if(distmin<0 || dist*ps.val<distmin) distmin=dist*ps.val;
			}
			assert(distmin>=0.0);
			if(distmin>maxdistmin) {maxdistmin=distmin; bestset=tmpset;}
		}
		// evaluate the chosen candidate and add it to the list
		set_parameter_C((REALTYPE)bestset.Cpar);
		set_parameter_gamma((REALTYPE)bestset.gamma);
		if(!mean_cv_accuracy(da,sub,val)) return false;
		bestset.val=val;
		paramlist.push_back(bestset);
		{
			std::ostringstream sos;
			sos << i+1 << ". ";
			if(val>maxval) {maxval=val; sos << "BEST accuracy=";} else sos << "test accuracy=";
			sos << bestset.val << ", C=" << _C << ", gamma=" << _gamma << std::endl;
			syncout::print(os,sos);
		}
	}
	// identify the best parameter set among all tested ones
	assert(!paramlist.empty());
	bestset=paramlist.front();
	for(const PSet &ps : paramlist) if(ps.val>bestset.val) bestset=ps;
	set_parameter_C((REALTYPE)bestset.Cpar);
	set_parameter_gamma((REALTYPE)bestset.gamma);
	{
		std::ostringstream sos;
		sos << "RBF SVM after optimization: accuracy=" << bestset.val << ", C=" << _C << ", gamma=" << _gamma << std::endl;
		sos << "Accuracy difference=" << bestset.val-firstval << std::endl << std::endl;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::solve_dual(REALTYPE *beta, const DIMTYPE positive_class)
{
	// kernel dual coordinate descent: min 1/2 a^T Qbar a - e^T a, 0<=a_i<=U,
	// Qbar_ij = y_i y_j K'_ij (+ diag 1/(2C) for L2-loss); the margin sums _s are
	// maintained so the gradient check is O(1); an alpha update touches one kernel row
	const std::size_t N=_y.size();
	const REALTYPE U=_l2loss?std::numeric_limits<REALTYPE>::infinity():_C;
	const REALTYPE Dii=_l2loss?(REALTYPE)1.0/((REALTYPE)2.0*_C):(REALTYPE)0.0;
	const REALTYPE Qii=(REALTYPE)2.0+Dii; // K'(x,x) = exp(0)+1 = 2 on the diagonal, always

	_alpha.assign(N,0.0);
	_s.assign(N,0.0);
	for(unsigned int sweep=0;sweep<_max_iter;sweep++)
	{
		REALTYPE pg_max=0.0;
		for(std::size_t i=0;i<N;i++) // FIXED cyclic order - deterministic by design (no RNG)
		{
			const REALTYPE yi=(_y[i]==positive_class)?(REALTYPE)1.0:(REALTYPE)-1.0;
			const REALTYPE g=yi*_s[i]-(REALTYPE)1.0+Dii*_alpha[i];

			REALTYPE pg=g; // projected gradient at the box bounds
			if(_alpha[i]<=0.0 && g>0.0) pg=0.0;
			else if(_alpha[i]>=U && g<0.0) pg=0.0;
			const REALTYPE apg=std::fabs(pg);
			if(apg>pg_max) pg_max=apg;
			if(apg>0.0)
			{
				const REALTYPE aold=_alpha[i];
				REALTYPE anew=aold-g/Qii;
				if(anew<0.0) anew=0.0; else if(anew>U) anew=U;
				const REALTYPE step=(anew-aold)*yi;
				_alpha[i]=anew;
				if(step!=0.0)
				{
					const REALTYPE *row=kernel_row(i,N);
					for(std::size_t k=0;k<N;k++) _s[k]+=step*row[k];
				}
			}
		}
		if(pg_max<_tol) break;
	}
	for(std::size_t i=0;i<N;i++) beta[i]=_alpha[i]*((_y[i]==positive_class)?(REALTYPE)1.0:(REALTYPE)-1.0);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_RBF_SVM::train().");
	assert(da);
	assert(sub);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	// buffer the feature subset index
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<_features); _index[_subfeatures++]=fsel;}
	assert(_subfeatures==sub->get_d_raw());
	assert(_subfeatures>0);
	const std::size_t d=_subfeatures;

	// pass 1 + standardize via the shared helper (dense_train_tools.hpp); no bias
	// coordinate here (the penalized-bias kernel offset supplies it), label rides along
	_y.clear();
	const std::size_t N=standardize_design<REALTYPE,IDXTYPE>(da,_classes,d,d,_index,_Z,_mean,_scale,
		[this](const auto *pat, const DIMTYPE c){(void)pat; _y.push_back(c);});
	if(N==0) return false;

	_gamma_eff=(_gamma>0.0)?_gamma:(REALTYPE)1.0/(REALTYPE)d;

	// kernel matrix, shared by all one-vs-rest models; above budget rows are recomputed on demand
	if(N*N<=KERNEL_BUDGET_ENTRIES)
	{
		_K.resize(N*N);
		for(std::size_t a=0;a<N;a++)
		{
			_K[a*N+a]=(REALTYPE)2.0;
			for(std::size_t b=a+1;b<N;b++) _K[a*N+b]=_K[b*N+a]=kernel_value(&_Z[a*d],&_Z[b*d]);
		}
		_krow.clear();
	}
	else {_K.clear(); _krow.resize(N);}

	// one-vs-rest models (two-class data: the single model's +1 side is class index 0)
	const DIMTYPE M=model_count();
	_beta_full.assign((std::size_t)M*N,0.0);
	for(DIMTYPE m=0;m<M;m++) solve_dual(&_beta_full[(std::size_t)m*N],m);

	// compact to support vectors: keep samples with a nonzero coefficient in ANY model
	_sv_count=0;
	_sv.clear(); _beta.clear();
	std::vector<std::size_t> keep;
	for(std::size_t n=0;n<N;n++)
	{
		bool used=false;
		for(DIMTYPE m=0;m<M && !used;m++) if(_beta_full[(std::size_t)m*N+n]!=0.0) used=true;
		if(used) keep.push_back(n);
	}
	_sv_count=(IDXTYPE)keep.size();
	_sv.resize(keep.size()*d);
	_beta.resize((std::size_t)M*keep.size());
	for(std::size_t k=0;k<keep.size();k++)
	{
		for(std::size_t f=0;f<d;f++) _sv[k*d+f]=_Z[keep[k]*d+f];
		for(DIMTYPE m=0;m<M;m++) _beta[(std::size_t)m*keep.size()+k]=_beta_full[(std::size_t)m*N+keep[k]];
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	assert(_subfeatures>0); // train() must have succeeded
	const std::size_t d=_subfeatures, nsv=_sv_count;
	scores.resize(_classes);
	// standardize the pattern, then evaluate the kernel against each support vector ONCE (shared by all models)
	_cs_x.resize(d); _cs_kv.resize(nsv);
	for(std::size_t f=0;f<d;f++) _cs_x[f]=((REALTYPE)pattern[_index[f]]-_mean[f])/_scale[f];
	for(std::size_t k=0;k<nsv;k++) _cs_kv[k]=kernel_value(&_sv[k*d],_cs_x.data());
	const DIMTYPE M=model_count();
	for(DIMTYPE m=0;m<M;m++)
	{
		const REALTYPE *b=&_beta[(std::size_t)m*nsv];
		REALTYPE dec=0.0; // the intercept lives inside the +1 kernel offset
		for(std::size_t k=0;k<nsv;k++) dec+=b[k]*_cs_kv[k];
		scores[m]=(RETURNTYPE)dec;
	}
	if(_classes==2) scores[1]=-scores[0]; // the single binary model scores both sides
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_RBF_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	if(!classify_scores(_cs_sc,pattern)) return false;
	cls=0;
	for(DIMTYPE c=1;c<_classes;c++) if(_cs_sc[c]>_cs_sc[cls]) cls=c; // first maximum wins
	return true;
}


} // namespace
#endif // FSTCLASSIFIERRBFSVM_H ///:~
