#ifndef FSTCLASSIFIERLOGREG_H
#define FSTCLASSIFIERLOGREG_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_logreg.hpp
   \brief   Implements multinomial logistic regression classifier (dependency-free, deterministic; regularized damped-Newton training on internally standardized data)
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
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "classifier.hpp"
#include "dense_train_tools.hpp"

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

/*! \brief Implements multinomial (softmax) logistic regression classifier

 One weight matrix W of shape classes x (d+1) (bias per class) trained by
 minimizing mean cross-entropy + (lambda/2)*||weights||^2 (biases unpenalized)
 with damped Newton iterations; the Newton system is solved by an internal
 Cholesky factorization, so no external linear-algebra dependency is needed.

 \note train() standardizes the selected features internally (mean/std computed
       on the current split's TRAIN part); classify() applies the stored
       transform. Constant features are inert (their weight stays exactly 0),
       so degenerate data that breaks the normal-model filters is safe here.
 \note Training is deterministic: no random number generator is consumed,
       weights initialize to zero, and data is traversed in fixed order -
       repeated runs give identical results, whatever the seed or the thread count.
 \note Convergence contract: iterate until the gradient infinity-norm drops
       below tolerance or max_iter is reached; step halving guarantees
       monotone descent on this convex objective. Defaults: lambda=0.01,
       max_iter=100, tolerance=1e-6.
 \note Cost: each iteration assembles a K(d+1) x K(d+1) Hessian in O(N*(K(d+1))^2)
       and solves it in O((K(d+1))^3) - comparable to a kNN wrapper evaluation
       for typical wrapper-FS subset sizes; many classes combined with large
       subsets is the expensive corner (guarded, see train()).
*/
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Logistic_Regression : public Clonable_Impl<Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Clonable_Impl<Classifier_Logistic_Regression,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > parent_impl; //!< implements clone() via the (implicit, member-wise) copy-constructor - all members are value-semantic
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> const PSubset;            //!< shared pointer to the (fixed, non-reassignable) feature subset being trained/classified on
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_Logistic_Regression() : _lambda(0.01), _max_iter(100), _tol(1e-6), _classes(0), _features(0), _subfeatures(0)
		{notify("Classifier_Logistic_Regression constructor.");}
	virtual ~Classifier_Logistic_Regression() {notify("Classifier_Logistic_Regression destructor.");}

	void set_lambda(const REALTYPE lambda) {if(lambda<0.0) throw fst_error("Classifier_Logistic_Regression: lambda must be >=0"); _lambda=lambda;} //!< sets the L2 weight penalty (must be >=0); default 0.01
	void set_max_iter(const unsigned int max_iter) {if(max_iter<1) throw fst_error("Classifier_Logistic_Regression: max_iter must be >=1"); _max_iter=max_iter;} //!< sets the damped-Newton iteration cap (must be >=1); default 100
	void set_tolerance(const REALTYPE tol) {if(!(tol>0.0)) throw fst_error("Classifier_Logistic_Regression: tolerance must be >0"); _tol=tol;} //!< sets the gradient infinity-norm stopping tolerance (must be >0); default 1e-6
	REALTYPE get_lambda() const {return _lambda;} //!< current L2 weight penalty
	unsigned int get_max_iter() const {return _max_iter;} //!< current Newton iteration cap
	REALTYPE get_tolerance() const {return _tol;} //!< current gradient infinity-norm stopping tolerance

	//! \brief Records data shape (classes, raw feature count) and resets to "no model yet".
	//! Must be called before train() or test() (and whenever the training set's shape changes).
	void initialize(const PDataAccessor da);

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // classifies pattern, returns the respective class index
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override; // per-class linear scores (log-odds up to a shared shift) for ranking metrics (AUC)
	virtual bool train(const PDataAccessor da, const PSubset sub) override; // learns from designated training part of data
	// test() is the accuracy loop inherited from the Classifier base

	DIMTYPE get_n() const {return _features;} //!< raw (full) feature count, as recorded by initialize()
	DIMTYPE get_d() const {return _subfeatures;} //!< current subset dimensionality, as recorded by the last train()

	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_Logistic_Regression(lambda=" << _lambda << ", max_iter=" << _max_iter << ", tol=" << _tol << ")"; return os;}
protected:
	// knobs
	REALTYPE _lambda;        //!< L2 weight penalty (biases unpenalized)
	unsigned int _max_iter;  //!< Newton iteration cap
	REALTYPE _tol;           //!< gradient infinity-norm stopping tolerance

	// data shape (set in initialize())
	DIMTYPE _classes;             //!< number of classes, as recorded by initialize()
	DIMTYPE _features;            //!< raw (full) feature count, as recorded by initialize()

	// learned model (valid after train())
	DIMTYPE _subfeatures;         //!< d = current subset size
	std::vector<DIMTYPE> _index;  //!< subset position -> raw feature index (first d entries valid)
	std::vector<REALTYPE> _mean;  //!< standardization shift per subset position
	std::vector<REALTYPE> _scale; //!< standardization scale per subset position (1 for constant features)
	std::vector<REALTYPE> _W;     //!< classes x (d+1) row-major; column d = bias

	// scratch buffers - contents transient, capacity persists across train() calls
	std::vector<REALTYPE> _Z;     //!< N x (d+1) standardized training data, last column fixed 1 (bias coordinate)
	std::vector<DIMTYPE> _y;      //!< N training labels
	std::vector<REALTYPE> _P;     //!< N x classes posterior cache for Hessian assembly
	std::vector<REALTYPE> _H;     //!< Newton system, m x m, m=classes*(d+1) (lower triangle used)
	std::vector<REALTYPE> _grad;  //!< objective gradient at the current Newton iterate (see eval_objective())
	std::vector<REALTYPE> _step;  //!< Newton step solved from the Hessian system (see assemble_hessian())
	std::vector<REALTYPE> _Wtry;  //!< trial weights during step-halving line search in train()
	std::vector<REALTYPE> _sc;    //!< per-class softmax score scratch used by eval_objective() (sized max(classes,d+1))

private:
	//! mean cross-entropy + L2 at weights W; with \p want_grad it ALSO refreshes _grad and the
	//! _P posterior cache at W (so the objective value alone can be obtained cheaply, without
	//! disturbing either). Requires _Z/_y filled by train().
	REALTYPE eval_objective(const std::vector<REALTYPE> &W, const bool want_grad);
	//! fills _H (lower triangle) with (1/N) Z^T S Z + regularization diagonal + extra_ridge.
	//! \note It reads the _P posterior cache rather than any W of its own, so it describes the
	//!       point of the LAST eval_objective(...,true) call - never call it after evaluating
	//!       the objective at a trial point.
	void assemble_hessian(const std::size_t m, const REALTYPE extra_ridge);
	// Cholesky factor/solve of the Newton system live in dense_train_tools.hpp (shared with regressor_ridge)
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Logistic_Regression::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_Logistic_Regression requires at least 2 classes");
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
REALTYPE Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::eval_objective(const std::vector<REALTYPE> &W, const bool want_grad)
{
	const std::size_t d=_subfeatures, dp1=d+1, K=_classes, N=_y.size();
	assert(N>0);
	if(want_grad) {_grad.assign(K*dp1,0.0); _P.resize(N*K);}
	REALTYPE J=0.0;
	for(std::size_t i=0;i<N;i++)
	{
		const REALTYPE *z=&_Z[i*dp1];
		REALTYPE smax; // assigned on the c==0 pass below; K>=2 is guaranteed by initialize()
		for(std::size_t c=0;c<K;c++) {
			REALTYPE s=0.0;
			const REALTYPE *w=&W[c*dp1];
			for(std::size_t f=0;f<dp1;f++) s+=w[f]*z[f];
			_sc[c]=s;
			if(c==0 || s>smax) smax=s;
		}
		REALTYPE sumexp=0.0;
		for(std::size_t c=0;c<K;c++) {_sc[c]=std::exp(_sc[c]-smax); sumexp+=_sc[c];}
		J-=std::log(_sc[_y[i]]/sumexp);
		if(want_grad)
			for(std::size_t c=0;c<K;c++) {
				const REALTYPE p=_sc[c]/sumexp;
				_P[i*K+c]=p;
				const REALTYPE r=p-(c==(std::size_t)_y[i]?1.0:0.0);
				REALTYPE *g=&_grad[c*dp1];
				for(std::size_t f=0;f<dp1;f++) g[f]+=r*z[f];
			}
	}
	J/=(REALTYPE)N;
	REALTYPE penalty=0.0;
	for(std::size_t c=0;c<K;c++)
		for(std::size_t f=0;f<d;f++) penalty+=W[c*dp1+f]*W[c*dp1+f]; // biases (f==d) unpenalized
	J+=0.5*_lambda*penalty;
	if(want_grad)
		for(std::size_t c=0;c<K;c++)
			for(std::size_t f=0;f<dp1;f++) {
				_grad[c*dp1+f]/=(REALTYPE)N;
				if(f<d) _grad[c*dp1+f]+=_lambda*W[c*dp1+f];
			}
	return J;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::assemble_hessian(const std::size_t m, const REALTYPE extra_ridge)
{
	const std::size_t d=_subfeatures, dp1=d+1, K=_classes, N=_y.size();
	_H.assign(m*m,0.0);
	for(std::size_t n=0;n<N;n++)
	{
		const REALTYPE *z=&_Z[n*dp1];
		const REALTYPE *pr=&_P[n*K];
		for(std::size_t c1=0;c1<K;c1++)
			for(std::size_t c2=0;c2<=c1;c2++)
			{
				const REALTYPE coef=(c1==c2)?pr[c1]*(1.0-pr[c1]):-pr[c1]*pr[c2];
				if(coef==0.0) continue;
				REALTYPE *hblk=&_H[(c1*dp1)*m+c2*dp1];
				for(std::size_t f1=0;f1<dp1;f1++)
				{
					const REALTYPE zc=coef*z[f1];
					const std::size_t fend=(c1==c2)?f1+1:dp1; // diagonal block: lower triangle only
					for(std::size_t f2=0;f2<fend;f2++) hblk[f1*m+f2]+=zc*z[f2];
				}
			}
	}
	for(std::size_t j=0;j<m*m;j++) _H[j]/=(REALTYPE)N; // only the lower triangle is meaningful, scaling all is harmless
	for(std::size_t c=0;c<K;c++)
		for(std::size_t f=0;f<dp1;f++)
		{
			const std::size_t j=c*dp1+f;
			// weights get the L2 term; biases a tiny fixed ridge (softmax is
			// shift-invariant in the biases, the gradient has no component in
			// that null direction - the ridge only keeps Cholesky defined)
			_H[j*m+j]+=((f<d)?_lambda:(REALTYPE)1e-8)+extra_ridge;
		}
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_Logistic_Regression::train().");
	assert(da);
	assert(sub);
	assert(sub->get_frozen_mode()==false);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	// buffer the feature subset index
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<_features); _index[_subfeatures++]=fsel;}
	assert(_subfeatures==sub->get_d_raw());
	assert(_subfeatures>0);

	const std::size_t d=_subfeatures, dp1=d+1, K=_classes, m=K*dp1;
	// Hessian memory guard: m=8192 means a 512MB system - beyond it an OOM crash
	// is likelier than a useful result, so fail with advice instead
	if(m>8192) throw fst_error("Classifier_Logistic_Regression: Newton system too large (classes*(subset size+1) > 8192); use smaller subsets (e.g. a forward search bounded to a small target subset size)");

	// pass 1 + standardize via the shared helper (dense_train_tools.hpp); the appended
	// bias coordinate and the label ride along in the per-pattern callback
	_y.clear();
	const std::size_t N=standardize_design<REALTYPE,IDXTYPE>(da,_classes,d,dp1,_index,_Z,_mean,_scale,
		[this](const auto *pat, const DIMTYPE c){(void)pat; _Z.push_back(1.0); _y.push_back(c);});
	if(N==0) return false;

	// damped Newton from W=0
	_W.assign(m,0.0);
	_sc.resize(K>dp1?K:dp1);
	_step.resize(m); _Wtry.resize(m);
	REALTYPE J=eval_objective(_W,true);
	for(unsigned int iter=0;iter<_max_iter;iter++)
	{
		REALTYPE gmax=0.0;
		for(std::size_t j=0;j<m;j++) {const REALTYPE a=std::fabs(_grad[j]); if(a>gmax) gmax=a;}
		if(gmax<_tol) break;

		// assemble and factor the Newton system, escalating the ridge on numerical failure
		// (in-place factoring destroys _H, so each retry reassembles). A large ridge makes the
		// Hessian approach a scaled identity, i.e. the "Newton" step degrades gracefully into a
		// short gradient step - still a descent direction, which is all the line search needs.
		REALTYPE ridge=0.0;
		assemble_hessian(m,ridge);
		bool factored=cholesky_factor(_H,m);
		while(!factored && ridge<1.0)
		{
			ridge=(ridge==0.0)?(REALTYPE)1e-6:ridge*100.0;
			assemble_hessian(m,ridge);
			factored=cholesky_factor(_H,m);
		}
		if(!factored) return false; // pathological even with heavy ridge

		cholesky_solve(_H,m,_grad.data(),true,_step.data()); // L L^T step = -grad

		// step halving: guarantee monotone descent. The 60 halvings are a hard floor rather than
		// a tuning knob - at t=2^-60 the trial weights differ from the current ones by less than
		// double precision can represent, so no further halving could ever improve J.
		REALTYPE t=1.0;
		bool improved=false;
		for(int half=0;half<60;half++)
		{
			for(std::size_t j=0;j<m;j++) _Wtry[j]=_W[j]+t*_step[j];
			const REALTYPE Jtry=eval_objective(_Wtry,false);
			if(Jtry<J && std::isfinite(Jtry)) {improved=true; _W.swap(_Wtry); J=Jtry; break;}
			t*=0.5;
		}
		if(!improved) break; // FP floor - treat as converged
		J=eval_objective(_W,true); // refresh gradient and posterior cache at the accepted point
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	notify("Classifier_Logistic_Regression::classify().");
	assert(_subfeatures>0); // train() must have succeeded
	const std::size_t d=_subfeatures, dp1=d+1, K=_classes;
	assert(_W.size()==K*dp1);
	// argmax of the linear scores == argmax of softmax posteriors (no exp needed)
	REALTYPE best;
	for(std::size_t c=0;c<K;c++)
	{
		const REALTYPE *w=&_W[c*dp1];
		REALTYPE score=w[d]; // bias
		for(std::size_t f=0;f<d;f++) score+=w[f]*(((REALTYPE)pattern[_index[f]]-_mean[f])/_scale[f]);
		if(c==0 || score>best) {best=score; cls=(DIMTYPE)c;} // first maximum wins
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Logistic_Regression<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	// the classify() linear scores, exposed per class (monotone in the softmax posteriors)
	notify("Classifier_Logistic_Regression::classify_scores().");
	assert(_subfeatures>0); // train() must have succeeded
	const std::size_t d=_subfeatures, dp1=d+1, K=_classes;
	assert(_W.size()==K*dp1);
	scores.resize(K);
	for(std::size_t c=0;c<K;c++)
	{
		const REALTYPE *w=&_W[c*dp1];
		REALTYPE score=w[d]; // bias
		for(std::size_t f=0;f<d;f++) score+=w[f]*(((REALTYPE)pattern[_index[f]]-_mean[f])/_scale[f]);
		scores[c]=(RETURNTYPE)score;
	}
	return true;
}


} // namespace
#endif // FSTCLASSIFIERLOGREG_H ///:~
