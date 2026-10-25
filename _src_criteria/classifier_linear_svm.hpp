#ifndef FSTCLASSIFIERLINSVM_H
#define FSTCLASSIFIERLINSVM_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_linear_svm.hpp
   \brief   Defines Classifier_Linear_SVM — dependency-free linear SVM (dual coordinate descent) for wrapper-based feature selection
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
#include "error.hpp"
#include "global.hpp"
#include "dense_train_tools.hpp"
#include "clonable.hpp"
#include "classifier.hpp"

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

/*! \brief Dependency-free linear SVM trained by dual coordinate descent
    (the method of Hsieh et al., ICML 2008, as popularized by LIBLINEAR —
    implemented from scratch, no external code).

    L2-regularized L1-loss (hinge) or L2-loss (squared hinge) dual, solved
    coordinate-wise with the weight vector maintained incrementally. The
    coordinate order is FIXED CYCLIC — deliberately no random permutation, so
    training is fully deterministic and RNG-free: it never touches the fst_rand
    stream, and results are therefore identical for any seed and at any thread
    count. The bias enters as an appended constant coordinate (LIBLINEAR's -B 1
    convention). Features are standardized internally per split like the
    logistic-regression classifier; constant features stay inert.

    Multiclass = one-vs-rest with K linear models; two-class data trains ONE
    model (class index 0 = +1 side) and derives both scores from its single
    decision value. classify() takes the argmax of the per-class decision
    values (first maximum wins); classify_scores() exposes them for the
    ranking metrics (AUC).

    This wrapper is also the remedy for a known LibSVM weakness: LibSVM's
    LINEAR-kernel training can fail to converge (appear to freeze) on some
    data, whereas this solver has no such behaviour — and linear-SVM
    selection no longer needs LibSVM at all. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Linear_SVM : public Clonable_Impl<Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Classifier_Linear_SVM,parent> parent_impl; //!< implements clone() via the (implicit, member-wise) copy-constructor -- all members are value-semantic
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being trained/evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR
	Classifier_Linear_SVM() : _C(1.0), _l2loss(true), _max_iter(1000), _tol(0.1),
		_classes(0), _features(0), _subfeatures(0) {notify("Classifier_Linear_SVM constructor.");}
	virtual ~Classifier_Linear_SVM() {notify("Classifier_Linear_SVM destructor.");}

	void set_parameter_C(const REALTYPE c) {assert(c>0.0); _C=c;} //!< sets the SVM regularization parameter C (box constraint upper bound U under L1-loss)
	void set_l2_loss(const bool l2) {_l2loss=l2;}                //!< true = squared hinge (default), false = hinge
	void set_max_iter(const unsigned int mi) {assert(mi>0); _max_iter=mi;} //!< cap on full coordinate sweeps
	void set_tolerance(const REALTYPE t) {assert(t>0.0); _tol=t;}          //!< max |projected gradient| sweep-stopping threshold

	//! \brief Records the class/feature counts of \p da and allocates the subset index buffer.
	//! Must be called before the first train(); throws fst_error when \p da has fewer than 2 classes.
	void initialize(const PDataAccessor da);

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // argmax of per-class decision values; first maximum wins
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override; // per-class decision values for ranking metrics (AUC)
	/*! \brief Trains the linear SVM: buffers the subset index, standardizes the design with an
	    appended bias coordinate (dense_train_tools.hpp), then solves one dual coordinate-descent
	    problem per one-vs-rest model (solve_dual(), FIXED cyclic order) into the corresponding
	    row of _W. Returns false only when the split has no training data (N==0); a subset/data
	    shape mismatch is an assertion failure, not a bool return.
	    \pre initialize() has been called for a data accessor of the same shape as \p da. */
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	// test() = the canonical accuracy loop inherited from Classifier, driven by classify() above

	virtual std::ostream& print(std::ostream& os) const override
		{os << "Classifier_Linear_SVM(C=" << _C << ", loss=" << (_l2loss?"l2":"l1") << ", max_iter=" << _max_iter << ", tol=" << _tol << ")"; return os;}
protected:
	// configuration
	REALTYPE _C;                   //!< regularization parameter, see set_parameter_C()
	bool _l2loss;                  //!< true = squared hinge (default), false = hinge
	unsigned int _max_iter;        //!< cap on full coordinate sweeps in solve_dual()
	REALTYPE _tol;                 //!< max |projected gradient| sweep-stopping threshold

	// data shape + trained state
	DIMTYPE _classes;              //!< class count, set by initialize()
	DIMTYPE _features;             //!< raw (full, unnarrowed) feature count, set by initialize()
	DIMTYPE _subfeatures;         //!< d = current subset size
	std::vector<DIMTYPE> _index;  //!< subset position -> raw feature index (first d entries valid)
	std::vector<RETURNTYPE> _cs_sc; //!< classify() score scratch (member to avoid per-pattern allocation)
	std::vector<REALTYPE> _mean;  //!< standardization shift per subset position
	std::vector<REALTYPE> _scale; //!< standardization scale per subset position (1 for constant features)
	std::vector<REALTYPE> _W;     //!< models x (d+1) row-major; column d = bias; two-class data holds ONE model

	// training scratch (copied by clones harmlessly)
	std::vector<REALTYPE> _Z;     //!< N x (d+1) standardized design incl. bias coordinate
	std::vector<DIMTYPE> _y;      //!< N training labels
	std::vector<REALTYPE> _alpha; //!< N dual variables of the model currently being trained (solve_dual() scratch)
	std::vector<REALTYPE> _qbar;  //!< N diagonal entries x_i^T x_i (+ 1/(2C) for L2-loss)

	void solve_dual(REALTYPE *w, const DIMTYPE positive_class); //!< one one-vs-rest dual CD run into w[d+1]
	//! number of one-vs-rest models actually trained: 1 for two-class data, _classes otherwise.
	//! With two classes the single model is a sign test - its decision value scores class 0,
	//! and classify_scores() mirrors it as the negated score of class 1.
	DIMTYPE model_count() const {return (_classes==2)?1:_classes;}
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Linear_SVM::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_Linear_SVM requires at least 2 classes");
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::solve_dual(REALTYPE *w, const DIMTYPE positive_class)
{
	// dual coordinate descent (Hsieh et al. 2008): min 1/2 a^T Qbar a - e^T a, 0<=a_i<=U,
	// Qbar_ij = y_i y_j z_i^T z_j (+ diag 1/(2C) for L2-loss); w = sum a_i y_i z_i maintained.
	// The two losses differ exactly in the box and the diagonal shift: L1-loss (hinge) has
	// the finite box U=C and no shift, L2-loss (squared hinge) has U=infinity (the dual
	// variables are unbounded) and the shift D_ii=1/(2C) in its place.
	const std::size_t d=_subfeatures, dp1=d+1, N=_y.size();
	const REALTYPE U=_l2loss?std::numeric_limits<REALTYPE>::infinity():_C;
	const REALTYPE Dii=_l2loss?(REALTYPE)1.0/((REALTYPE)2.0*_C):(REALTYPE)0.0;

	_alpha.assign(N,0.0);
	for(std::size_t f=0;f<dp1;f++) w[f]=0.0;
	for(unsigned int sweep=0;sweep<_max_iter;sweep++)
	{
		REALTYPE pg_max=0.0;
		for(std::size_t i=0;i<N;i++) // FIXED cyclic order — deterministic by design (no RNG)
		{
			const REALTYPE *z=&_Z[i*dp1];
			const REALTYPE yi=(_y[i]==positive_class)?(REALTYPE)1.0:(REALTYPE)-1.0;
			REALTYPE g=0.0;
			for(std::size_t f=0;f<dp1;f++) g+=w[f]*z[f];
			g=yi*g-(REALTYPE)1.0+Dii*_alpha[i];

			REALTYPE pg=g; // projected gradient at the box bounds
			if(_alpha[i]<=0.0 && g>0.0) pg=0.0;
			else if(_alpha[i]>=U && g<0.0) pg=0.0;
			const REALTYPE apg=std::fabs(pg);
			if(apg>pg_max) pg_max=apg;
			if(apg>0.0)
			{
				const REALTYPE aold=_alpha[i];
				REALTYPE anew=aold-g/_qbar[i];
				if(anew<0.0) anew=0.0; else if(anew>U) anew=U;
				_alpha[i]=anew;
				const REALTYPE step=(anew-aold)*yi;
				if(step!=0.0) for(std::size_t f=0;f<dp1;f++) w[f]+=step*z[f];
			}
		}
		if(pg_max<_tol) break;
	}
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_Linear_SVM::train().");
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
	const std::size_t d=_subfeatures, dp1=d+1;

	// pass 1 + standardize via the shared helper (dense_train_tools.hpp); the appended
	// bias coordinate and the label ride along in the per-pattern callback
	_y.clear();
	const std::size_t N=standardize_design<REALTYPE,IDXTYPE>(da,_classes,d,dp1,_index,_Z,_mean,_scale,
		[this](const auto *pat, const DIMTYPE c){(void)pat; _Z.push_back(1.0); _y.push_back(c);});
	if(N==0) return false;

	// per-sample Qbar diagonal (z_i^T z_i + D_ii)
	const REALTYPE Dii=_l2loss?(REALTYPE)1.0/((REALTYPE)2.0*_C):(REALTYPE)0.0;
	_qbar.resize(N);
	for(std::size_t n=0;n<N;n++)
	{
		REALTYPE q=0.0;
		const REALTYPE *z=&_Z[n*dp1];
		for(std::size_t f=0;f<dp1;f++) q+=z[f]*z[f];
		_qbar[n]=q+Dii;
	}

	// one-vs-rest models (two-class data: the single model's +1 side is class index 0)
	const DIMTYPE M=model_count();
	_W.assign((std::size_t)M*dp1,0.0);
	for(DIMTYPE m=0;m<M;m++) solve_dual(&_W[(std::size_t)m*dp1],m);
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	assert(_subfeatures>0); // train() must have succeeded
	const std::size_t d=_subfeatures, dp1=d+1;
	scores.resize(_classes);
	const DIMTYPE M=model_count();
	for(DIMTYPE m=0;m<M;m++)
	{
		const REALTYPE *w=&_W[(std::size_t)m*dp1];
		REALTYPE dec=w[d]; // bias
		for(std::size_t f=0;f<d;f++) dec+=w[f]*(((REALTYPE)pattern[_index[f]]-_mean[f])/_scale[f]);
		scores[m]=(RETURNTYPE)dec;
	}
	if(_classes==2) scores[1]=-scores[0]; // the single binary model scores both sides
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Linear_SVM<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	if(!classify_scores(_cs_sc,pattern)) return false; // member scratch: avoids a per-pattern allocation
	cls=0;
	for(DIMTYPE c=1;c<_classes;c++) if(_cs_sc[c]>_cs_sc[cls]) cls=c; // first maximum wins
	return true;
}


} // namespace
#endif // FSTCLASSIFIERLINSVM_H ///:~
