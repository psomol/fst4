#ifndef FSTCLASSIFIERRF_H
#define FSTCLASSIFIERRF_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_rf.hpp
   \brief   Defines Classifier_Random_Forest — dependency-free, deterministic-by-design Random Forest for wrapper-based feature selection
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
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

/*! \brief Dependency-free Random Forest classifier (CART trees, Gini impurity,
    bagging, per-node feature subsampling), deterministic by design.

    Randomness design: RF training draws randomness inside worker-side criterion
    clones, which must not touch the thread_local fst_rand() stream (the
    reproducibility of the threaded evaluators depends on the master owning it).
    Instead the training RNG is a LOCAL xorshift64* generator seeded by an FNV-1a
    hash of the candidate subset content (the ascending selected feature indices),
    salted with the process seed and the data accessor's current split index. Every
    evaluation is therefore a pure function of (subset, data, seed):
    threading-invariant and evaluation-order-invariant by construction, at any
    thread count, under any evaluator.

    Determinism details: bootstrap draws and per-node feature subsampling come
    from the hash-seeded stream in fixed order; candidate-value sorting uses a
    total order (value, then sample index); equal-gain splits keep the first
    candidate encountered; vote ties resolve to the lowest class index. No
    criterion estimation uses out-of-bag machinery - subset quality comes from
    the FST4 splitters (CV/holdout) as with every other wrapper.

    Trees are per-train() state; the wrapper retrains per candidate subset.
    classify_scores() returns per-class tree-vote fractions, which is what makes
    ranking metrics such as AUC available for this classifier. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Random_Forest : public Clonable_Impl<Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the abstract base type
	typedef Clonable_Impl<Classifier_Random_Forest,parent> parent_impl; //!< implements clone() via the (implicit, member-wise) copy-constructor - all members are value-semantic
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being trained/classified on
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR (a raw pointer to its feature vector)
	Classifier_Random_Forest() : _tree_count(100), _mtry(0), _max_depth(0), _min_leaf(1), _seed_salt(1),
		_classes(0), _features(0), _subfeatures(0) {notify("Classifier_Random_Forest constructor.");}
	virtual ~Classifier_Random_Forest() {notify("Classifier_Random_Forest destructor.");}

	void set_tree_count(const unsigned int t) {assert(t>0); _tree_count=t;} //!< number of trees to grow (must be >0); default 100
	void set_mtry(const DIMTYPE m) {_mtry=m;}                    //!< features tried per node; 0 = floor(sqrt(subset size))
	void set_max_depth(const unsigned int dp) {_max_depth=dp;}   //!< 0 = unlimited
	void set_min_leaf(const IDXTYPE ml) {assert(ml>0); _min_leaf=ml;} //!< minimum samples in each child after a split (must be >0); default 1
	void set_seed_salt(const unsigned int s) {_seed_salt=s;}     //!< salts the subset-content hash the training RNG is seeded from; default 1 (the same value the unseeded library PRNG uses), so leaving it alone still gives a fixed, reproducible forest

	void initialize(const PDataAccessor da); //!< records data shape, allocates the index buffer

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;  // majority tree vote; ties resolve to the lowest class index
	virtual bool classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern) override; // per-class tree-vote fractions for ranking metrics (AUC)
	/*! \brief Grows the forest on the designated training part of \p da, restricted to \p sub's
	    selected features.
	    Buffers the raw (unstandardized -- CART is scale-invariant) selected feature values and
	    labels, seeds the local xorshift64* RNG (see class doc) from an FNV-1a hash of (subset
	    content, `--seed` salt, current split index), derives mtry (default floor(sqrt(subset
	    size))), then grows _tree_count trees from independent bootstrap resamples (rng_below(N)
	    draws with replacement, fixed order).
	    \return false if the designated training part is empty; true otherwise. */
	virtual bool train(const PDataAccessor da, const PSubset sub) override; // grows the forest on the designated training part
	// test() is the accuracy loop inherited from the Classifier base

	virtual std::ostream& print(std::ostream& os) const override
		{os << "Classifier_Random_Forest(trees=" << _tree_count << ", mtry=" << _mtry << ", max_depth=" << _max_depth << ", min_leaf=" << _min_leaf << ")"; return os;}
protected:
	//! one CART node; left<0 marks a leaf (leaf_class valid), otherwise feature/threshold/left/right.
	//! A tree is a flat vector of these: the root is always index 0, and a split appends its two
	//! children at the end of the vector, so child indices always exceed the parent's.
	struct Node {
		DIMTYPE feature;    //!< split feature, given as a subset position (see _index for the raw feature index)
		REALTYPE threshold; //!< split threshold; a pattern routes to \c left when its value is <=threshold
		long left;          //!< left-child index into the tree's node vector; <0 marks this node a leaf
		long right;         //!< right-child index into the tree's node vector (meaningful only when left>=0)
		DIMTYPE leaf_class; //!< predicted class (meaningful only when left<0, i.e. at a leaf)
	};

	// configuration
	unsigned int _tree_count; //!< number of trees to grow; set by set_tree_count()
	DIMTYPE _mtry;             //!< features tried per split node; 0 = auto (floor(sqrt(subset size))); set by set_mtry()
	unsigned int _max_depth;   //!< tree depth cap; 0 = unlimited; set by set_max_depth()
	IDXTYPE _min_leaf;         //!< minimum samples in each child after a split; set by set_min_leaf()
	unsigned int _seed_salt;   //!< salts the per-evaluation subset-content hash; set by set_seed_salt(), default 1

	// data shape + trained state
	DIMTYPE _classes;                   //!< number of classes, as recorded by initialize()
	DIMTYPE _features;                  //!< raw (full) feature count, as recorded by initialize()
	DIMTYPE _subfeatures;               //!< d = current subset size
	std::vector<DIMTYPE> _index;        //!< subset position -> raw feature index (first d entries valid)
	std::vector<IDXTYPE> _votes;        //!< classify() vote scratch (member to avoid per-pattern allocation)
	std::vector<std::vector<Node> > _forest; //!< the trained forest; outer index = tree, inner = that tree's node vector (root at index 0)

	// training scratch (rebuilt per train(); copied by clones harmlessly)
	std::vector<REALTYPE> _X;           //!< N x d row-major selected training values
	std::vector<DIMTYPE> _y;            //!< N training labels

	// hash-seeded local RNG (xorshift64*) - deliberately NOT the thread_local fst_rand stream
	std::uint64_t _rng_state; //!< xorshift64* generator state; seeded per train() call (see train())
	std::uint64_t rng_next() {std::uint64_t &s=_rng_state; s^=s>>12; s^=s<<25; s^=s>>27; return s*2685821657736338717ULL;} //!< advances the xorshift64* generator, returns its next 64-bit output
	IDXTYPE rng_below(const IDXTYPE n) {assert(n>0); return (IDXTYPE)(rng_next()%(std::uint64_t)n);} //!< a value in [0,n) via modulo of rng_next(). The modulo bias is of order n/2^64 - unmeasurable for any sample or feature count - and only determinism matters here, not statistical perfection

	//! \brief Grows one CART tree (Gini-impurity best split, \p mtry random candidate features per
	//! node) into \p tree, over the bootstrap sample indices in \p samples (indices into _X/_y).
	//! Splits are found by a total order on (feature value, sample index) so ties are resolved
	//! deterministically; equal-gain candidates keep the first one encountered; a node becomes a
	//! leaf when pure, when either child would fall below _min_leaf, or at _max_depth; the leaf
	//! class is the majority vote with lowest-class-index tie-break.
	void grow_tree(std::vector<Node> &tree, std::vector<IDXTYPE> &samples, const DIMTYPE mtry);
	//! \brief Walks \p tree from its root to a leaf for \p pattern (via _index to map each node's
	//! subset-position feature to the pattern's raw feature index) and returns the leaf's class.
	DIMTYPE tree_vote(const std::vector<Node> &tree, const PPattern &pattern) const;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Random_Forest::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_Random_Forest requires at least 2 classes");
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
	_forest.clear();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_Random_Forest::train().");
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

	// gather raw selected training values and labels (CART is scale-invariant — no standardization)
	const DIMTYPE stride=da->getNoOfFeatures();
	_X.clear(); _y.clear();
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	for(DIMTYPE c=0;c<_classes;c++)
	{
		da->setClass(c);
		for(const auto &blk : da->blocks(TRAIN)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
		{
			const auto *pat=&p[(std::size_t)i*stride];
			for(std::size_t f=0;f<d;f++) _X.push_back((REALTYPE)pat[_index[f]]);
			_y.push_back(c);
		}
	}
	const IDXTYPE N=(IDXTYPE)_y.size();
	if(N==0) return false;

	// seed the local RNG from (subset content, `--seed` salt, split index): FNV-1a 64-bit
	std::uint64_t h=14695981039346656037ULL;
	const std::uint64_t prime=1099511628211ULL;
	auto mix=[&h](std::uint64_t v) {for(int b=0;b<8;b++) {h^=(v&0xffu); h*=prime; v>>=8;}};
	mix((std::uint64_t)_seed_salt);
	mix((std::uint64_t)da->getSplitIndex());
	for(std::size_t f=0;f<d;f++) mix((std::uint64_t)_index[f]);
	if(h==0) h=0x9E3779B97F4A7C15ULL; // xorshift state must be nonzero
	_rng_state=h;

	// per-node feature-subsample size
	DIMTYPE mtry=_mtry;
	if(mtry==0) mtry=(DIMTYPE)std::floor(std::sqrt((double)d));
	if(mtry<1) mtry=1;
	if(mtry>(DIMTYPE)d) mtry=(DIMTYPE)d;

	// grow the forest on bootstrap samples (sequential, fixed order — deterministic)
	_forest.assign(_tree_count,std::vector<Node>());
	std::vector<IDXTYPE> samples(N);
	for(unsigned int t=0;t<_tree_count;t++)
	{
		for(IDXTYPE n=0;n<N;n++) samples[n]=rng_below(N);
		grow_tree(_forest[t],samples,mtry);
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::grow_tree(std::vector<Node> &tree, std::vector<IDXTYPE> &samples, const DIMTYPE mtry)
{
	const std::size_t d=_subfeatures;
	const DIMTYPE K=_classes;
	struct Work {long node; IDXTYPE lo, hi; unsigned int depth;}; // samples[lo..hi) belong to the node
	std::vector<Work> stack;
	std::vector<IDXTYPE> cnt(K), cntL(K);
	std::vector<DIMTYPE> feats(d);
	std::vector<IDXTYPE> sortbuf;

	tree.clear();
	tree.push_back(Node());
	stack.push_back(Work{0,0,(IDXTYPE)samples.size(),0});
	while(!stack.empty())
	{
		const Work w=stack.back(); stack.pop_back();
		const IDXTYPE n=w.hi-w.lo;
		assert(n>0);

		// class histogram; majority with lowest-index tie-break
		std::fill(cnt.begin(),cnt.end(),(IDXTYPE)0);
		for(IDXTYPE ii=w.lo;ii<w.hi;ii++) cnt[_y[samples[ii]]]++;
		DIMTYPE major=0;
		for(DIMTYPE c=1;c<K;c++) if(cnt[c]>cnt[major]) major=c;
		const bool pure=(cnt[major]==n);

		// n<2*_min_leaf is a shortcut only: no split of n samples can give both children at
		// least _min_leaf. The per-split-point check further below enforces _min_leaf exactly.
		bool make_leaf=pure || n<2*_min_leaf || (_max_depth>0 && w.depth>=_max_depth);
		DIMTYPE best_f=0; REALTYPE best_thr=0.0, best_gain=(REALTYPE)-1.0;
		if(!make_leaf)
		{
			// draw mtry distinct candidate features (partial Fisher-Yates, fixed order)
			for(std::size_t f=0;f<d;f++) feats[f]=(DIMTYPE)f;
			for(DIMTYPE k=0;k<mtry;k++) {const IDXTYPE j=k+rng_below((IDXTYPE)(d-k)); std::swap(feats[k],feats[j]);}

			for(DIMTYPE k=0;k<mtry;k++)
			{
				const DIMTYPE f=feats[k];
				sortbuf.assign(samples.begin()+w.lo,samples.begin()+w.hi);
				// total order (value, sample index) — deterministic under equal values
				std::sort(sortbuf.begin(),sortbuf.end(),[this,d,f](const IDXTYPE a, const IDXTYPE b)
					{const REALTYPE va=_X[(std::size_t)a*d+f], vb=_X[(std::size_t)b*d+f]; return va<vb || (va==vb && a<b);});
				// sweep split points between distinct consecutive values; with sL,sR the sums of
				// squared class counts on each side, maximizing sL/nL + sR/nR is exactly the same
				// ordering as minimizing the weighted Gini impurity (nL*giniL + nR*giniR), and
				// avoids the divisions the impurity form would need
				std::fill(cntL.begin(),cntL.end(),(IDXTYPE)0);
				for(IDXTYPE ii=0;ii+1<n;ii++)
				{
					cntL[_y[sortbuf[ii]]]++;
					const REALTYPE v0=_X[(std::size_t)sortbuf[ii]*d+f], v1=_X[(std::size_t)sortbuf[ii+1]*d+f];
					if(!(v0<v1)) continue; // tie group — no split point here
					const IDXTYPE nL=ii+1, nR=n-nL;
					if(nL<_min_leaf || nR<_min_leaf) continue;
					REALTYPE sL=0.0, sR=0.0;
					for(DIMTYPE c=0;c<K;c++) {
						const REALTYPE l=(REALTYPE)cntL[c], r=(REALTYPE)(cnt[c]-cntL[c]);
						sL+=l*l; sR+=r*r;
					}
					const REALTYPE gain=sL/(REALTYPE)nL+sR/(REALTYPE)nR;
					if(gain>best_gain) { // strict: equal gain keeps the first candidate encountered
						REALTYPE thr=(v0+v1)/(REALTYPE)2.0;
						// FP-midpoint guard: for adjacent representable values the midpoint can round
						// up to v1, which would put the v1 samples on the left as well and empty the
						// right child; falling back to v0 keeps the partition as the sweep assumed
						if(!(thr<v1)) thr=v0;
						best_gain=gain; best_f=f; best_thr=thr;
					}
				}
			}
			if(best_gain<(REALTYPE)0.0) make_leaf=true; // no valid split point on any candidate (e.g. all-tied values)
		}

		if(make_leaf)
		{
			tree[w.node].left=-1; tree[w.node].right=-1;
			tree[w.node].feature=0; tree[w.node].threshold=0.0;
			tree[w.node].leaf_class=major;
			continue;
		}

		// partition the slice (stable — deterministic order for the children)
		auto mid=std::stable_partition(samples.begin()+w.lo,samples.begin()+w.hi,
			[this,d,best_f,best_thr](const IDXTYPE a){return _X[(std::size_t)a*d+best_f]<=best_thr;});
		const IDXTYPE cut=(IDXTYPE)(mid-samples.begin());
		assert(cut>w.lo && cut<w.hi); // guaranteed by the split search (both children hold >=_min_leaf>=1 samples);
		                              // were it ever violated with asserts disabled, an empty child slice would
		                              // trip the assert(n>0) at the top of the next iteration

		const long left=(long)tree.size(); tree.push_back(Node());
		const long right=(long)tree.size(); tree.push_back(Node());
		tree[w.node].feature=best_f; tree[w.node].threshold=best_thr;
		tree[w.node].left=left; tree[w.node].right=right; tree[w.node].leaf_class=0;
		stack.push_back(Work{right,cut,w.hi,w.depth+1});
		stack.push_back(Work{left,w.lo,cut,w.depth+1});
	}
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
DIMTYPE Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::tree_vote(const std::vector<Node> &tree, const PPattern &pattern) const
{
	long node=0;
	while(tree[node].left>=0)
	{
		const Node &nd=tree[node];
		node=((REALTYPE)pattern[_index[nd.feature]]<=nd.threshold)?nd.left:nd.right;
	}
	return tree[node].leaf_class;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	assert(_subfeatures>0 && !_forest.empty()); // train() must have succeeded
	_votes.assign(_classes,(IDXTYPE)0); // member scratch: avoids a per-pattern allocation
	for(const auto &tree : _forest) _votes[tree_vote(tree,pattern)]++;
	cls=0;
	for(DIMTYPE c=1;c<_classes;c++) if(_votes[c]>_votes[cls]) cls=c; // ties resolve to the lowest class index
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Random_Forest<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify_scores(std::vector<RETURNTYPE> &scores, const PPattern &pattern)
{
	assert(_subfeatures>0 && !_forest.empty());
	scores.assign(_classes,(RETURNTYPE)0.0);
	for(const auto &tree : _forest) scores[tree_vote(tree,pattern)]+=1.0;
	for(DIMTYPE c=0;c<_classes;c++) scores[c]/=(RETURNTYPE)_forest.size();
	return true;
}


} // namespace
#endif // FSTCLASSIFIERRF_H ///:~
