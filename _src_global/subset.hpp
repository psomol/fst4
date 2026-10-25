#ifndef FSTSUBSET_H
#define FSTSUBSET_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    subset.hpp
   \brief   Stores the info on currently selected features, enables generating permutations, etc.
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
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
#include <cassert>
#include <iostream>
#include <cstdlib>
#include "fst_rand.hpp" // FST-owned PRNG: identical random sequences on every platform
#include "error.hpp"
#include "global.hpp"

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

template<class BINTYPE, typename DIMTYPE>
class Subset;

template<class BINTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb);

/*! \brief Stores the info on currently selected features, enables generating permutations, etc.
     \note the actual subset info is stored in _bin[],
     \note negative values mark de-selected features, positive values mark selected features. 0 is undefined.
     \note Basic identifier meaning: -1 feature deselected, 1 feature selected, 3 feature temporarily selected during traversal, 9 frozen feature (temporarily selected and protected from any changes)
     \note Those four concrete values are the FORWARD-mode convention (the constructor default).
     set_forward_mode(false) flips the sign of all of them: the step operations always write
     id_sel to the features they "select", so in backward mode selecting a candidate writes -1
     and thereby REMOVES it from the subset. Membership itself is mode-independent - a feature
     is in the subset iff its marker is positive (get_d(), selected_features(), stateless_copy()
     and every consumer rely on that). Code that drives traversal or steps directly must compare
     against id_sel/id_desel/id_traverse/id_frozen, never against literals.
*/
template<class BINTYPE, typename DIMTYPE>
class Subset {
	//! Streams a subset's selected features to \p os (as its print(), see the .cpp-side definition).
	friend std::ostream& operator<< <>(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb);
public:
	Subset(const DIMTYPE n);	//!< n - max possible no. of features = allocation size
	//! \brief Copy constructor: allocates its own marker array and reproduces \p sub's selection, its
	//! forward/frozen mode and its selected-feature cursor. The state of a candidate-subset traversal
	//! in progress is NOT reproduced, so a copy must not be used to continue one -- start a fresh
	//! getFirstCandidateSubset() on the copy instead.
	Subset(const Subset<BINTYPE,DIMTYPE>& sub);
	~Subset();
	//! \brief Overwrites this subset's selection state from \p sub's currently selected features (via getFirstFeature()/getNextFeature()).
	//! Every feature ends up exactly id_sel or id_desel (frozen features are left frozen); \p sub's own
	//! internal iteration/traversal state and forward/frozen mode are NOT copied -- only which features
	//! are selected. \p this must have get_n_raw() >= \p sub's.
	//! \note The markers are written as the literal values +1 (member) and -1 (non-member). That is correct
	//! in BOTH modes: membership is always encoded by the SIGN of the marker (get_d(), selected_features()
	//! and every consumer test "marker > 0"), and set_forward_mode() only flips which sign the step
	//! operations' notion of "select" writes - so a backward-mode subset receives exactly \p sub's
	//! selection, as the backward searches rely on. What is not copied is this subset's mode and
	//! traversal state. The call advances \p sub's embedded getFirstFeature()/getNextFeature() cursor,
	//! which is why \p sub is taken by non-const reference.
	void stateless_copy(Subset<BINTYPE,DIMTYPE>& sub); // copies subset from sub - all features converted to id_sel or id_desel only
	bool equivalent(const Subset<BINTYPE,DIMTYPE>& sub);	//!< tests whether sub represents the same subset (ignoring internal state)

	// adaptable methods - output depends on _frozen_mode - to be called in SearchMethods()
	void select(const DIMTYPE i);	//!< mark i-th feature as selected
	void select(const DIMTYPE _from, const DIMTYPE _to) {assert(_to>_from); assert(_to<get_n()); for(DIMTYPE i=_from;i<=_to;i++) select(i);}	//!< mark features _from,..,_to as selected
	void select_all() {for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_sel;}	//!< mark all (non-frozen) features as selected
	void deselect(const DIMTYPE i);	//!< mark i-th feature as deselected
	void deselect(const DIMTYPE _from, const DIMTYPE _to) {assert(_to>_from); assert(_to<get_n()); for(DIMTYPE i=_from;i<=_to;i++) deselect(i);}	//!< mark features _from,..,_to as deselected
	void deselect_all() {for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_desel;}	//!< mark all (non-frozen) features as deselected
	DIMTYPE get_n() const {DIMTYPE z=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_frozen) ++z; return _n-z;} //!< the number of features select()/deselect() address: get_n_raw() minus currently frozen features
	DIMTYPE get_d() const {DIMTYPE d=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]>0 && _bin[i]!=id_frozen && _bin[i]!=-id_frozen) ++d; return d;} //!< number of currently selected, non-frozen features (get_d_raw() minus frozen ones)

	// "raw" versions - to be called in Criterion()
	void select_raw(const DIMTYPE i) {assert(i<_n); _bin[i]=id_sel;}	//!< mark i-th feature as selected
	void select_raw(Subset<BINTYPE,DIMTYPE>& sub);	//!< select those features that are selected in sub
	void select_all_raw() {for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_sel;}	//!< mark all features (including frozen ones) as selected
	void deselect_raw(const DIMTYPE i) {assert(i<_n); _bin[i]=id_desel;}	//!< mark i-th feature as deselected
	void deselect_raw(Subset<BINTYPE,DIMTYPE>& sub);	//!< deselect those features that are selected in sub
	void deselect_all_raw() {for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_desel;}	//!< mark all features (including frozen ones) as deselected
	DIMTYPE get_n_raw() const {return _n;} //!< the full allocation size (max_n passed to the constructor), regardless of frozen features
	DIMTYPE get_d_raw() const {DIMTYPE d=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]>0) ++d; return d;} //!< number of features whose raw marker is positive: selected/traversed/frozen in forward mode, or deselected in backward mode (see set_forward_mode()) -- NOT simply "selected", and mode-dependent unlike get_d()
	bool selected_raw(const DIMTYPE i) const {if(i>=0&&i<_n) return (_bin[i]>0); else return false;} //!< true if raw feature index \p i is currently selected (or frozen); false (not an error) for an out-of-range index

	//! \brief Generate random subset of exactly \p d features.
	//! Draws from the library PRNG (FST::fst_rand(), seeded through FST::fst_srand()), so a fixed seed
	//! reproduces the same subsets on every platform. Frozen features are left untouched and do not count
	//! towards \p d, which must satisfy 1 <= d <= get_n().
	void make_random_subset(const DIMTYPE d);
	//! \brief Generate a random subset whose size is uniformly drawn from [lo_d,hi_d], then filled as make_random_subset(DIMTYPE) does.
	//! Requires lo_d < hi_d, both within 1..get_n().
	void make_random_subset(const DIMTYPE lo_d, const DIMTYPE hi_d);
	//! \brief Generate a random subset of random size; \p d_prob in (0,1], mean size over many calls is d_prob times the full feature count.
	//! Each non-frozen feature is selected independently with probability \p d_prob. The empty outcome is
	//! rejected and the whole draw repeated, so the result always has at least one selected feature; the
	//! size distribution is therefore binomial conditioned on being non-zero, not exactly binomial.
	void make_random_subset(const float d_prob);

	//! \brief Flips the sign convention of id_sel/id_desel/id_traverse/id_frozen (forward: selected>0; backward: selected<0).
	//! Lets SFS-style (forward) and SBS-style (backward) algorithms share the same representation.
	//! Precondition: no feature/candidate-subset traversal in progress, and NOT in frozen mode (freeze/unfreeze
	//! with getFirstCandidateSubset()'s \p reverse parameter instead if direction needs to change while frozen).
	void set_forward_mode(const bool forward=true);	//!< enables inverting the meaning of id_sel, id_desel etc.
	bool get_forward_mode() const {return _forward_mode;} //!< true if in forward mode (the constructor default); see set_forward_mode()

	//! \brief Freezes (or unfreezes) the currently selected features, excluding them from further select()/deselect()/get_n()/get_d().
	//! Freezing converts every id_sel (and id_traverse) feature to id_frozen; unfreezing converts every
	//! id_frozen (and id_traverse) feature back to id_sel. Frozen features are still visible to the
	//! get*Feature() traversals and count in get_n_raw()/get_d_raw(), just not in get_n()/get_d() or the
	//! adaptable select()/deselect() indexing. Precondition: no feature/candidate-subset traversal in
	//! progress, and \p freeze must differ from the current mode (repeated calls with the same value are
	//! asserted against, not idempotent).
	void set_frozen_mode(const bool freeze=false);
	bool get_frozen_mode() const {return _frozen_mode;} //!< true if currently frozen; see set_frozen_mode()

	// NOTE: regardles _frozen_mode always returns all selected features (id_sel, id_frozen, id_traverse)
	//! \deprecated Deprecated since FST4 4.0; prefer the reentrant selected_features() range, which owns its
	//! iteration state instead of sharing the one cursor embedded in the Subset. Removal planned for the next
	//! major release. \p selected and \p looplevel are accepted for source compatibility and ignored: the pair
	//! always walks every feature with a positive raw marker.
	bool getFirstFeature(DIMTYPE &feat, bool selected=true, const int looplevel=0); // NOTE: selected and looplevel so far ignored
	//! \deprecated Deprecated since FST4 4.0; prefer the reentrant selected_features() range. Must be preceded
	//! by getFirstFeature() (asserted); returns false once the features are exhausted, and resets the cursor.
	bool getNextFeature(DIMTYPE &feat);

	bool getFirstCandidateSubset(const DIMTYPE g, bool reverse=false); //!< begins traversal of all g-tuples of currently DESELECTED features (candidates to ADD) if \p reverse is false, or of currently SELECTED features (candidates to REMOVE) if true; each tuple is marked via tmp_id_traverse, read through getFirstTemporaryFeature()/temporary_features()
	bool getNextCandidateSubset(); //!< advances to the next g-tuple begun by getFirstCandidateSubset(); returns false once exhausted
	DIMTYPE getNoOfCandidateSubsets(const DIMTYPE g, bool reverse=false); //!< the number of g-tuples getFirstCandidateSubset()/getNextCandidateSubset() will traverse; throws on DIMTYPE overflow (possible for large n with g far from 1 or n)
	//! \deprecated Deprecated since FST4 4.0; prefer the reentrant temporary_features() range, which owns its
	//! iteration state instead of sharing the one cursor embedded in the Subset. Removal planned for the next
	//! major release.
	bool getFirstTemporaryFeature(DIMTYPE &feat); // traversing through temporary features in current CandidateSubset
	//! \deprecated Deprecated since FST4 4.0; prefer the reentrant temporary_features() range. Must be preceded
	//! by getFirstTemporaryFeature() (asserted); returns false once the g-tuple is exhausted, and resets the cursor.
	bool getNextTemporaryFeature(DIMTYPE &feat); // traversing through temporary features in current CandidateSubset
	DIMTYPE getNoOfTemporaryFeatures(); //!< the number of features currently marked tmp_id_traverse (the g-tuple size of the CandidateSubset traversal in progress)

	/*! \brief Read-only forward range over feature indices

	    Returned by selected_features()/temporary_features(). Unlike the deprecated
	    get*Feature() cursor pairs the range owns its complete iteration state:
	    nested and early-exited loops are safe and the Subset stays untouched.
	    Usage: for(DIMTYPE f : sub->selected_features()) {...}
	    The range holds a raw pointer to the Subset and reads its markers lazily, so it must not
	    outlive the Subset, and the selection must not be modified while a range is being iterated. */
	class Feature_Range {
	public:
		//! Which raw features a Feature_Range/iterator matches.
		enum Kind {
			SELECTED, //!< features with a positive raw marker (selected, frozen, or under traversal; matches get_d_raw(), not get_d() -- see selected_features())
			TEMPORARY //!< features currently marked tmp_id_traverse by an in-progress getFirstCandidateSubset()/getNextCandidateSubset() traversal (see temporary_features())
		};
		//! Forward iterator over raw feature indices matching a Kind, skipping non-matching ones.
		class iterator {
		public:
			//! Positions at feature index \p i or the first matching feature at or after it (advance() skips non-matches); \p s is the subset iterated, \p k the Kind matched.
			iterator(const Subset *s, const DIMTYPE i, const Kind k) : _s(s), _i(i), _k(k) {advance();}
			DIMTYPE operator*() const {return _i;} //!< the current (raw) feature index
			iterator& operator++() {++_i; advance(); return *this;} //!< advances to the next matching feature index
			bool operator!=(const iterator& o) const {return _i!=o._i;} //!< range-for only ever compares against end(); true until the index reaches end()'s
		private:
			bool match() const {return (_k==SELECTED) ? (_s->_bin[_i]>0) : (_s->_bin[_i]==_s->tmp_id_traverse);} //!< whether the current index matches this iterator's Kind
			void advance() {while(_i<_s->_n && !match()) ++_i;} //!< skips forward from the current index to the next match (or to end())
			const Subset *_s; //!< the subset iterated
			DIMTYPE _i;        //!< the current raw feature index
			Kind _k;           //!< which Kind is matched
		};
		//! Constructs a range over \p s's features of the given \p k (SELECTED or TEMPORARY).
		Feature_Range(const Subset *s, const Kind k) : _s(s), _k(k) {}
		iterator begin() const {return iterator(_s,0,_k);} //!< an iterator positioned at the first matching feature, or already at end() if there are none
		iterator end() const {return iterator(_s,_s->_n,_k);} //!< the sentinel end iterator
	private:
		const Subset *_s; //!< the subset iterated
		Kind _k;           //!< which kind of feature this range selects
	};
	//! reentrant range over all currently selected features (id_sel, id_frozen, id_traverse - as with getFirstFeature())
	Feature_Range selected_features() const {return Feature_Range(this,Feature_Range::SELECTED);}
	//! reentrant range over temporary features in the current CandidateSubset (as with getFirstTemporaryFeature())
	Feature_Range temporary_features() const {return Feature_Range(this,Feature_Range::TEMPORARY);}

private:
	const DIMTYPE _n;	//!< max. No. of features (full set size)
	const std::unique_ptr<BINTYPE[]> _bin;	//!< default format to store subset info

	BINTYPE id_sel;      //!< the marker value for a selected feature; sign flips with set_forward_mode()
	BINTYPE id_desel;    //!< the marker value for a deselected feature; sign flips with set_forward_mode()
	BINTYPE id_traverse; //!< the marker value for a feature in the g-tuple currently being traversed by getFirstCandidateSubset()/getNextCandidateSubset()
	BINTYPE id_frozen;   //!< the marker value for a frozen feature (see set_frozen_mode())
	BINTYPE tmp_id_sel;      //!< id_sel/id_desel/id_traverse as resolved for the CURRENT getFirstCandidateSubset() call's \p reverse direction (independent of set_forward_mode()); local to get*CandidateSubset()
	BINTYPE tmp_id_desel;    //!< see tmp_id_sel; local to get*CandidateSubset()
	BINTYPE tmp_id_traverse; //!< see tmp_id_sel; this is the value Feature_Range's TEMPORARY kind (temporary_features()) actually matches against; local to get*CandidateSubset()
	//! Number of size-\p k combinations of \p n features (n choose k); throws if the result overflows DIMTYPE (or a 64-bit intermediate).
	DIMTYPE get_combinations(const DIMTYPE n, const DIMTYPE k); // NOTE: throws if C(n,k) overflows DIMTYPE (or 64-bit intermediates)
	
	bool _forward_mode; //!< set by set_forward_mode(); read by get_forward_mode()
	bool _frozen_mode;  //!< set by set_frozen_mode(); read by get_frozen_mode()

	DIMTYPE _gen_features;   //!< the g-tuple size of the getFirstCandidateSubset() traversal in progress, or 0 if none
	DIMTYPE _enum_temporary; //!< cursor state for the deprecated getFirstTemporaryFeature()/getNextTemporaryFeature() pair

	DIMTYPE _enum_selected;  //!< cursor state for the deprecated getFirstFeature()/getNextFeature() pair
};

template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::Subset(const DIMTYPE n) : _n(n), _bin(new BINTYPE[n]) 
{
	id_desel=-1; id_sel=1; id_traverse=3; id_frozen=9;
	for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_desel;
	_forward_mode=true;
	_frozen_mode=false;
	_gen_features=0;
	_enum_selected=0;
	_enum_temporary=0;
	notify("Subset constructor.");
}


template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::~Subset() 
{
	notify("Subset destructor.");
}


template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::Subset(const Subset<BINTYPE,DIMTYPE>& sub) : _n(sub._n), _bin(new BINTYPE[sub._n])
{
	id_sel=sub.id_sel; id_desel=sub.id_desel; id_traverse=sub.id_traverse; id_frozen=sub.id_frozen;
	for(DIMTYPE i=0;i<_n;i++) _bin[i]=sub._bin[i];
	_forward_mode=sub._forward_mode;
	_frozen_mode=sub._frozen_mode;
	_gen_features=sub._gen_features;
	_enum_selected=sub._enum_selected;
	_enum_temporary=sub._enum_temporary;
	notify("Subset copy-constructor.");
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::stateless_copy(Subset<BINTYPE,DIMTYPE>& sub)
{
	// NOTE: the literal +1/-1 written below are mode-independent - membership is encoded by the
	//       marker's sign in both modes (see the class notes); only the step operations' meaning
	//       of "select" flips with set_forward_mode()
	assert(_bin);
	assert(_n>=sub.get_n_raw());
	for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen && _bin[i]!=-id_frozen) _bin[i]=-1; // NOTE: hard constant
	DIMTYPE f;
	for(bool b=sub.getFirstFeature(f);b!=false;b=sub.getNextFeature(f)) {
		assert(f<_n);
		if(_bin[f]!=id_frozen && _bin[f]!=-id_frozen) _bin[f]=1; // NOTE: hard constant
	}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::equivalent(const Subset<BINTYPE,DIMTYPE>& sub)
{
	assert(_bin);
	assert(_n>0);
	bool eq=true;
	DIMTYPE fidx=0;
	while(eq && fidx<_n) if((_bin[fidx]<=0 && sub._bin[fidx]>0)||(_bin[fidx]>0 && sub._bin[fidx]<=0)) eq=false; else ++fidx;
	return eq;
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::select(const DIMTYPE i)
{
	assert(_bin);
	assert(i<get_n());
	DIMTYPE u=0, c=i;
	while(u<_n && _bin[u]==id_frozen) ++u;
	while(u<_n && c>0) {
		++u; --c;
		while(u<_n && _bin[u]==id_frozen) ++u;
	}
	assert(u<_n);
	_bin[u]=id_sel;
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::deselect(const DIMTYPE i)
{
	assert(_bin);
	assert(i<get_n());
	DIMTYPE u=0, c=i;
	while(u<_n && _bin[u]==id_frozen) ++u;
	while(u<_n && c>0) {
		++u; --c;
		while(u<_n && _bin[u]==id_frozen) ++u;
	}
	assert(u<_n);
	_bin[u]=id_desel;
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::select_raw(Subset<BINTYPE,DIMTYPE>& sub)
{
	assert(_bin);
	//for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen && _bin[i]!=-id_frozen) _bin[i]=-1; // NOTE: hard constant
	bool go = true;
	DIMTYPE f;
	for(bool b=sub.getFirstFeature(f);go && b!=false;b=sub.getNextFeature(f)) {
		if(f<_n) _bin[f]=id_sel; else go=false;
	}
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::deselect_raw(Subset<BINTYPE,DIMTYPE>& sub)
{
	assert(_bin);
	//for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen && _bin[i]!=-id_frozen) _bin[i]=-1; // NOTE: hard constant
	bool go = true;
	DIMTYPE f;
	for(bool b=sub.getFirstFeature(f);go && b!=false;b=sub.getNextFeature(f)) {
		if(f<_n) _bin[f]=id_desel; else go=false;
	}
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const DIMTYPE d)
{
	assert(_bin);
	assert(d>=1 && d<=get_n());
	// assert bin must contain >=d non-frozen slots
	DIMTYPE i,piv;
	for(i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_desel;
	for(i=0;i<d;i++)
	{ 
		piv=(DIMTYPE)fst_rand_below((unsigned long)_n); assert(piv<_n);
		while(_bin[piv]!=id_desel) {piv++; if(piv>_n-1) piv=0;}
		_bin[piv]=id_sel;
	}
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const DIMTYPE lo_d, const DIMTYPE hi_d)
{
	assert(lo_d>=1 && lo_d<=get_n());
	assert(hi_d>=1 && hi_d<=get_n());
	assert(lo_d<=hi_d);
	DIMTYPE d;
	d=lo_d+(DIMTYPE)fst_rand_below((unsigned long)(hi_d-lo_d+1)); assert(lo_d<=d && d<=hi_d);
	make_random_subset(d);
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const float d_prob) // d_prob must be from (0,1]; generates random subset of random size where the mean size over a series of calls is d_prob*no_of_all_features
{
	assert(_bin);
	assert(d_prob>0.0 && d_prob<=1.0);
	do {
		DIMTYPE i;
		for(i=0;i<_n;i++) if(_bin[i]!=id_frozen) {
			if((float)fst_rand()/(float)FST_RAND_MAX<d_prob) _bin[i]=id_sel;
			else _bin[i]=id_desel;
		}
	} while(get_d()==0);
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::set_forward_mode(const bool forward)
{
	assert(_gen_features==0); // no traversal is going on
	assert(_enum_selected==0); // no feature iteration is going on
	assert(_frozen_mode==false); // in frozen mode change direction of traversal using getFirstCandidateSubset() 'reverse' parameter
	if(forward) {
		id_desel=-1; id_sel=1; id_traverse=3; /*if(_frozen_mode==0)*/ id_frozen=9;
		_forward_mode=true;
	} else {
		id_desel=1; id_sel=-1; id_traverse=-3; /*if(_frozen_mode==0)*/ id_frozen=-9;
		_forward_mode=false;
	}
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::set_frozen_mode(const bool freeze)
{
	assert(_bin);
	assert(_gen_features==0); // no traversal is going on
	assert(_enum_selected==0); // no feature iteration is going on
	assert(freeze!=_frozen_mode); // repeated initiation of the same mode would produce unexpected results
	if(freeze) {
		for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_sel || _bin[i]==id_traverse) _bin[i]=id_frozen;
		_frozen_mode=true;
	} else {
		for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_frozen || _bin[i]==id_traverse) _bin[i]=id_sel;
		_frozen_mode=false;
	}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstFeature(DIMTYPE &feat, bool selected, const int looplevel)
{
	assert(_bin);
	assert(_n>0);
	_enum_selected=0;
	while(_enum_selected<_n && _bin[_enum_selected]<=0) ++_enum_selected;
	feat=_enum_selected;
	_enum_selected++;
	// reset on unsuccessful end-of-iteration just as getNextFeature() does: set_forward_mode() and
	// set_frozen_mode() assert that no iteration is in progress, and a cursor left non-zero by a
	// traversal that found nothing would make those asserts fire spuriously
	if(feat<_n) return true; else {_enum_selected=0; return false;}
}


template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextFeature(DIMTYPE &feat)
{
	assert(_bin);
	assert(_n>0);
	assert(_enum_selected>0);	// i.e., getFirstFeature has been called before
	while(_enum_selected<_n && _bin[_enum_selected]<=0) ++_enum_selected;
	feat=_enum_selected;
	_enum_selected++;
	if(feat<_n) return true; else {_enum_selected=0; return false;}
}


template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstCandidateSubset(const DIMTYPE g, bool reverse) // traversing through candidate subsets (adding/removing) g-touples of features
{
	assert(_bin);
	assert(g>=1 && g<=_n);
	if(reverse) {tmp_id_sel=-id_sel; tmp_id_desel=-id_desel; tmp_id_traverse=-id_traverse;} 
	else {tmp_id_sel=id_sel; tmp_id_desel=id_desel; tmp_id_traverse=id_traverse;}	
	DIMTYPE rr=g;
	for(DIMTYPE i=0;((rr>0)&&(i<_n));i++) if(_bin[i]==tmp_id_desel) {_bin[i]=tmp_id_traverse; --rr;} /* initialize _bin for exhaustive step */
	if(rr==0) {_gen_features=g; return true;}
	else {for(DIMTYPE i=0;((rr>0)&&(i<_n));i++) if(_bin[i]==tmp_id_traverse) _bin[i]=tmp_id_desel; _gen_features=0; return false;}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextCandidateSubset()
{
	assert(_bin);
	assert(_gen_features>0 && _gen_features<=_n); // g==_n is a legitimate traversal (the single full set), reached by an all-cardinalities exhaustive search
	DIMTYPE beg, piv, pom;
	/* finding new candidate configuration.
	   The g traversed positions are enumerated in lexicographic order over the eligible
	   (tmp_id_desel) positions: find the rightmost traversed marker that still has an eligible
	   position to its right, move it one eligible position right, then pack all traversed markers
	   left of it back against the start of the array. Exhaustion is detected by there being no
	   eligible position to the right at all, in which case every traversed marker is restored to
	   tmp_id_desel and the traversal ends. */
	for(beg=0;beg<_n && _bin[beg]!=tmp_id_traverse;beg++);
	for(piv=beg;piv<_n && _bin[piv]!=tmp_id_desel;piv++);
	if(piv==_n) { // no more configurations
		for(beg=0;beg<_n;beg++) if(_bin[beg]==tmp_id_traverse) _bin[beg]=tmp_id_desel;
		_gen_features=0;
		return false;
	}
	else
	{
		pom=piv; /* remember the position of first (desel) on the right */
		do piv--; while(_bin[piv]!=tmp_id_traverse); /* find the real pivot */
		_bin[piv]=tmp_id_desel; _bin[pom]=tmp_id_traverse; /* shift pivot to the right */
		pom=0;
		/* run "pom" from left, "piv" from right. the desel,travers pairs found are changed to travers,desel */
		if(piv>0) --piv;
		while((piv>0)&&(_bin[piv]!=tmp_id_traverse)) --piv;
		while((pom<piv)&&(_bin[pom]!=tmp_id_desel)) pom++;
		while(piv>pom)
		{
			_bin[piv]=tmp_id_desel; _bin[pom]=tmp_id_traverse;
			if(piv>0) --piv;
			while((piv>0)&&(_bin[piv]!=tmp_id_traverse)) --piv;
			while((pom<piv)&&(_bin[pom]!=tmp_id_desel))pom++;
		}
	}
	return true;
}

template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::getNoOfCandidateSubsets(const DIMTYPE g, bool reverse) // returns loop size (the number of to-be-traversed subsets)
{
	// NOTE: get_combinations() throws if the count overflows DIMTYPE;
	//       no issue for g==1, possible for large n with g farther from 1 or n
	assert(_bin);
	assert(g>=1 && g<_n);
	BINTYPE id_empty; if(reverse) id_empty=-id_desel; else id_empty=id_desel;
	DIMTYPE ecnt=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_empty) ++ecnt;
	if(g>ecnt) return 0;
	if(g==1) return ecnt; // just acceleration
	else return get_combinations(ecnt,g);
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstTemporaryFeature(DIMTYPE &feat) // traversing through temporary features in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	_enum_temporary=0;
	while(_enum_temporary<_n && _bin[_enum_temporary]!=tmp_id_traverse) ++_enum_temporary;
	feat=_enum_temporary;
	_enum_temporary++;
	if(feat<_n) return true; else {_enum_temporary=0; return false;}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextTemporaryFeature(DIMTYPE &feat) // traversing through temporary features in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	assert(_enum_temporary>0);	// i.e., getFirstTemporaryFeature has been called before
	while(_enum_temporary<_n && _bin[_enum_temporary]!=tmp_id_traverse) ++_enum_temporary;
	feat=_enum_temporary;
	_enum_temporary++;
	if(feat<_n) return true; else {_enum_temporary=0; return false;}
}
template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::getNoOfTemporaryFeatures() // returns the number of temporarily selected features (indicated by tmp_id_traverse) in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	DIMTYPE _sum=0;
	for(DIMTYPE i=0; i<_n; i++) if(_bin[i]==tmp_id_traverse) _sum++;
	return _sum;
}

template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::get_combinations(const DIMTYPE n, const DIMTYPE k)
{
	assert(n>0);
	assert(n>=k);
	if(k==0 || n==k) return 1;
	// C(n,k)==C(n,n-k) - iterate over the smaller of the two to minimize intermediates
	const unsigned long long kk=((unsigned long long)k<=(unsigned long long)(n-k))?(unsigned long long)k:(unsigned long long)(n-k);
	// incremental product C(n,kk)=prod_{i=1..kk}(n-kk+i)/i; after step i the value equals
	// C(n-kk+i,i), an integer, so each division is exact and overflow can occur only in
	// the multiplication (checked below) or in the final narrowing to DIMTYPE
	unsigned long long result=1, tempcheck;
	for(unsigned long long i=1;i<=kk;i++) {
		const unsigned long long factor=(unsigned long long)n-kk+i;
		tempcheck=result*factor;
		if(tempcheck/factor!=result) throw FST::fst_error("Subset::get_combinations() numeric representation overflow.");
		result=tempcheck/i;
	}
	DIMTYPE dimresult=(DIMTYPE)result;
	if((unsigned long long)dimresult!=result) throw FST::fst_error("Subset::get_combinations() numeric representation overflow.");
	return dimresult;
}

//! \brief Streams a subset's size and selected feature indices to \p os, e.g. "Subset<n=20, d=3>: 2 5 11".
//! The listed indices are those with a positive raw marker, so in backward mode they are the DEselected
//! ones. In builds compiled with the DEBUG macro defined, the raw marker array is dumped before the index
//! list (separated by " ~ "), which makes such output differ from a release build's.
template<class BINTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb)
{
	assert(tdb._bin);
	assert(tdb._n>0);
	assert(tdb._bin);
	const DIMTYPE d_raw=tdb.get_d_raw();
	os << "Subset<n=" << tdb._n << ", d=" << d_raw <<">: ";
#ifdef DEBUG
	for(DIMTYPE i=0;i<tdb._n;i++) {os << (int)tdb._bin[i] << " ";} 
	os << "  ~  ";
#endif
	if(d_raw==0) os << "empty";
	else for(DIMTYPE i=0;i<tdb._n;i++) {if(tdb._bin[i]>0) os << i << " ";}
	return os;
}

} // namespace
#endif // FSTSUBSET_H ///:~
