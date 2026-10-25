#ifndef FSTRESULTTRACKERFEATURESTATS_H
#define FSTRESULTTRACKERFEATURESTATS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    result_tracker_feature_stats.hpp
   \brief   Computes feature occurrence statistics in a series of evaluated subsets
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
#ifdef FST_THREADED
#include <mutex>
#endif
#include <iostream>
#include <sstream>
#include <list>
#include <vector>
#include <algorithm>
#include <cstddef>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker_dupless.hpp"

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

/*! \brief Collects evaluated subsets to eventually provide Dependency-Aware Feature Ranking coefficients.

	Dependency-Aware Feature ranking (DAF) is a new type of ranking method especially
	suitable for very-high-dimensional feature selection. Unlike standard individual
	feature ranking, the DAF ranking reflects "average feature quality in context"
	and as such is capable of providing significantly better results than BIF
	(provided the data actually do contain mutually dependent features). The method
	has been described in combination with Monte Carlo based feature selection
	permitting Wrapper Criteria even in very-high-dimensional setting.
	The tracker is not a search of its own: attach it to any search that generates
	many probe subsets (Monte Carlo search is the usual choice), let the search run,
	then call compute_stats() and read the ranking off getFirstDAF()/getNextDAF().
	For ready-to-run examples see the demo34 and demo35t protocols
	(`demos/demo34.cfg`, `demos/demo35t.cfg`). For more detailed information see
	UTIA Technical Report No. 2295.
*/
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Feature_Stats : public Clonable_Impl<Result_Tracker_Feature_Stats<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>,Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> {
public:
	typedef Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> parent; //!< the base tracker: dupless probe storage sorted descending by criterion value; this class adds DAF post-processing over it
	typedef Clonable_Impl<Result_Tracker_Feature_Stats,parent,CLONE_INHERIT,CLONE_INHERIT,CLONE_DEEP> parent_impl; //!< implements stateless_clone(); clone()/sharing_clone() throw in Result_Tracker
	typedef typename parent::ResultRec ResultRec; //!< [subset,criterion value] result record, see Result_Tracker::ResultRec
	typedef ResultRec* PResultRec; //!< raw pointer to a ResultRec, as used by the inherited get_first()/get_next() iteration
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a tracked feature subset
	//! \param capacity_limit forwarded to Result_Tracker_Dupless; 0 (the fst4 CLI default for this tracker) means unlimited probe storage
	Result_Tracker_Feature_Stats(const IDXTYPE capacity_limit=0):parent_impl(capacity_limit), _n(0), _pair_itersize(0) {_itersize[0]=0; _itersize[1]=0; _itersize[2]=0; notify("Result_Tracker_Feature_Stats constructor.");}
	Result_Tracker_Feature_Stats(const Result_Tracker_Feature_Stats& rtfs):parent_impl(rtfs), _n(0), _pair_itersize(0) {_itersize[0]=0; _itersize[1]=0; _itersize[2]=0; notify("Result_Tracker_Feature_Stats copy-constructor.");} //!< the stateless_clone() copy: replicates configuration (capacity limit, margin) via the parent copy-constructor but starts with an empty probe list and no computed statistics -- compute_stats()/compute_pair_stats() must be rerun on the clone
	virtual ~Result_Tracker_Feature_Stats() {notify("Result_Tracker_Feature_Stats destructor.");}

	/*! \brief Aggregates DAF ranking statistics (DAF0/DAF1/DAF2) from all probe subsets recorded so far.

	    For each feature, splits the recorded (subset,value) probes into those where the
	    feature is selected ("is") and not selected ("isnot"), and computes per-partition
	    mean/stdev of the criterion value plus DAF0 = mean_is-mean_isnot and DAF1 = DAF0
	    normalized by the frequency-weighted partition stdevs; DAF2 is a further variant,
	    normalized per probe by the stdev of same-cardinality probes (the "DAF3" measure of
	    UTIA Technical Report No. 2295) -- unlike DAF0/DAF1, DAF2 is marked valid for every
	    feature unconditionally, even when insufficient same-cardinality probes force its
	    value to 0. Also aggregates per-cardinality (per subset size) mean/stdev of the
	    criterion value. Builds the getFirstDAF()/getNextDAF() ordering for all three
	    variants (descending by DAF value, features without a valid value excluded). Must be
	    called (and re-called after adding more probes) before print_stats(),
	    getFirstDAF()/getNextDAF() or compute_pair_stats() return meaningful results.
	    \param os progress/warning messages are written here at NORMAL output detail or
	           above (see Result_Tracker::set_output_detail()); unused at SILENT
	    \return false if no probes have been recorded yet */
	bool compute_stats(std::ostream& os=std::cout);
	//! Prints the per-cardinality and per-feature statistics computed by compute_stats() to \p os.
	//! \return false if compute_stats() has not been run yet (nothing to print)
	bool print_stats(std::ostream& os=std::cout);

	/*! \brief Starts an iteration (descending by value) over the feature ordering built by compute_stats() for the given DAF variant.
	    \param value set to the feature's DAF value on success
	    \param feature set to the (global) feature index on success
	    \param DAFidx selects the DAF variant: 0 (DAF0), 1 (DAF1) or 2 (DAF2); the fst4 CLI's
	           `--daf-rank` only exposes 0 and 1, DAF2 is library-only
	    \return false if compute_stats() has not run or no feature has a valid value for
	            this variant */
	bool getFirstDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx=0) const;
	//! Continues the getFirstDAF() iteration for the same \p DAFidx.
	//! \return false once the ordering is exhausted (or if getFirstDAF() was never called for this DAFidx)
	bool getNextDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx=0) const;

	// ---- opt-in pairwise (group) DAF statistics ----
	/*! \brief Computes pairwise (group) DAF statistics: 2x2 presence-cell contrasts over all pairs of the top \p top_k DAF0-ranked features.

	    A post-processing pass purely over the probes already recorded by compute_stats()
	    (search logic and the per-feature DAF0/DAF1/DAF2 statistics above are unaffected).
	    It exposes the statistics only - no ranking heuristic is baked in. For each pair
	    (f1,f2) drawn from the top-k DAF0-ranked feature set, splits the recorded probes
	    into the four 2x2 presence cells (both selected, both absent, f1-only, f2-only);
	    only the both-present cell is accumulated directly per probe, the other three are
	    derived from it plus the per-feature and grand totals by inclusion-exclusion.
	    Reported per pair: pair-DAF0 = mean(both present) - mean(both absent); pair-DAF1 =
	    pair-DAF0 normalized by the frequency-weighted stdevs of those two cells (the DAF1
	    analogue); synergy = mean_both + mean_neither - mean_f1only - mean_f2only (the 2x2
	    interaction contrast). A pair is valid only if all four presence cells are
	    non-empty; among valid pairs, pair-DAF1 is additionally marked invalid if its
	    normalization denominator is zero. Builds the getFirstPairDAF()/getNextPairDAF()
	    ordering (valid pairs only, decreasing pair-DAF0).
	    \param top_k requires a prior compute_stats() call: the top \p top_k DAF0-ranked
	           features define the pair set; must be >=2
	    \param os progress/warning messages are written here at NORMAL output detail or above
	    \return false if compute_stats() has not been run, if fewer than 2 features have a
	            valid DAF0, if \p top_k<2, or if no pair turns out valid */
	bool compute_pair_stats(const DIMTYPE top_k, std::ostream& os=std::cout);
	//! Prints the pairwise statistics computed by compute_pair_stats() to \p os, best pair-DAF0 first.
	//! \param os the output stream to print to
	//! \param top_pairs how many top pairs to print; 0 (the default) prints all valid pairs
	//! \return false if compute_pair_stats() has not run (or found no valid pair)
	bool print_pair_stats(std::ostream& os=std::cout, const std::size_t top_pairs=0);
	/*! \brief Starts an iteration (decreasing pair-DAF0) over the pairwise statistics built by compute_pair_stats().
	    \param daf0 set to mean(both present) - mean(both absent) for the pair
	    \param daf1 set to the DAF1-analogue normalization of \p daf0; 0 if \p daf1_valid is false
	    \param daf1_valid set to false when the pair's normalization denominator is zero (\p daf1 then holds 0)
	    \param synergy set to the 2x2 interaction contrast (mean_both + mean_neither - mean_f1only - mean_f2only)
	    \param f1 set to the (global) index of the pair's first feature
	    \param f2 set to the (global) index of the pair's second feature
	    \return false if compute_pair_stats() has not run or found no valid pair */
	bool getFirstPairDAF(RETURNTYPE &daf0, RETURNTYPE &daf1, bool &daf1_valid, RETURNTYPE &synergy, DIMTYPE &f1, DIMTYPE &f2) const;
	//! Continues the getFirstPairDAF() iteration.
	//! \return false once the ordering is exhausted (or if getFirstPairDAF() was never called)
	bool getNextPairDAF(RETURNTYPE &daf0, RETURNTYPE &daf1, bool &daf1_valid, RETURNTYPE &synergy, DIMTYPE &f1, DIMTYPE &f2) const;



	virtual std::ostream& print(std::ostream& os) const override {os << "Result_Tracker_Feature_Stats(limit=" << parent::_capacity_limit << ") size " << parent::results.size(); return os;}

protected:
	DIMTYPE _n; //!< total number of features in the tracked dataset (from the first recorded probe's Subset::get_n(), assumed the same for all probes); also the length of _stats and one less than the length of _dstat
	//! Structure to gather feature occurrence statistics over probe subset evaluations
	typedef struct {
		RETURNTYPE freq_is;    //!< number of recorded probes containing this feature
		RETURNTYPE freq_isnot; //!< number of recorded probes not containing this feature
		RETURNTYPE mean_is;    //!< mean criterion value over probes containing this feature
		RETURNTYPE mean_isnot; //!< mean criterion value over probes not containing this feature
		RETURNTYPE stdev_is;   //!< criterion value stdev over probes containing this feature
		RETURNTYPE stdev_isnot; //!< criterion value stdev over probes not containing this feature
		//! DAF0 = mean_is-mean_isnot; DAF1 = DAF0 normalized by the frequency-weighted
		//! stdev_is/stdev_isnot; DAF2 = the per-cardinality-normalized variant (the "DAF3"
		//! measure of UTIA Technical Report No. 2295) -- see compute_stats()
		RETURNTYPE daf[3];
		bool daf_valid[3]; //!< per-DAF-variant validity flag; DAF2's entry is always true (see compute_stats())
	} FeatureStat;
	std::vector<FeatureStat> _stats; //!< one entry per feature (index 0.._n-1), aggregated over all recorded probes; filled by compute_stats()

	//! Structure to gather probe subset cardinality statistics
	typedef struct {
		RETURNTYPE freq;  //!< number of recorded probes with this exact subset size (cardinality)
		RETURNTYPE mean;  //!< mean criterion value over probes of this cardinality
		RETURNTYPE stdev; //!< criterion value stdev over probes of this cardinality
		bool valid;       //!< true once freq>0 and mean/stdev have been computed (see compute_stats())
	} SubSizeStat;
	std::vector<SubSizeStat> _dstat; //!< one entry per subset size 1.._n (index 0 unused); filled by compute_stats()

	typedef std::vector<DIMTYPE> ORDERTYPE; //!< feature-index ordering vector, as built per DAF variant by compute_stats()
	ORDERTYPE _order[3];  //!< per DAF variant (0=DAF0, 1=DAF1, 2=DAF2): feature indices with a valid DAF value, sorted descending by that value; built by compute_stats()
	mutable DIMTYPE _itersize[3]; //!< per DAF variant: getFirstDAF()/getNextDAF() cursor position (index into _order[DAFidx]); 0 = iteration not started
#ifdef FST_THREADED
	mutable std::mutex _itersize_mutex; //!< serializes access to the iteration cursors _itersize AND _pair_itersize across concurrent getFirst/getNext calls (threaded builds only)
#endif

	//! Structure to hold pairwise feature statistics (2x2 presence-cell contrasts over probe subset evaluations)
	typedef struct {
		DIMTYPE f1;              //!< global index of the pair's first feature
		DIMTYPE f2;              //!< global index of the pair's second feature
		RETURNTYPE freq_both;    //!< number of probes containing both features
		RETURNTYPE freq_neither; //!< number of probes containing neither feature
		RETURNTYPE daf0;         //!< mean(both present) - mean(both absent)
		RETURNTYPE daf1;         //!< daf0 normalized by frequency-weighted stdevs of the two cells (0 if daf1_valid==false)
		RETURNTYPE synergy;      //!< mean_both + mean_neither - mean_f1only - mean_f2only
		bool valid;              //!< all four presence cells non-empty
		bool daf1_valid;         //!< valid AND the daf1 normalization denominator is positive
	} PairStat;
	std::vector<PairStat> _pair_stats;      //!< upper-triangle linearization (via pair_index()) over the top-k feature set; filled by compute_pair_stats()
	std::vector<DIMTYPE> _pair_features;    //!< the top-k DAF0-ranked (global) feature indices the pairs are drawn from
	std::vector<std::size_t> _pair_order;   //!< indices into _pair_stats, valid pairs only, sorted decreasing pair-DAF0
	mutable std::size_t _pair_itersize;     //!< getFirstPairDAF()/getNextPairDAF() cursor position (index into _pair_order); 0 = iteration not started
	//! Upper-triangle linearization of pair (a,b), a<b<k, into a flat index over PairStat storage (k = the top-k feature-set size).
	static std::size_t pair_index(const std::size_t a, const std::size_t b, const std::size_t k) {assert(a<b && b<k); return a*k - a*(a+1)/2 + (b-a-1);}
};



template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::compute_stats(std::ostream& os)
{
	if(parent::results.empty()) return false;
	
	typename parent::CONSTRESULTSITER iter=parent::results.begin();
	// get n from first recorded sub (assumes all others have the same n)
	_n=iter->sub->get_n();
	_stats.resize(_n);
	for(DIMTYPE f=0;f<_n;f++) {
		_stats[f].freq_is=_stats[f].freq_isnot=_stats[f].mean_is=_stats[f].mean_isnot=_stats[f].stdev_is=_stats[f].stdev_isnot=0;
		_stats[f].daf_valid[0]=false;
		_stats[f].daf_valid[1]=false;
		_stats[f].daf_valid[2]=false;
	}
	_dstat.resize(_n+1);
	for(DIMTYPE ss=1;ss<=_n;ss++) {
		_dstat[ss].mean=_dstat[ss].stdev=_dstat[ss].freq=0;
		_dstat[ss].valid=false;
	}

	if(parent::output_normal()) {std::ostringstream sos; sos << std::endl << "compute_stats() 1/4.." << std::endl << std::flush; syncout::print(os,sos);}
	// gather initial stats
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		for(DIMTYPE f=0;f<_n;f++) {
			if(iter->sub->selected_raw(f)) {
				_stats[f].freq_is++;
				_stats[f].mean_is+=iter->value;
			} else {
				_stats[f].freq_isnot++;
				_stats[f].mean_isnot+=iter->value;
			}
		}
		_dstat[d].mean+=iter->value;
		_dstat[d].freq++;
	}
	
	// calculate mean values
	for(DIMTYPE f=0;f<_n;f++)
	{
		if(_stats[f].freq_is>0) _stats[f].mean_is/=_stats[f].freq_is;
		if(_stats[f].freq_isnot>0) _stats[f].mean_isnot/=_stats[f].freq_isnot;
	}
	for(DIMTYPE ss=1;ss<=_n;ss++) if(_dstat[ss].freq>0) {
		_dstat[ss].mean/=_dstat[ss].freq;
	}
	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 2/4.." << std::endl << std::flush; syncout::print(os,sos);}
		
	// gather more stats
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		for(DIMTYPE f=0;f<_n;f++) {
			if(iter->sub->selected_raw(f)) {
				_stats[f].stdev_is+=(_stats[f].mean_is-iter->value)*(_stats[f].mean_is-iter->value);
			} else {
				_stats[f].stdev_isnot+=(_stats[f].mean_isnot-iter->value)*(_stats[f].mean_isnot-iter->value);
			}
		}
		_dstat[d].stdev+=(_dstat[d].mean-iter->value)*(_dstat[d].mean-iter->value);
	}
	
	// calculate stddev values
	for(DIMTYPE f=0;f<_n;f++)
	{
		if(_stats[f].freq_is>0) _stats[f].stdev_is=sqrt(_stats[f].stdev_is/_stats[f].freq_is);
		// each guard must test the SAME frequency its line divides by: guarding the 'isnot'
		// line with freq_is would leave stdev_isnot unnormalized, or divide it by zero
		if(_stats[f].freq_isnot>0) _stats[f].stdev_isnot=sqrt(_stats[f].stdev_isnot/_stats[f].freq_isnot);
	}
	for(DIMTYPE ss=1;ss<=_n;ss++) if(_dstat[ss].freq>0) {
		_dstat[ss].stdev=sqrt(_dstat[ss].stdev/_dstat[ss].freq);
		_dstat[ss].valid=true;
	}
	
	
	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 3/4.." << std::endl << std::flush; syncout::print(os,sos);}
	// calculate DAF ranks
	for(DIMTYPE f=0;f<_n;f++)
	{
		// DAF0 needs only both partitions to be non-empty, but it deliberately shares DAF1's
		// validity test: a feature whose criterion values never vary at all carries no evidence,
		// so ranking it alongside features that do would be misleading
		if(_stats[f].freq_is>0 && _stats[f].freq_isnot>0 && (_stats[f].stdev_is>0 || _stats[f].stdev_isnot>0) ) 
		{
			RETURNTYPE freq=_stats[f].freq_is+_stats[f].freq_isnot;
			//DAF1
			_stats[f].daf[1]=(_stats[f].mean_is-_stats[f].mean_isnot)/( (_stats[f].freq_is/freq)*_stats[f].stdev_is + (_stats[f].freq_isnot/freq)*_stats[f].stdev_isnot );
			_stats[f].daf_valid[1]=true;
			//DAF0
			_stats[f].daf[0]=_stats[f].mean_is-_stats[f].mean_isnot;
			_stats[f].daf_valid[0]=true;
		} 
		else _stats[f].daf_valid[1]=_stats[f].daf_valid[0]=false;
	}
	// DAF2: like DAF0, but every probe's value is first divided by the stdev of all probes of
	// the SAME cardinality, which removes the systematic level difference between small and
	// large subsets. Probes of a cardinality whose stdev is 0 contribute nothing, so a feature
	// can end up with no usable probe on one side - its DAF2 is then 0 (but still marked
	// valid, unlike DAF0/DAF1). mi1/mi2/cnt1/cnt2 are indexed by FEATURE, 0.._n-1.
	std::vector<RETURNTYPE> mi1,mi2; mi1.resize(_n); mi2.resize(_n);
	std::vector<IDXTYPE> cnt1,cnt2; cnt1.resize(_n); cnt2.resize(_n);
	for(DIMTYPE f=0;f<_n;f++) {mi1[f]=mi2[f]=0; cnt1[f]=cnt2[f]=0;}
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		
		if(_dstat[d].stdev>0) {
			for(DIMTYPE f=0;f<_n;f++) {
				if(iter->sub->selected_raw(f)) {mi1[f]+=iter->value/_dstat[d].stdev; cnt1[f]++;}
				else {mi2[f]+=iter->value/_dstat[d].stdev; cnt2[f]++;}
			}
		}
	}	
	for(DIMTYPE f=0;f<_n;f++) {
		if(cnt1[f]==0 || cnt2[f]==0) _stats[f].daf[2]=0; else
		{
			mi1[f]/=cnt1[f];
			mi2[f]/=cnt2[f];
			_stats[f].daf[2]=mi1[f]-mi2[f];
		}
		_stats[f].daf_valid[2]=true;
	}

	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 4/4.." << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	
	// order features according to DAF0, DAF1 and DAF2
	for(unsigned int DAFidx=0; DAFidx<3; DAFidx++)
	{
		_order[DAFidx].clear();
		for(DIMTYPE f=0;f<_n;f++)
		{
			if(_stats[f].daf_valid[DAFidx]) {
				typename ORDERTYPE::iterator iter=_order[DAFidx].begin();
				while(iter!=_order[DAFidx].end() && _stats[f].daf[DAFidx]<_stats[*iter].daf[DAFidx]) iter++;
				_order[DAFidx].insert(iter,f);
			}
		}
		if(_order[DAFidx].size()<_n && parent::output_normal()) {std::ostringstream sos; sos << "WARNING: DAF"<<DAFidx<<" rank could not be computed for all features due to insufficient number of evaluated probe subsets." << std::endl << std::flush; syncout::print(os,sos);}
	}
	if(parent::output_normal()) {
		RETURNTYPE minfreq=_stats[0].freq_is; if(_stats[0].freq_isnot<_stats[0].freq_is) minfreq=_stats[0].freq_isnot;
		for(DIMTYPE f=1;f<_n;f++) {
			if(_stats[f].freq_isnot<minfreq) minfreq=_stats[f].freq_isnot;
			if(_stats[f].freq_is<minfreq) minfreq=_stats[f].freq_is;
		}
		{
			std::ostringstream sos; 
			sos << "Lowest feature evaluation frequency over all probes is "<<minfreq<<"." << std::endl;
			if(minfreq<100) sos << "WARNING: To ensure reliable DAF estimates the minimum feature evaluation frequency should be at least in the order of hundreds. More probes may be needed." << std::endl;
			sos << std::endl << std::flush;
			syncout::print(os,sos);
		}
	}

	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::print_stats(std::ostream& os)
{
	std::ostringstream sos;
	if(_n<1) {sos << "Result_Tracker_Feature_Stats:: Nothing to print out." << std::endl; syncout::print(os,sos); return false;}
	{
		sos << "Probe subset stats for each (valid) cardinality:"<< std::endl;
		for(DIMTYPE ss=1;ss<=_n;ss++) 
			if(_dstat[ss].valid) sos << " d: "<< ss << "  mean=" << _dstat[ss].mean << ", stddev=" << _dstat[ss].stdev << ", freq=" << _dstat[ss].freq << std::endl;
		sos << std::endl << "Feature stats over all probe subsets:"<< std::endl;
		for(DIMTYPE f=0;f<_n;f++) {
			sos << " F:"<<f<< " daf0 = "; if(_stats[f].daf_valid[0]) sos << _stats[f].daf[0]; else sos << "n/a"; 
			sos << ", daf1 = "; if(_stats[f].daf_valid[1]) sos << _stats[f].daf[1]; else sos << "n/a";
			sos << ", daf2 = "; if(_stats[f].daf_valid[2]) sos << _stats[f].daf[2]; else sos << "n/a";
			sos<<",  freq "<<_stats[f].freq_is<<"|"<<_stats[f].freq_isnot<<",  mean ";
			if(_stats[f].freq_is>0) sos<<_stats[f].mean_is; else sos<< "n/a";
			sos <<"|";
			if(_stats[f].freq_isnot>0) sos<<_stats[f].mean_isnot; else sos<< "n/a";
			sos<<", stdev ";
			if(_stats[f].freq_is>0) sos<<_stats[f].stdev_is; else sos<<"n/a";
			sos<<"|";
			if(_stats[f].freq_isnot>0) sos<<_stats[f].stdev_isnot; else sos << "n/a";
			sos<<std::endl;
		}
		sos<<std::endl;
	}
	syncout::print(os,sos);
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getFirstDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx) const
{
	assert(DAFidx<3);
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(_itersize_mutex);
#endif
	_itersize[DAFidx]=0;
	if(!_order[DAFidx].empty())	{
		feature=_order[DAFidx][0];
		assert(feature<_n);
		value=_stats[feature].daf[DAFidx];
		_itersize[DAFidx]=1;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getNextDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx) const
{
	assert(DAFidx<3);
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(_itersize_mutex);
#endif
	if(0<_itersize[DAFidx] && _itersize[DAFidx]<_order[DAFidx].size()) {
		feature=_order[DAFidx][_itersize[DAFidx]];
		assert(feature<_n);
		value=_stats[feature].daf[DAFidx];
		++_itersize[DAFidx];
		return true;
	}
	return false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::compute_pair_stats(const DIMTYPE top_k, std::ostream& os)
{
	_pair_stats.clear(); _pair_features.clear(); _pair_order.clear(); _pair_itersize=0;
	if(parent::results.empty() || _n<1 || _order[0].empty()) return false; // compute_stats() must have run first
	if(top_k<2) return false;
	const std::size_t K = std::min<std::size_t>(top_k, _order[0].size());
	if(K<2) return false;

	// the top-K DAF0-ranked features define the pair set; local index <-> global feature
	_pair_features.resize(K);
	for(std::size_t k=0;k<K;k++) _pair_features[k]=_order[0][k];

	// accumulation pass: only the both-present cell needs per-probe work; the
	// one-only and both-absent cells derive by inclusion-exclusion below
	const std::size_t P=K*(K-1)/2;
	std::vector<RETURNTYPE> both_freq(P,0), both_sum(P,0), both_sumsq(P,0);
	std::vector<RETURNTYPE> f_freq(K,0), f_sum(K,0), f_sumsq(K,0);
	RETURNTYPE t_freq=0, t_sum=0, t_sumsq=0;
	std::vector<std::size_t> present; present.reserve(K);
	for(typename parent::CONSTRESULTSITER iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		const RETURNTYPE v=iter->value;
		t_freq+=1; t_sum+=v; t_sumsq+=v*v;
		present.clear();
		for(std::size_t k=0;k<K;k++) if(iter->sub->selected_raw(_pair_features[k])) present.push_back(k);
		for(std::size_t i=0;i<present.size();i++) {
			const std::size_t a=present[i];
			f_freq[a]+=1; f_sum[a]+=v; f_sumsq[a]+=v*v;
			for(std::size_t j=i+1;j<present.size();j++) {
				const std::size_t ij=pair_index(a,present[j],K);
				both_freq[ij]+=1; both_sum[ij]+=v; both_sumsq[ij]+=v*v;
			}
		}
	}

	// derive the 2x2 cells and the pair indicators
	_pair_stats.resize(P);
	std::size_t valid_count=0;
	for(std::size_t a=0;a<K;a++) for(std::size_t b=a+1;b<K;b++) {
		const std::size_t ij=pair_index(a,b,K);
		PairStat &ps=_pair_stats[ij];
		ps.f1=_pair_features[a]; ps.f2=_pair_features[b];
		const RETURNTYPE fb=both_freq[ij], sb=both_sum[ij], qb=both_sumsq[ij];
		const RETURNTYPE f1o=f_freq[a]-fb, s1o=f_sum[a]-sb;       // f1 present, f2 absent
		const RETURNTYPE f2o=f_freq[b]-fb, s2o=f_sum[b]-sb;       // f2 present, f1 absent
		const RETURNTYPE fn=t_freq-f_freq[a]-f_freq[b]+fb;        // neither present
		const RETURNTYPE sn=t_sum-f_sum[a]-f_sum[b]+sb;
		const RETURNTYPE qn=t_sumsq-f_sumsq[a]-f_sumsq[b]+qb;
		ps.freq_both=fb; ps.freq_neither=fn;
		ps.daf0=ps.daf1=ps.synergy=0;
		ps.valid=(fb>0 && fn>0 && f1o>0 && f2o>0);
		ps.daf1_valid=false;
		if(ps.valid) {
			const RETURNTYPE mb=sb/fb, mn=sn/fn, m1o=s1o/f1o, m2o=s2o/f2o;
			ps.daf0=mb-mn;
			ps.synergy=mb+mn-m1o-m2o;
			RETURNTYPE vb=qb/fb-mb*mb; if(vb<0) vb=0;             // guard FP negatives
			RETURNTYPE vn=qn/fn-mn*mn; if(vn<0) vn=0;
			const RETURNTYPE denom=(fb/(fb+fn))*sqrt(vb) + (fn/(fb+fn))*sqrt(vn);
			if(denom>0) {ps.daf1=ps.daf0/denom; ps.daf1_valid=true;}
			++valid_count;
		}
	}

	// order valid pairs by decreasing pair-DAF0 (deterministic tie-break on feature indices)
	_pair_order.reserve(valid_count);
	for(std::size_t ij=0;ij<P;ij++) if(_pair_stats[ij].valid) _pair_order.push_back(ij);
	std::sort(_pair_order.begin(),_pair_order.end(),[this](const std::size_t x, const std::size_t y)->bool {
		if(_pair_stats[x].daf0!=_pair_stats[y].daf0) return _pair_stats[x].daf0>_pair_stats[y].daf0;
		if(_pair_stats[x].f1!=_pair_stats[y].f1) return _pair_stats[x].f1<_pair_stats[y].f1;
		return _pair_stats[x].f2<_pair_stats[y].f2;
	});

	if(parent::output_normal()) {
		std::ostringstream sos;
		sos << "compute_pair_stats(): pairwise stats over the top " << K << " DAF0-ranked features (" << P << " pairs, " << valid_count << " valid)." << std::endl;
		if(valid_count<P) sos << "WARNING: " << (P-valid_count) << " pairs have an empty presence cell (insufficient probes); they carry no pair-DAF value." << std::endl;
		sos << std::flush;
		syncout::print(os,sos);
	}
	return valid_count>0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::print_pair_stats(std::ostream& os, const std::size_t top_pairs)
{
	std::ostringstream sos;
	if(_pair_order.empty()) {sos << "Result_Tracker_Feature_Stats:: No pairwise stats to print out." << std::endl; syncout::print(os,sos); return false;}
	const std::size_t count = (top_pairs==0 || top_pairs>_pair_order.size()) ? _pair_order.size() : top_pairs;
	sos << "Pairwise DAF stats over the top " << _pair_features.size() << " DAF0-ranked features (showing " << count << " of " << _pair_order.size() << " valid pairs, best pair-DAF0 first):" << std::endl;
	for(std::size_t i=0;i<count;i++) {
		const PairStat &ps=_pair_stats[_pair_order[i]];
		sos << " P:("<<ps.f1<<","<<ps.f2<<") pairdaf0 = " << ps.daf0;
		sos << ", pairdaf1 = "; if(ps.daf1_valid) sos << ps.daf1; else sos << "n/a";
		sos << ", synergy = " << ps.synergy;
		sos << ",  freq both|neither " << ps.freq_both << "|" << ps.freq_neither << std::endl;
	}
	sos << std::endl;
	syncout::print(os,sos);
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getFirstPairDAF(RETURNTYPE &daf0, RETURNTYPE &daf1, bool &daf1_valid, RETURNTYPE &synergy, DIMTYPE &f1, DIMTYPE &f2) const
{
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(_itersize_mutex);
#endif
	_pair_itersize=0;
	if(!_pair_order.empty()) {
		const PairStat &ps=_pair_stats[_pair_order[0]];
		daf0=ps.daf0; daf1=ps.daf1; daf1_valid=ps.daf1_valid; synergy=ps.synergy; f1=ps.f1; f2=ps.f2;
		_pair_itersize=1;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getNextPairDAF(RETURNTYPE &daf0, RETURNTYPE &daf1, bool &daf1_valid, RETURNTYPE &synergy, DIMTYPE &f1, DIMTYPE &f2) const
{
#ifdef FST_THREADED
	std::unique_lock<std::mutex> lock(_itersize_mutex);
#endif
	if(0<_pair_itersize && _pair_itersize<_pair_order.size()) {
		const PairStat &ps=_pair_stats[_pair_order[_pair_itersize]];
		daf0=ps.daf0; daf1=ps.daf1; daf1_valid=ps.daf1_valid; synergy=ps.synergy; f1=ps.f1; f2=ps.f2;
		++_pair_itersize;
		return true;
	}
	return false;
}


} // namespace
#endif // FSTRESULTTRACKERFEATURESTATS_H ///:~
