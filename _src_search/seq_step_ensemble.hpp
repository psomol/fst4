#ifndef FSTSEARCHSEQSTEPENSEMBLE_H
#define FSTSEARCHSEQSTEPENSEMBLE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    seq_step_ensemble.hpp
   \brief   Implements voting ensemble selection step in sequential search type of methods
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
#include <sstream>
#include <list>
#include <vector>
#include <map>
#include "error.hpp"
#include "global.hpp"
#include "seq_step.hpp"
#include "criterion.hpp"
#include "candidate_evaluator.hpp"
#include "candidate_evaluator_sequential.hpp"

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

//----------------------------------------------------------------------------

/*! \brief Implements voting ensemble selection step in sequential search type of methods
to possibly improve robustness and stability of feature selection result.

\details Criteria ensembles may improve generalization and stability properties of the selected feature subset, see
"P. Somol, J. Grim,  and P. Pudil.  Criteria Ensembles in Feature Selection.  In Proc. MCS, LNCS 5519, pages 304–313.  Springer, 2009."
The idea is to reduce possible over-training, i.e., excessive result adjustment to particular
criterion properties. By employing more different criteria the result is likely to be
more robust in different contexts. Criteria ensembles as implemented here
are based on voting about feature preferences. In sequential algorithm step, feature
candidates are ordered separately according to each considered criterion. The various
orderings are then joined (the position indices of a feature are summed over the
criteria, which ranks the features the same way as averaging them would) to produce
the final feature ordering. The best feature is then selected for addition to the current working subset.
Note that this mechanism allows to use completely unrelated criteria, as the various criterion
values are never combined - the only information that is combined is position index in ordered
feature lists. This is advantageous as it enables combinations of filter and wrapper criteria
(that yield values from different intervals) etc. Note, however, that the value
obtained as result of ensemble evaluation is not usable for assessing the whole feature
subset - it can be used only within the selection step to identify one feature
(can be extended to feature c-tuple) for next inclusion/removal. For this reason
it is also necessary to employ one single criterion to be used by selection algorithm
for evaluating current subsets (and thus directing next search steps). This single
criterion is used in the same way as in other FST4 Sequential_Step implementations,
i.e. it must be passed in evaluate_candidates call. The criteria that form the ensemble
are to be passed to Sequential_Step_Ensemble constructor.

\note Single feature ordering implementation only. Support for generalized search
      (evaluation of candidate c-tuples instead of single features) may be implemented in future,
      but so far is not: evaluate_candidates() THROWS fst_error for any generalization level
      other than 1.

\note Like Sequential_Step_Straight this class serves both the sequential and the threaded
      execution model: the per-candidate evaluations of each ensemble member run through the
      injected Candidate_Evaluator, one batch per member criterion (default: sequential
      in-line evaluation; inject Candidate_Evaluator_Threaded for thread-pooled batches).
      The engine is typed on the abstract Criterion because that is how the ensemble members
      are held. Results are identical to sequential for any thread count: per-result callbacks
      fire in candidate order on the master, so the per-member ordered lists — and therefore
      the vote combination — come out the same whichever engine runs the batches (member
      criteria are deterministic and stateless across evaluate() calls, so evaluating
      member-by-member instead of candidate-by-candidate yields the same values).

\note One instance keeps a single set of iteration cursors and scratch lists, so a
      Sequential_Step_Ensemble object must not be used by two searches at the same time.
      Give each concurrent search its own instance.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Sequential_Step_Ensemble : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< base class alias
	typedef std::shared_ptr<CRITERION> PCriterion;             //!< shared pointer to the single criterion passed into evaluate_candidates() and used to (re-)score the winning subset -- distinct from the ensemble member criteria
	typedef std::shared_ptr<Criterion<RETURNTYPE,SUBSET> > PAbstractCriterion; //!< shared pointer to one ensemble member, held via the abstract Criterion base so unrelated (filter/wrapper) criterion types can be mixed in one ensemble
	typedef std::vector<PAbstractCriterion> PAbstractCriteria; //!< the ordered collection of ensemble member criteria
	typedef std::shared_ptr<PAbstractCriteria> PEnsembleCriteria; //!< shared pointer to the ensemble member collection, as supplied to the constructor
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to a feature subset
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,Criterion<RETURNTYPE,SUBSET> > EVALUATORENGINE; //!< typed on the abstract Criterion (the ensemble members' type)
	typedef std::shared_ptr<EVALUATORENGINE> PEvaluatorEngine; //!< shared pointer to the candidate-evaluation engine (see candidate_evaluator.hpp)
	/*! \brief Constructs the ensemble step over the given member criteria.
	    \param ensemble the (non-null) collection of member criteria whose votes are combined each step; each member
	           is expected to be already initialize()d before use, like \p crit passed to evaluate_candidates()
	    \param engine the candidate-evaluation engine used to run each member's per-candidate batch; if null
	           (the default), a Candidate_Evaluator_Sequential is constructed, i.e. sequential in-line evaluation */
	Sequential_Step_Ensemble(const PEnsembleCriteria ensemble, PEvaluatorEngine engine=PEvaluatorEngine()):Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),_engine(engine),track(false) {assert(ensemble); _ensemble=ensemble; if(!_engine) _engine.reset(new Candidate_Evaluator_Sequential<RETURNTYPE,DIMTYPE,SUBSET,Criterion<RETURNTYPE,SUBSET> >()); notify("Sequential_Step_Ensemble constructor.");}
	virtual ~Sequential_Step_Ensemble() {notify("Sequential_Step_Ensemble destructor.");}

	// NOTE: crit (as well as all criteria in ensemble) is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout) override; //!< chooses among subsets offered by sub->get*CandidateSubset()

	//! Writes a human-readable identification of the step, incl. the engine's thread count (if threaded) and any attached result tracker.
	virtual std::ostream& print(std::ostream& os) const override {
		if(_engine->is_threaded()) os << "Sequential_Step_Ensemble(threads="<<_engine->get_max_threads()<<")";
		else os << "Sequential_Step_Ensemble()";
		if(parent::result_tracker_active()) os << " with " << *parent::_tracker;
		return os;
	}
protected:
	/*! \brief Builds, for each ensemble member, the ordered (best-first) list of candidate subsets in _lists,
	    and seeds _final with one zero-valued entry per candidate feature.

	    \details Two passes: first the master enumerates sub's candidate subsets once (single-feature
	    add/remove, i.e. generalization level 1) to register every candidate's temporary feature in _final
	    and count the candidates. Then, for each ensemble member in turn, the candidates are re-enumerated
	    and evaluated as one batch through _engine; the per-result callback performs a descending-value
	    sorted insertion into that member's list in _lists, so the list is ordered from the member's most to
	    least preferred candidate regardless of evaluation/callback order (results fire on the master in
	    candidate order for any thread count).

	    \param sub the subset whose candidate subsets are ordered
	    \return false if no candidate subsets exist, or if candidate enumeration or any member's evaluation fails */
	inline bool order_candidates(const PSubset sub);

	PEvaluatorEngine _engine; //!< candidate-evaluation engine driving the per-member candidate batches
	PEnsembleCriteria _ensemble; //!< the ensemble member criteria, as supplied to the constructor
	
	//! Nested class to hold feature/subset candidate info in the course of the ensemble voting process
	class SubsetCandidate {
	public:
		//! Constructs a candidate entry, taking a deep (owned) copy of \p sub -- the caller's subset object is mutated/reused across candidates during enumeration, so it cannot be aliased here.
		SubsetCandidate(const RETURNTYPE value, const PSubset sub, const DIMTYPE tfeature) : _value(value), _sub(new SUBSET(*sub)), _tempfeature(tfeature) {}
		//! Copy constructor -- unlike the constructor above this SHARES (does not deep-copy) the source's _sub, since by this point the source subset is a stable, privately-owned snapshot rather than a reused enumeration object.
		SubsetCandidate(const SubsetCandidate& fc) : _value(fc._value), _sub(fc._sub), _tempfeature(fc._tempfeature) {}
		//! Dual-purpose: in a member's per-candidate list (_lists) this is the raw value that member criterion returned for _sub; in a _final entry it is instead the accumulated vote (summed dense rank across all members, reset to 0.0 at the start of evaluate_candidates() and filled by its "collect ordering votes" loop) -- lower means more preferred.
		RETURNTYPE _value;
		const PSubset _sub;         //!< owned snapshot of the candidate subset (see the two constructors above)
		const DIMTYPE _tempfeature; //!< copy of sub->getFirstTemporaryFeature() at the time this candidate was constructed -- the feature this candidate would add/remove
	};
	typedef std::shared_ptr<SubsetCandidate> PSubsetCandidate; //!< shared pointer to a SubsetCandidate
	typedef std::list<SubsetCandidate> CANDIDATELIST;          //!< one ensemble member's candidates, kept sorted best (highest _value) first by order_candidates()
	typedef std::shared_ptr<CANDIDATELIST> PCANDIDATELIST;     //!< shared pointer to a CANDIDATELIST
	typedef std::vector<PCANDIDATELIST> CANDIDATELISTS;        //!< one CANDIDATELIST per ensemble member, indexed the same as _ensemble
	CANDIDATELISTS _lists; //!< per-member ordered candidate lists, rebuilt by order_candidates() on every evaluate_candidates() call
	typedef std::map<DIMTYPE, PSubsetCandidate> FINALVALUES;   //!< maps each candidate feature to its tallied ensemble vote (see SubsetCandidate::_value); iteration order is thus ascending feature index
	FINALVALUES _final; //!< the per-feature vote tally built by evaluate_candidates() from _lists; the entry with the lowest (best) vote determines the step's chosen subset

	typename CANDIDATELISTS::iterator oiter; //!< iterates _lists, i.e. over ensemble members
	typename CANDIDATELIST::iterator iter;   //!< iterates one member's ordered candidate list (an element of _lists)
	typename FINALVALUES::iterator fiter;    //!< iterates _final, i.e. over candidate features and their tallied votes

private:
	std::unique_ptr<SUBSET> tmp_step_sub; //!< holds the winning candidate subset while evaluate_candidates() assembles it, before it is copied into the caller's \p sub (lazily allocated/grown to sub's feature count)
	bool track; //!< cached parent::result_tracker_active() for the duration of the current evaluate_candidates() call
};


template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::order_candidates(const PSubset sub)
{
	assert(sub);
	assert(_engine);
	assert(_ensemble);
	assert(!_ensemble->empty());
	assert(_ensemble->size()==_lists.size());

	_final.clear();

	// first pass (master, no criterion evaluation): register each candidate's temporary
	// feature in _final and count the candidates (the threaded engine sizes its result
	// storage from the count)
	DIMTYPE _tfeat;
	DIMTYPE _candidates=0;
	bool b=sub->getFirstCandidateSubset(1,false/*reverse*/); if(!b) return false; // no candidate subsets can be traversed for whatever reason
	while(b) {
		if(!sub->getFirstTemporaryFeature(_tfeat)) return false; // just one exists due to the getFirstCandidateSubset(1,false/*reverse*/) call above
		assert(_tfeat<sub->get_n());
		PSubsetCandidate _cs(new SubsetCandidate(0.0,sub,_tfeat));
		_final.insert( make_pair(_tfeat,_cs));
		++_candidates;
		b=sub->getNextCandidateSubset();
	}
	if(_candidates==0) return false;

	// per-member engine batches: candidates are re-enumerated for each ensemble member and
	// evaluated through the injected engine; the per-result callback (fired on the master in
	// candidate order) does the sorted insertion into the member's ordered list, so the
	// resulting lists are the same for any thread count. The candidate's temporary feature
	// rides along as the engine's feature tag.
	for(typename PAbstractCriteria::size_type j=0;j<_ensemble->size();j++)
	{
		PCANDIDATELIST lst=_lists[j];
		typename EVALUATORENGINE::ResultCallback on_result=[lst](const RETURNTYPE val, const PSubset &csub, const DIMTYPE tfeat) {
			typename CANDIDATELIST::iterator it;
			for(it=lst->begin(); it!=lst->end() && val<it->_value; it++);
			lst->insert(it,SubsetCandidate(val,csub,tfeat));
		};
		_engine->initialize((*_ensemble)[j],_candidates,on_result);
		b=sub->getFirstCandidateSubset(1,false/*reverse*/); if(!b) return false;
		while(b) {
			if(!sub->getFirstTemporaryFeature(_tfeat)) return false;
			if(!_engine->add_candidate(sub,_tfeat)) return false;
			b=sub->getNextCandidateSubset();
		}
		if(!_engine->collect()) return false;
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(sub->get_n()>0);
	assert(crit);
	assert(_ensemble);
	assert(!_ensemble->empty());
	if(_generalization_level!=1) throw FST::fst_error("Sequential_Step_Ensemble currently does not support generalization level other than 1.");
	track=parent::result_tracker_active();
	
	_lists.reserve(_ensemble->size());
	while(_lists.size()<_ensemble->size()) _lists.push_back(PCANDIDATELIST(new CANDIDATELIST));
	assert(_lists.size()==_ensemble->size());
	for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++) (*oiter)->clear();

	if(!order_candidates(sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE order_candidates() sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}

	assert(_final.size()>0);
	for(fiter=_final.begin();fiter!=_final.end();fiter++) (*fiter).second->_value=0.0;

	// Collect ordering votes: walk each member's best-first list and add the candidate's
	// RANK (1 = the member's most preferred) to that feature's tally in _final. Ranking is
	// DENSE and ties-aware - the rank advances only when the criterion value actually drops,
	// so candidates a member scored equally receive the same rank and the next distinct
	// value gets rank+1 (not rank+number-of-tied). A feature's tally therefore lies between
	// (number of members) for a unanimous first choice and (number of members)*(number of
	// distinct values) at worst, and LOWER is better.
	RETURNTYPE _lastval;
	DIMTYPE _curposition;
	for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++)
	{
		iter=(*oiter)->begin();
		_curposition=1;
		_lastval=iter->_value;
		_final[iter->_tempfeature]->_value+=_curposition;
		iter++;
		while(iter!=(*oiter)->end())
		{
			if(iter->_value<_lastval) {_curposition++; _lastval=iter->_value;}
			_final[iter->_tempfeature]->_value+=_curposition;
			iter++;
		}
	}
	
	if(!tmp_step_sub || tmp_step_sub->get_n()<sub->get_n()) tmp_step_sub.reset(new SUBSET(sub->get_n()));
	
	// Pick the candidate with the LOWEST vote tally. _lastval doubles as the
	// "nothing picked yet" sentinel: a real tally is at least 1 per ensemble member, so the
	// initial 0.0 (tested as <1.0) makes the first entry of _final the provisional winner,
	// after which only a strictly smaller positive tally displaces it. _final is a map keyed
	// by feature index, so when several features tie on the tally the LOWEST feature index
	// wins - the step is deterministic.
	_lastval=0.0;
	_curposition=0;
	for(fiter=_final.begin();fiter!=_final.end();fiter++) 
	{
		if(_lastval<1.0 || ( (*fiter).second->_value < _lastval && (*fiter).second->_value > 0.0 ))
		{
			_lastval=(*fiter).second->_value; 
			_curposition=(*fiter).second->_tempfeature;
			tmp_step_sub->stateless_copy(*((*fiter).second->_sub)); // keep the best candidate's subset
		}
	}
	sub->stateless_copy(*tmp_step_sub);
	assert(sub->get_d()!=0);
	
	if(!crit->evaluate(result,sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	if(track) parent::_tracker->add(result,sub);

	if(parent::output_detailed()) {
		std::ostringstream sos; 
		sos << "---------------- "<< std::endl;
		sos << "candidate lists: "<< std::endl;
		int lst=0;
		for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++)
		{
			sos << lst++ << ": ";
			for(iter=(*oiter)->begin();iter!=(*oiter)->end();iter++) sos << iter->_tempfeature << "," <<iter->_value << " | ";
			sos << std::endl;
		}
		sos << "final list: "<<std::endl;
		sos << "   ";
		assert(_final.size()>0);
	
		for(fiter=_final.begin();fiter!=_final.end();fiter++) sos << (*fiter).first << ") " << (*fiter).second->_value << " | ";
		sos << std::endl;
		sos << "verified crit value=" << result << std::endl;
		sos << std::endl << std::flush;	
		syncout::print(os,sos);
	}
	
	return true;
}


} // namespace
#endif // FSTSEARCHSEQSTEPENSEMBLE_H ///:~
