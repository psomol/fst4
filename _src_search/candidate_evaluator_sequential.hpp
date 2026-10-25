#ifndef FSTCANDIDATEEVALUATORSEQUENTIAL_H
#define FSTCANDIDATEEVALUATORSEQUENTIAL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    candidate_evaluator_sequential.hpp
   \brief   Implements sequential in-line evaluation of a criterion on a set of subset candidates
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    July 2026
   \version 4.2.0

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
#include "error.hpp"
#include "global.hpp"
#include "candidate_evaluator.hpp"

namespace FST {

/*! \brief Implements sequential in-line evaluation of a criterion on a set of subset candidates

	The sequential counterpart of Candidate_Evaluator_Threaded: candidates are
	evaluated immediately in the caller's thread. In batch mode add_candidate()
	evaluates on the spot and fires the per-result callback synchronously, so
	evaluations and the caller's own output stay strictly interleaved in add
	order. Only the running best candidate is stored - memory does not grow with
	the batch size. The criterion is used directly (never cloned), so no
	thread-safety or clone-staleness concerns apply here.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Candidate_Evaluator_Sequential : public Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> {
public:
	typedef Candidate_Evaluator<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent; //!< the abstract Candidate_Evaluator interface this class implements
	typedef typename parent::PCriterion PCriterion;             //!< shared pointer to the criterion this engine evaluates candidates against
	typedef typename parent::PSubset PSubset;                   //!< shared pointer to a candidate feature subset
	typedef typename parent::PResultTracker PResultTracker;     //!< shared pointer to the result tracker fed during evaluate_stream()
	typedef typename parent::Generator Generator;                //!< see Candidate_Evaluator::Generator
	typedef typename parent::ResultCallback ResultCallback;      //!< see Candidate_Evaluator::ResultCallback
	typedef typename parent::BestCallback BestCallback;          //!< see Candidate_Evaluator::BestCallback
	typedef typename parent::BetterPredicate BetterPredicate;    //!< see Candidate_Evaluator::BetterPredicate

	Candidate_Evaluator_Sequential() : _best_available(false), _bestval(RETURNTYPE()), _bestfeature(0), _evaluated(0), _failed(false) {notify("Candidate_Evaluator_Sequential constructor.");}
	virtual ~Candidate_Evaluator_Sequential() {notify("Candidate_Evaluator_Sequential destructor.");}

	virtual unsigned int get_max_threads() const override {return 1;}
	virtual bool is_threaded() const override {return false;}

	virtual void initialize(const PCriterion &crit, const DIMTYPE max_candidates, ResultCallback on_result=ResultCallback()) override
	{
		assert(crit);
		_crit=crit;
		_on_result=on_result;
		_best_available=false;
		_evaluated=0;
		_failed=false;
	}

	virtual bool add_candidate(const PSubset &sub, const DIMTYPE feature=0) override
	{
		assert(_crit);
		assert(sub);
		RETURNTYPE val;
		if(!_crit->evaluate(val,sub)) {_failed=true; return false;}
		++_evaluated;
		if(_on_result) _on_result(val,sub,feature);
		if(!_best_available || val>_bestval) {
			if(!_bestsub || _bestsub->get_n()!=sub->get_n()) _bestsub.reset(new SUBSET(sub->get_n()));
			_bestsub->stateless_copy(*sub);
			_bestval=val;
			_bestfeature=feature;
			_best_available=true;
		}
		return true;
	}

	//! nothing is left pending in this engine (every candidate was evaluated inside add_candidate()), so collect() only reports the batch outcome: true if at least one candidate was evaluated and none failed
	virtual bool collect() override {return _evaluated>0 && !_failed;}

	virtual bool getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) override
	{
		if(!_best_available || _failed) return false; // a failed batch hands out nothing, as in the threaded engine
		result=_bestval;
		sub=_bestsub;
		feature=_bestfeature;
		return true;
	}

	virtual bool evaluate_stream(Generator generator, const PCriterion &crit, const PResultTracker tracker, BetterPredicate better, BestCallback on_best, RETURNTYPE &result, const PSubset bestsub) override
	{
		assert(generator);
		assert(crit);
		assert(better);
		assert(bestsub);
		assert(bestsub->get_n()>0);
		PSubset tmp_sub(new SUBSET(bestsub->get_n()));
		RETURNTYPE val;
		bool best_available=false;
		unsigned long long ordinal=0; // 64-bit to match BestCallback (see the LLP64 note there)
		while(generator(*tmp_sub)) {
			if(!crit->evaluate(val,tmp_sub)) return false;
			if(tracker) tracker->add(val,tmp_sub);
			if(!best_available || better(val,tmp_sub,result,bestsub)) {
				result=val;
				bestsub->stateless_copy(*tmp_sub);
				best_available=true;
				if(on_best) on_best(-1,ordinal,val,tmp_sub);
			}
			++ordinal;
		}
		return best_available;
	}

	virtual std::ostream& print(std::ostream& os) const override {os << "Candidate_Evaluator_Sequential()"; return os;}

protected:
	PCriterion _crit;              //!< the criterion supplied to initialize(), evaluated in-line by add_candidate()
	ResultCallback _on_result;     //!< the per-result callback supplied to initialize(), fired synchronously from add_candidate()
	PSubset _bestsub;               //!< the best candidate seen so far in the current batch (only the best is retained, not the whole batch)
	bool _best_available;          //!< true once at least one candidate has been evaluated successfully in the current batch
	RETURNTYPE _bestval;           //!< criterion value of _bestsub
	DIMTYPE _bestfeature;          //!< the feature tag passed to add_candidate() for _bestsub
	DIMTYPE _evaluated;            //!< number of candidates evaluated in the current batch (collect() requires at least one)
	bool _failed;                  //!< set when a criterion evaluate() call returned false; makes collect() and getBestResult() report failure (the threaded engine's _batch_failed semantics)
};

} // namespace
#endif // FSTCANDIDATEEVALUATORSEQUENTIAL_H ///:~
