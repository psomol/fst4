#ifndef FSTCRITERIONEXTERNVALUE_H
#define FSTCRITERIONEXTERNVALUE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    criterion_extern_value.hpp
   \brief   External-tool criterion: a user-supplied command computes the criterion value over the VALUE line protocol
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
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"
#include "criterion.hpp"
#include "extern_bridge.hpp"

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

/*! \brief External-tool criterion: a user-supplied command computes the
    criterion value itself - unlike wrapper-extern, which exchanges labels and
    computes accuracy on the FST4 side.

    One PERSISTENT child process per criterion instance (spawned lazily on
    first use; clones spawn their own, so threaded evaluation is
    isolation-safe), speaking a line protocol on stdin/stdout:

        -> DATA <N> <D> <K>\n   followed by N lines  <label> <v1> ... <vD>
        <- OK\n                 (anything else fails the evaluation, loudly)
        -> VALUE <d> <f1> ... <fd>\n   (0-based RAW feature indices, ascending)
        <- one line: the criterion value (parsed as a strict full-line double)

    The DATA block carries the FULL-width TRAIN part of the current split
    (the filter convention: the criterion sees training data only) and is
    sent once per split - re-sent whenever initialize() is called (the
    pipeline's refresh contract, same as filter model re-estimation) or the
    child was (re)spawned. Each evaluation then costs one VALUE request of d
    integers: the tool holds the data (or better, per-feature SUFFICIENT
    STATISTICS computed at DATA time) and answers from the indices - O(d)
    per request instead of re-streaming N*d values (the cost model that
    matters under B&B's thousands of evaluations). Raw indices also give the
    tool feature IDENTITIES (per-feature caching, acquisition-cost-aware
    measures). Values are exchanged as %.17g decimal (bit round-trip for
    doubles). A reference implementation ships in
    tests/extern_value_meandiff.awk (the sufficient-statistics pattern).

    NO-DATA MODE (set_no_data(), the CLI's `--extern-no-data`): the DATA block is
    suppressed entirely and the tool answers index-only VALUE requests from logic
    of its own - cost tables, simulations, external experiments: FST4 as a
    black-box SET-FUNCTION optimizer over an abstract feature universe. A dataset
    is then only a source of the universe size, and the fst4 CLI does not need one
    at all in this mode: `--features N` replaces `--data` and supplies a
    feature-only accessor of N features and no classes. The three-layer design:
    wrapper-extern = external CLASSIFIER, extern-value with DATA = external
    DATA-DEPENDENT CRITERION (the DATA block is the split-synchronization
    mechanism - only FST4 knows the current split, and a self-loading tool would
    leak held-out data), extern-value with no data = external SET FUNCTION.

    VALUE SEMANTICS: FST4 maximizes - larger values must mean better subsets
    (negate error-style measures inside the tool). There is NO range or scale
    requirement beyond that: criterion values are only ever compared within
    the same search, never across criteria (built-in filters return unbounded
    distances, wrappers [0,1] accuracies — both coexist for the same reason).
    Values must be FINITE doubles (non-finite responses fail loudly, see
    parse_value) and deterministic for identical requests. Flags expressed in
    value units (`--tracker-margin`, `--regularize-margins`) are in the external
    measure's own units, as with every criterion.

    NORMALIZATION follows the standard FST4 convention: the streamed values
    are the accessor's in-memory data AFTER the global `--scaler` (applied
    per-feature over the whole dataset at load) — exactly what built-in
    criteria see; the bridge adds NO internal per-training standardization.
    If the tool normalizes further, beware the monotonicity assertion:
    per-feature-independent normalization is safe for per-feature-sum
    measures, but normalization coupling the streamed columns (subset-level
    variance/totals) can silently break the asserted monotonicity and void
    the B&B optimality guarantee.

    MONOTONICITY AND BRANCH & BOUND: is_monotonic() returns what the user
    asserted via set_monotonic() (the CLI's `--extern-monotonic`; default
    false). Asserting it admits the criterion to the optimal B&B searches —
    THE OPTIMALITY GUARANTEE THEN RESTS ENTIRELY ON THE ASSERTION being true
    of the external measure (removing a feature must never increase the
    value). A false assertion yields silently suboptimal "optimal" results.
    Without the assertion, Search_Branch_And_Bound prints its warning (and
    the fst4 CLI refuses bb-* searches outright).

    All LOUD CAVEATS of the extern mechanism apply as for wrapper-extern
    (classifier_extern.hpp): USER-SUPPLIED CODE EXECUTION through the system
    shell; REPRODUCIBILITY GUARANTEES END AT THE BRIDGE (a stochastic
    external measure breaks run- and thread-reproducibility AND, under B&B,
    the optimality logic itself); no timeout machinery (extern_bridge.hpp). */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Extern_Value : public Clonable_Impl<Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Criterion<RETURNTYPE,SUBSET> > {
public:
	typedef Criterion<RETURNTYPE,SUBSET> parent; //!< base class: the abstract criterion interface
	typedef Clonable_Impl<Criterion_Extern_Value,parent> parent_impl; //!< implements clone(); sharing/stateless_clone throw in Criterion
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;  //!< shared pointer to the data accessor supplying the DATA block's TRAIN part
	typedef std::shared_ptr<SUBSET> PSubset;              //!< shared pointer to the feature subset being evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;     //!< raw feature-vector pointer type, as sent full-width by send_row()
	Criterion_Extern_Value() : _bridge("Criterion_Extern_Value"), _monotonic(false), _no_data(false), _data_stale(true), _classes(0), _features(0), _subfeatures(0) {notify("Criterion_Extern_Value constructor.");}
	virtual ~Criterion_Extern_Value() {notify("Criterion_Extern_Value destructor.");} // the bridge destructor shuts the child down

	void set_command(const std::string &cmd) {_bridge.set_command(cmd);} //!< sets the external command line (`--extern-cmd`); required before the first evaluate()
	const std::string& get_command() const {return _bridge.get_command();} //!< returns the currently configured external command line
	void set_monotonic(const bool m) {_monotonic=m;} //!< the user's assertion (`--extern-monotonic`); see the class note
	virtual bool is_monotonic() const override {return _monotonic;}
	void set_no_data(const bool nd) {_no_data=nd;} //!< `--extern-no-data`: suppress the DATA block - pure set-function optimization, the tool answers index-only VALUE requests from its own logic

	//! \brief Records the data shape and marks the tool's stored DATA stale; the criterion then
	//! evaluates on \p da's TRAIN part. Must be called before the first evaluate(), and again
	//! whenever the effective split changes -- that is what makes the next evaluate() re-send
	//! the DATA block. Throws fst_error when no command has been set, or (unless no-data mode is
	//! in effect) when \p da has fewer than 2 classes.
	//! \param da the accessor the criterion reads; its currently designated TRAIN part becomes
	//!        the DATA block, and its feature count the size of the feature universe.
	//! \param da_for_evaluate accepted for signature compatibility with the criteria that
	//!        evaluate on a second accessor; this criterion has no such split and ignores it.
	void initialize(const PDataAccessor da, const PDataAccessor da_for_evaluate=PDataAccessor());
	/*! \brief Evaluates \p sub by one VALUE request to the external tool.

	    Sends "VALUE d f1 ... fd" with the 0-based RAW feature indices in ascending subset order
	    and reads back one line holding the value. Before that, in the DATA modes, the DATA block
	    is (re)sent whenever it is stale -- after an initialize(), and after the child process was
	    spawned or respawned, since a fresh child holds nothing.

	    \return false when the subset is empty, or when the current split's TRAIN part holds no
	            patterns (so there is no DATA to send). Everything else that can go wrong throws:
	            a rejected DATA block, a non-numeric or non-finite value line, a child that died.
	            The value itself is whatever the tool returned -- FST4 does not check its range. */
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) override;

	virtual std::ostream& print(std::ostream& os) const override {os << "Criterion_Extern_Value(cmd=\"" << _bridge.get_command() << "\", monotonic=" << (_monotonic?"asserted":"no") << (_no_data?", no-data":"") << ")"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! Copy-constructor; spawns its own fresh bridge (clones never share pipes) and copies the recorded data shape.
	Criterion_Extern_Value(const Criterion_Extern_Value& cev) :
		_bridge("Criterion_Extern_Value"), // clones NEVER share pipes - each fresh bridge spawns its own child lazily
		_monotonic(cev._monotonic),
		_no_data(cev._no_data),
		_data_stale(true), // the clone's own child holds no DATA yet
		_da(cev._da),
		_classes(cev._classes),
		_features(cev._features),
		_subfeatures(cev._subfeatures),
		_index(cev._index)
	{
		_bridge.set_command(cev._bridge.get_command());
		notify("Criterion_Extern_Value copy-constructor.");
	}
protected:
	Extern_Bridge _bridge;         //!< persistent child process + line I/O (per instance; never shared by clones)
	bool _monotonic;               //!< the user's cardinality-monotonicity assertion
	bool _no_data;                 //!< `--extern-no-data`: never send DATA (the tool is a pure set function over index sets)
	bool _data_stale;              //!< the tool's stored DATA no longer matches the current split (set by initialize() and child respawn)
	PDataAccessor _da;             //!< the data accessor whose TRAIN part feeds the DATA block
	DIMTYPE _classes;              //!< class count recorded by initialize(), from _da->getNoOfClasses()
	DIMTYPE _features;             //!< raw feature count recorded by initialize(), from _da->getNoOfFeatures()
	DIMTYPE _subfeatures;          //!< d = current subset size (number of valid entries in _index)
	std::vector<DIMTYPE> _index;   //!< subset position -> raw feature index (first _subfeatures entries valid)

	bool send_data_block();        //!< sends one full-width DATA block over the current split's TRAIN part (returns false on an empty train part)
	void send_row(const DIMTYPE label, const PPattern &pattern); //!< sends one %.17g-formatted labelled FULL-WIDTH data row
	double parse_value(const std::string &line); //!< strict full-line numeric parse incl. finiteness (throws on violation, symmetric with the classifier bridge's loudness)
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da, const PDataAccessor)
{
	notify("Criterion_Extern_Value::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	// the class-count requirement guards the DATA protocol (its header carries K, its rows carry
	// labels); in no-data mode the tool never sees either, and the data-free `--features` entry
	// legitimately supplies a classless universe accessor — so the check applies only when DATA flows
	if(!_no_data && da->getNoOfClasses()<2) throw fst_error("Criterion_Extern_Value requires at least 2 classes");
	if(_bridge.get_command().empty()) throw fst_error("Criterion_Extern_Value: no command set (see --extern-cmd)");
	_da=da;
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0;
	_data_stale=true; // the pipeline calls initialize() whenever the effective split changes
	                  // (the filter re-estimation contract) - the tool's DATA must follow
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::send_row(const DIMTYPE label, const PPattern &pattern)
{
	std::FILE *out=_bridge.to_child();
	char buf[64];
	std::snprintf(buf,sizeof(buf),"%ld",(long)label); std::fputs(buf,out);
	for(DIMTYPE f=0;f<_features;f++) // FULL width: the DATA block carries every feature
	{
		std::snprintf(buf,sizeof(buf),"%.17g",(double)pattern[f]); // bit round-trip for doubles
		std::fputc(' ',out);
		std::fputs(buf,out);
	}
	if(std::fputc('\n',out)==EOF)
		_bridge.die_on_write();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::send_data_block()
{
	// header first: count the training patterns (cheap block-size sweep)
	unsigned long long n=0;
	{IDXTYPE s; for(DIMTYPE c=0;c<_classes;c++) {_da->setClass(c); for(const auto &blk : _da->blocks(TRAIN)) {s=blk.size; n+=(unsigned long long)s;}}}
	if(n==0) return false;
	_bridge.send_line("DATA "+std::to_string(n)+" "+std::to_string((long)_features)+" "+std::to_string((long)_classes));

	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	const DIMTYPE stride=_da->getNoOfFeatures();
	for(DIMTYPE c=0;c<_classes;c++)
	{
		_da->setClass(c);
		for(const auto &blk : _da->blocks(TRAIN)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
			send_row(c,&p[(std::size_t)i*stride]);
	}
	_bridge.flush_to_child();
	const std::string resp=_bridge.read_line();
	if(resp!="OK") throw fst_error("Criterion_Extern_Value: '"+_bridge.get_command()+"' rejected DATA: "+resp);
	_data_stale=false;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
double Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::parse_value(const std::string &line)
{
	// strict full-line numeric parse: bare strtod would accept any garbage line (stray
	// child diagnostics, "ERROR ..." messages) as a valid 0.0 value - silently wrong
	// search results; fail loudly instead, like the classifier bridge's prediction parse
	const char *s=line.c_str();
	char *end=0;
	const double v=std::strtod(s,&end);
	if(end==s || *end!='\0')
		{_bridge.shutdown_process(); throw fst_error("Criterion_Extern_Value: '"+_bridge.get_command()+"' returned a non-numeric value line: '"+line+"'");}
	// strtod accepts "nan"/"inf" as numbers, but a non-finite criterion value would poison
	// every comparison SILENTLY (NaN orders false against everything - subsets never win,
	// B&B cut logic misbehaves invisibly); reject it as loudly as garbage text
	if(!std::isfinite(v))
		{_bridge.shutdown_process(); throw fst_error("Criterion_Extern_Value: '"+_bridge.get_command()+"' returned a non-finite value: '"+line+"' (criterion values must be finite; negate error-style measures inside the tool rather than returning infinities)");}
	return v;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Extern_Value<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, const PSubset sub)
{
	notify("Criterion_Extern_Value::evaluate().");
	assert(_da);
	assert(sub);
	assert(_classes==_da->getNoOfClasses());
	assert(_features==_da->getNoOfFeatures());
	assert(sub->get_n_raw()==_da->getNoOfFeatures());

	// buffer the feature subset index
	_subfeatures=0;
	for(DIMTYPE fsel : sub->selected_features()) {assert(_subfeatures<_features); _index[_subfeatures++]=fsel;}
	assert(_subfeatures==sub->get_d_raw());
	if(_subfeatures==0) return false;

	const bool was_running=_bridge.running();
	_bridge.ensure_process();
	if(!was_running) _data_stale=true; // a freshly (re)spawned child holds no DATA

	if(!_no_data && _data_stale && !send_data_block()) return false; // empty train part
	// (in no-data mode the tool never receives DATA - it computes from the indices alone)

	// one VALUE request: d + the 0-based raw feature indices (ascending subset order)
	std::string req="VALUE "+std::to_string((long)_subfeatures);
	for(DIMTYPE f=0;f<_subfeatures;f++) {req+=' '; req+=std::to_string((long)_index[f]);}
	_bridge.send_line(req);
	_bridge.flush_to_child();
	result=(RETURNTYPE)parse_value(_bridge.read_line());
	return true;
}

} // namespace
#endif // FSTCRITERIONEXTERNVALUE_H ///:~
