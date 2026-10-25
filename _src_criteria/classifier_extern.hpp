#ifndef FSTCLASSIFIEREXTERN_H
#define FSTCLASSIFIEREXTERN_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    classifier_extern.hpp
   \brief   Defines Classifier_Extern — bridge delegating train/test to a user-supplied external command (persistent process, line protocol)
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
   \note    SECURITY: this header launches the user-supplied command through the
            system shell. An fst4 config file with an extern-cmd key is CODE —
            never run configs from untrusted sources without reading them first.

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
#include <iostream>
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"
#include "classifier.hpp"
#include "extern_bridge.hpp" // the platform-sensitive process plumbing, shared with Criterion_Extern_Value

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

/*! \brief Bridge classifier delegating training and prediction to a
    user-supplied external command.

    One PERSISTENT child process per classifier instance (spawned lazily on
    first use; criterion clones spawn their own, so threaded evaluation is
    isolation-safe), speaking a line protocol on stdin/stdout:

        -> TRAIN <N> <d> <K>\n  followed by N lines  <label> <v1> ... <vd>
        <- OK\n                 (anything else, e.g. "ERROR ...", fails the evaluation)
        -> TEST <M> <d>\n       followed by M lines  <v1> ... <vd>
        <- M lines, each the predicted class index (0-based integer)

    TRAIN/TEST pairs repeat for every candidate subset; closing stdin (fst4
    exit or clone destruction) tells the child to terminate. Values are
    exchanged as %.17g decimal (bit round-trip for doubles). A reference
    implementation ships in tests/extern_centroid.awk.

    What the external command MUST do:
    - FLUSH ITS STDOUT after each reply (the "OK" line, and the last prediction
      line of a TEST block). Both sides are blocking pipes: a child that leaves
      its reply in a stdio buffer deadlocks against a bridge already waiting to
      read it. Line-buffered output (or an explicit flush) is the usual fix -
      awk's print is line-buffered to a pipe on most implementations, but C and
      Python programs need fflush()/flush=True.
    - Read until stdin reports EOF and then exit. EOF is the only shutdown
      signal sent; no signal, no QUIT command.
    - Write diagnostics to STDERR, never to stdout. Stderr is not read by the
      bridge at all: it is inherited from fst4 and lands wherever fst4's own
      stderr goes. Anything unexpected on stdout desynchronizes the protocol
      and fails the evaluation loudly.
    - Answer every TRAIN with exactly one line and every TEST with exactly M
      lines, in the order the rows were sent. The row order is the canonical
      class-major, block-order enumeration used throughout FST4.

    LOUD CAVEATS (by design, documented rather than hidden):
    - REPRODUCIBILITY GUARANTEES END AT THE BRIDGE. FST4's determinism story
      (a single controlled PRNG, results independent of the thread count)
      covers everything up to the protocol; whether equal TRAIN/TEST input yields equal output is
      entirely the external command's property. A stochastic external trainer
      makes results run- and thread-count-dependent.
    - THE COMMAND IS USER-SUPPLIED CODE EXECUTION through the system shell.
      Config files carrying extern-cmd must be treated like scripts.
    - No timeout machinery: a hung command hangs the search (kill fst4). In
      particular shutdown_process() (and thus the destructor) waits for the
      child to exit; a child that ignores stdin-EOF keeps fst4 waiting at
      teardown until the child is killed externally.
    - SIGPIPE handling: on macOS/BSD the pipe fd is marked F_SETNOSIGPIPE; on
      other POSIX platforms SIGPIPE is ignored PROCESS-WIDE from the first
      spawn (there is no per-fd control) so that a dying child surfaces as an
      FST error instead of silently killing fst4. Windows has no SIGPIPE.
    - classify_scores() is unsupported (the protocol exchanges labels only),
      so ranking metrics (AUC) are unavailable; the confusion-matrix metrics
      (balanced accuracy, F1, class weights) work via the batched
      test_confusion() override. */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Extern : public Clonable_Impl<Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>,Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > {
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent; //!< the immediate abstract base
	typedef Clonable_Impl<Classifier_Extern,parent> parent_impl; //!< implements clone() via the copy-constructor below
	typedef std::shared_ptr<DATAACCESSOR> PDataAccessor;      //!< shared pointer to the data accessor supplying train/test blocks
	typedef std::shared_ptr<SUBSET> PSubset;                  //!< shared pointer to the feature subset being trained/evaluated
	typedef typename DATAACCESSOR::PPattern PPattern;         //!< one data sample as exposed by DATAACCESSOR
	Classifier_Extern() : _bridge("Classifier_Extern"), _classes(0), _features(0), _subfeatures(0) {notify("Classifier_Extern constructor.");}
	virtual ~Classifier_Extern() {notify("Classifier_Extern destructor.");} // the bridge destructor shuts the child down

	void set_command(const std::string &cmd) {_bridge.set_command(cmd);}       //!< sets the external command line (`--extern-cmd`); the bridge asserts it is not already running
	const std::string& get_command() const {return _bridge.get_command();}    //!< returns the configured command line

	//! \brief Records the data shape and allocates the index buffer. Must be called before
	//! train(). Throws fst_error when \p da has fewer than 2 classes, or when no command has
	//! been set yet (set_command()) -- the child is not spawned here, but a bridge without a
	//! command could never spawn one.
	void initialize(const PDataAccessor da);

	//! a single-row TEST round-trip: one header line, one data line, one reply line per call.
	//! Correct but slow -- prefer test()/test_confusion(), which batch the whole test part into
	//! a single round-trip. Returns false (and drops the child, so the next call respawns it)
	//! when the reply is not a valid class index; a non-numeric reply throws instead.
	virtual bool classify(DIMTYPE &cls, const PPattern &pattern) override;
	/*! \brief Streams the designated training part as one TRAIN block: buffers the subset
	    index, counts the training patterns with a cheap block-size sweep, sends the
	    "TRAIN <n> <d> <classes>" header followed by one labeled row per pattern (send_row()),
	    then requires the single-line response "OK" -- anything else throws fst_error (the
	    external tool's own rejection reason is included verbatim). Returns false only when
	    there are zero training patterns; a subset/data shape mismatch is an assertion
	    failure, not a bool return. */
	virtual bool train(const PDataAccessor da, const PSubset sub) override;
	//! accuracy over the designated test part, obtained in ONE batched TEST block. Returns false
	//! either when the test part is empty or when the child's answers were unusable (see
	//! run_test_block()) -- the two cases are not distinguished at this interface.
	virtual bool test(RETURNTYPE &result, const PDataAccessor da) override;
	//! the confusion matrix over the designated test part, from the same single batched TEST
	//! block as test() (overridden so the metric palette does not fall back to the base class's
	//! per-pattern classify() loop, which would cost one round-trip per pattern). Same two
	//! false-return cases as test().
	virtual bool test_confusion(std::vector<RETURNTYPE> &confusion, DIMTYPE &classes, const PDataAccessor da) override;

	virtual std::ostream& print(std::ostream& os) const override {os << "Classifier_Extern(cmd=\"" << _bridge.get_command() << "\")"; return os;}
private:
	friend parent_impl; //!< grants Clonable_Impl access to the copy-constructor
	//! \brief Copy-constructor backing clone(): copies shape/config (class and feature counts,
	//! the narrowing _index, the configured command line) but deliberately gives the clone a
	//! FRESH, not-yet-running Extern_Bridge -- clones never share pipes with their source, each
	//! spawns its own child process lazily on first use (Extern_Bridge::ensure_process()),
	//! which is what makes threaded evaluation isolation-safe.
	Classifier_Extern(const Classifier_Extern& ce) :
		_bridge("Classifier_Extern"), // clones NEVER share pipes — each fresh bridge spawns its own child lazily
		_classes(ce._classes),
		_features(ce._features),
		_subfeatures(ce._subfeatures),
		_index(ce._index)
	{
		_bridge.set_command(ce._bridge.get_command());
		notify("Classifier_Extern copy-constructor.");
	}
protected:
	Extern_Bridge _bridge;         //!< persistent child process + line I/O (per instance; never shared by clones)
	DIMTYPE _classes;              //!< class count, set by initialize()
	DIMTYPE _features;             //!< raw (full, unnarrowed) feature count, set by initialize()
	DIMTYPE _subfeatures;          //!< d = current subset size
	std::vector<DIMTYPE> _index;   //!< subset position -> raw feature index (first d entries valid)

	//! \brief Writes one %.17g-formatted data row over the selected subset to the child's
	//! stdin. When \p with_label is true, \p label_or_none is written first (space-separated
	//! from the feature values); otherwise it is ignored and the row starts directly with
	//! features. Throws (via Extern_Bridge::die_on_write()) if the child has stopped
	//! accepting input.
	void send_row(const DIMTYPE label_or_none, const bool with_label, const PPattern &pattern);
	//! \brief Strict full-line numeric parse of one predicted class index: the ENTIRE line
	//! must be consumed by strtol(), unlike a bare strtol() call which would silently accept
	//! any garbage line (a stray child diagnostic, an "ERROR ..." message) as a valid class-0
	//! prediction. Throws fst_error (and shuts the bridge down) on a non-numeric or partial
	//! line, symmetric with train()'s TRAIN-rejection loudness.
	long parse_prediction(const std::string &line);
	//! \brief Runs one batched TEST round-trip over the designated test part, shared by
	//! test()/test_confusion(): sends the "TEST <m> <d>" header and one unlabeled row per
	//! pattern (send_row()), then collects \p predictions in the canonical class-major,
	//! block-order enumeration. Returns false (without throwing) when there is no test data,
	//! or when any returned prediction is out of [0,_classes) -- in the latter case the
	//! remaining unread response lines are abandoned and the bridge is shut down so the next
	//! call respawns a clean child instead of reading a desynced stream.
	bool run_test_block(std::vector<DIMTYPE> &predictions, const PDataAccessor da);
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Extern::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	if(da->getNoOfClasses()<2) throw fst_error("Classifier_Extern requires at least 2 classes");
	if(_bridge.get_command().empty()) throw fst_error("Classifier_Extern: no command set (see --extern-cmd)");
	_classes=da->getNoOfClasses();
	_features=da->getNoOfFeatures();
	_index.resize(_features);
	_subfeatures=0; // no model yet
}

// (the process spawn/shutdown/line-I/O machinery lives in extern_bridge.hpp, shared with
//  Criterion_Extern_Value; its error messages are prefixed with the component name)
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::send_row(const DIMTYPE label_or_none, const bool with_label, const PPattern &pattern)
{
	std::FILE *out=_bridge.to_child();
	char buf[64];
	if(with_label) {std::snprintf(buf,sizeof(buf),"%ld",(long)label_or_none); std::fputs(buf,out);}
	for(DIMTYPE f=0;f<_subfeatures;f++)
	{
		std::snprintf(buf,sizeof(buf),"%.17g",(double)pattern[_index[f]]); // bit round-trip for doubles
		if(with_label || f>0) std::fputc(' ',out);
		std::fputs(buf,out);
	}
	if(std::fputc('\n',out)==EOF)
		_bridge.die_on_write();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
long Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::parse_prediction(const std::string &line)
{
	// strict full-line numeric parse: a bare strtol would accept any garbage line (a stray child
	// diagnostic, an "ERROR ..." message) as a VALID class 0 prediction and silently produce
	// wrong results; the TRAIN path fails loudly on non-OK, TEST must be equally loud
	const char *s=line.c_str();
	char *end=0;
	const long v=std::strtol(s,&end,10);
	if(end==s || *end!='\0')
		{_bridge.shutdown_process(); throw fst_error("Classifier_Extern: '"+_bridge.get_command()+"' returned a non-numeric prediction line: '"+line+"'");}
	return v;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	notify("Classifier_Extern::train().");
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

	_bridge.ensure_process();

	// header first: count the training patterns (cheap block-size sweep)
	unsigned long long n=0;
	{typename DATAACCESSOR::PPattern p; IDXTYPE s,i; (void)p; (void)i;
	 for(DIMTYPE c=0;c<_classes;c++) {da->setClass(c); for(const auto &blk : da->blocks(TRAIN)) {s=blk.size; n+=(unsigned long long)s;}}}
	if(n==0) return false;
	_bridge.send_line("TRAIN "+std::to_string(n)+" "+std::to_string((long)_subfeatures)+" "+std::to_string((long)_classes));

	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c=0;c<_classes;c++)
	{
		da->setClass(c);
		for(const auto &blk : da->blocks(TRAIN)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
			send_row(c,true,&p[(std::size_t)i*stride]);
	}
	_bridge.flush_to_child();
	const std::string resp=_bridge.read_line();
	if(resp!="OK") throw fst_error("Classifier_Extern: '"+_bridge.get_command()+"' rejected TRAIN: "+resp);
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	assert(_subfeatures>0); // train() must have succeeded
	_bridge.ensure_process();
	_bridge.send_line("TEST 1 "+std::to_string((long)_subfeatures));
	send_row(0,false,pattern);
	_bridge.flush_to_child();
	const long v=parse_prediction(_bridge.read_line());
	if(v<0 || v>=(long)_classes) {_bridge.shutdown_process(); return false;} // desynced/misbehaving child: respawn on next use
	cls=(DIMTYPE)v;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::run_test_block(std::vector<DIMTYPE> &predictions, const PDataAccessor da)
{
	// one batched TEST block over the designated test part; prediction order =
	// the canonical enumeration (class-major, block order) shared by all classifiers
	assert(da);
	assert(_subfeatures>0);
	_bridge.ensure_process();
	unsigned long long m=0;
	{IDXTYPE s; for(DIMTYPE c=0;c<_classes;c++) {da->setClass(c); for(const auto &blk : da->blocks(TEST)) {s=blk.size; m+=(unsigned long long)s;}}}
	if(m==0) return false;
	_bridge.send_line("TEST "+std::to_string(m)+" "+std::to_string((long)_subfeatures));
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	const DIMTYPE stride=da->getNoOfFeatures();
	for(DIMTYPE c=0;c<_classes;c++)
	{
		da->setClass(c);
		for(const auto &blk : da->blocks(TEST)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
			send_row(0,false,&p[(std::size_t)i*stride]);
	}
	_bridge.flush_to_child();
	predictions.clear();
	predictions.reserve((std::size_t)m);
	for(unsigned long long k=0;k<m;k++)
	{
		const long v=parse_prediction(_bridge.read_line());
		if(v<0 || v>=(long)_classes) {
			// the remaining m-k-1 response lines stay unread - a desynced child would make
			// the NEXT round-trip fail with a misleading "rejected TRAIN: <number>" error;
			// drop the child instead, the next evaluation respawns it cleanly
			_bridge.shutdown_process();
			return false;
		}
		predictions.push_back((DIMTYPE)v);
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	notify("Classifier_Extern::test().");
	std::vector<DIMTYPE> pred;
	if(!run_test_block(pred,da)) return false;
	// tally against true labels in the same canonical enumeration order
	unsigned long long count=0, correct=0;
	IDXTYPE s,i;
	for(DIMTYPE c_test=0;c_test<_classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) for(i=0,s=blk.size;i<s;i++)
		{
			if(pred[count]==c_test) correct++;
			count++;
		}
	}
	assert(count==pred.size());
	if(count==0) return false;
	result=(RETURNTYPE)correct/(RETURNTYPE)count;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Extern<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test_confusion(std::vector<RETURNTYPE> &confusion, DIMTYPE &classes, const PDataAccessor da)
{
	// batched override: ONE TEST round-trip instead of the base default's per-pattern classify()
	notify("Classifier_Extern::test_confusion().");
	std::vector<DIMTYPE> pred;
	if(!run_test_block(pred,da)) return false;
	classes=_classes;
	confusion.assign((std::size_t)_classes*(std::size_t)_classes,(RETURNTYPE)0.0);
	unsigned long long count=0;
	IDXTYPE s,i;
	for(DIMTYPE c_test=0;c_test<_classes;c_test++)
	{
		da->setClass(c_test);
		for(const auto &blk : da->blocks(TEST)) for(i=0,s=blk.size;i<s;i++)
		{
			confusion[(std::size_t)c_test*(std::size_t)_classes+(std::size_t)pred[count]]+=1.0;
			count++;
		}
	}
	assert(count==pred.size());
	return count>0;
}

} // namespace
#endif // FSTCLASSIFIEREXTERN_H ///:~
