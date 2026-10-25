#ifndef FSTDATAINTERVALLER_H
#define FSTDATAINTERVALLER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_intervaller.hpp
   \brief   Container to hold a list of Data_Interval; implements nested interval reduction
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    October 2010 (last updated July 2026)
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
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector> // for blockiter - array of iterators indexed by loopdepth
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

//! Support structure to hold data interval (referring into data held by data accessor)
template<typename IDXTYPE>
struct Data_Interval {
	IDXTYPE startidx; //!< index of the interval's first sample
	IDXTYPE count;    //!< number of samples in the interval
};

template <class CONTAINER, typename IDXTYPE>
class Data_Intervaller;

template <class CONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia);

//! Container to hold list of Data_Interval; implements nested interval reduction
template<class CONTAINER, typename IDXTYPE>
//! \invariant CONTAINER must be a container of Data_Interval supporting iterators, push_back(), clear()
class Data_Intervaller : public CONTAINER {
public:
	Data_Intervaller() {notify("Data_Intervaller constructor.");};
	//! Deep-copies \p tin's interval list; blockiter (the active get*Block() loop cursors) is
	//! deliberately NOT copied -- it holds iterators into the source container, so the copy
	//! starts with no active block loop (getFirstBlock() (re)initializes it per loopdepth).
	Data_Intervaller(const Data_Intervaller& tin);
	virtual ~Data_Intervaller() {notify("Data_Intervaller destructor.");};
	Data_Interval<IDXTYPE>* getFirstBlock(const unsigned int loopdepth=0); //!< positions loop \p loopdepth at the first interval and returns it, or nullptr if there are none
	Data_Interval<IDXTYPE>* getNextBlock(const unsigned int loopdepth=0); //!< advances loop \p loopdepth (previously started via getFirstBlock() with the same loopdepth) and returns the next interval, or nullptr once exhausted
	//! \brief checks whether all intervals are non-overlapping, ordered ascending, and all fit to <0,maxidx> (maxidx==0 disables maxidx test)
	//! \note On failure it also writes a short diagnostic to std::cerr (through syncout::print()) saying which check failed.
	bool check_validity(const IDXTYPE maxidx);
	IDXTYPE sum();	//!< sums the sizes of all contained intervals, i.e. the total number of samples they cover
	IDXTYPE max_idx();	//!< returns the largest sample index any interval covers, i.e. the maximum of startidx+count-1 over non-empty intervals (0 if there are none)
	/*! \brief Translates \p reducer's RELATIVE index intervals into absolute ones and stores them in \p target.

	    The intervals held here select a subset of the data by absolute sample index. \p reducer selects
	    a further subset of THAT selection, addressing it as if the intervals here were concatenated into
	    one contiguous run: its index 0 is this list's first sample, its index 1 the second, and so on
	    across interval boundaries. reduce() maps every such relative interval back to the absolute
	    indexes it actually denotes, splitting it wherever it straddles a boundary of this list, and
	    appends the results to \p target.

	    This is what makes nested splitting work: a splitter at a deeper level enumerates positions
	    within the enclosing level's training part without knowing where in the data that part lies.

	    \param reducer the relative-index selection to translate; its indexes must all be smaller than sum()
	    \param target receives the absolute-index result; CLEARED first, so any previous content is lost */
	void reduce(const std::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > reducer, std::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > target);
	//! Streams every contained interval as "si:<startidx>, c:<count>" lines to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia);
private:
	typedef typename Data_Intervaller<CONTAINER,IDXTYPE>::iterator TCI; //!< this container's iterator type, used to type blockiter and the sum()/max_idx()/reduce() loop variables
	std::vector<TCI> blockiter; //!< to enable multiple independent get*Block() loops
};

template<class CONTAINER, typename IDXTYPE>
Data_Intervaller<CONTAINER,IDXTYPE>::Data_Intervaller(const Data_Intervaller& tin)
{
	notify("Data_Intervaller copy constructor.");
	this->clear();
	for(typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator iter=tin.begin();iter!=tin.end();iter++) this->push_back(*iter);
	// blockiter deliberately NOT copied: it holds iterators into the SOURCE container, which
	// would be invalid here — the copy starts with no active block loops; getFirstBlock()
	// (re)initializes the cursor of whichever loopdepth is used next
}

//! Streams every contained interval as "si:<startidx>, c:<count>" lines to \p os.
template<class CONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia) {
	for(typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator iter=ia.begin();iter!=ia.end();iter++) os << "si:" << iter->startidx << ", c:" << iter->count << std::endl;
	return os;
}

template<class CONTAINER, typename IDXTYPE>
Data_Interval<IDXTYPE>* Data_Intervaller<CONTAINER,IDXTYPE>::getFirstBlock(const unsigned int loopdepth)
{
	if(blockiter.size()<=loopdepth) blockiter.resize(loopdepth+1);
	blockiter[loopdepth]=this->begin();
	if(blockiter[loopdepth]!=this->end()) return &(*blockiter[loopdepth]);
	else return nullptr;
}

template<class CONTAINER, typename IDXTYPE>
Data_Interval<IDXTYPE>* Data_Intervaller<CONTAINER,IDXTYPE>::getNextBlock(const unsigned int loopdepth)
{
	// NOTE: getNextBlock() must be preceded by getFirstBlock() with the same loopdepth value, otherwise the behaviour is undefined
	assert(loopdepth<blockiter.size());
	++blockiter[loopdepth];
	if(blockiter[loopdepth]!=this->end()) return &(*blockiter[loopdepth]);
	else return nullptr;
}

template<class CONTAINER, typename IDXTYPE>
bool Data_Intervaller<CONTAINER,IDXTYPE>::check_validity(const IDXTYPE maxidx)
{
	if(this->empty()) {{std::ostringstream sos; sos << "Data_Intervaller is empty" << std::endl; syncout::print(std::cerr,sos);} return false;}
	TCI iter=this->begin();
	IDXTYPE pivot=iter->startidx+iter->count;
	if(iter->startidx<0 || (maxidx>0 && (pivot>maxidx)) ) {{std::ostringstream sos; sos <<"first is wrong"<<std::endl; syncout::print(std::cerr,sos);} return false;} // the first interval reaches outside <0,maxidx>
	iter++;
	while(iter!=this->end())
	{
		if(iter->startidx<pivot) {{std::ostringstream sos; sos <<"iter->startidx="<<iter->startidx<<", pivot="<<pivot<<std::endl; syncout::print(std::cerr,sos);} return false;}
		pivot=iter->startidx+iter->count;
		if(maxidx>0 && (pivot>maxidx)) {{std::ostringstream sos; sos <<"startidx+count>maxidx"<<std::endl; syncout::print(std::cerr,sos);} return false;}
		++iter;
	}
	return true;
}

template<class CONTAINER, typename IDXTYPE>
IDXTYPE Data_Intervaller<CONTAINER,IDXTYPE>::sum()
{
	IDXTYPE cnt=0;
	for(TCI iter=this->begin();iter!=this->end();iter++) cnt+=iter->count;
	return cnt;
}

template<class CONTAINER, typename IDXTYPE>
IDXTYPE Data_Intervaller<CONTAINER,IDXTYPE>::max_idx()
{
	IDXTYPE mx=0;
	for(TCI iter=this->begin();iter!=this->end();iter++) if(iter->count>0 && iter->startidx+iter->count-1>mx) mx=iter->startidx+iter->count-1; // count==0: startidx+count-1 underflows unsigned IDXTYPE
	return mx;
}

template<class CONTAINER, typename IDXTYPE>
void Data_Intervaller<CONTAINER,IDXTYPE>::reduce(const std::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > reducer, std::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > target)
{
	try{
		assert(sum()>reducer->max_idx()); // the reducer indexes INTO this list: its largest index must be below this list's sample count
		target->clear();
		typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator rediter=reducer->begin();
		IDXTYPE Mbottom=0;
		for(TCI iter=this->begin();iter!=this->end();iter++) {
			if((*iter).count==0) continue; // empty interval: Mtop below would underflow unsigned IDXTYPE
			bool stop=false;
			while((!stop) && (rediter!=reducer->end())) {
				if((*rediter).count==0) {rediter++; continue;} // empty interval: Rtop below would underflow unsigned IDXTYPE
				IDXTYPE Rbottom=(*rediter).startidx;
				IDXTYPE Rtop=Rbottom+(*rediter).count-1;
				IDXTYPE MTRUEbottom=(*iter).startidx;
				IDXTYPE Mtop=Mbottom+(*iter).count-1;
				if(!(Rtop<Mbottom || Rbottom>Mtop)) { //write intersection to target
					IDXTYPE ISCbottom=Rbottom; if(Rbottom<Mbottom) ISCbottom=Mbottom;
					IDXTYPE ISCtop=Rtop; if(Rtop>Mtop) ISCtop=Mtop;
					Data_Interval<IDXTYPE> I={MTRUEbottom+(ISCbottom-Mbottom), ISCtop-ISCbottom+1};
					target->push_back(I);
				}
				if(Rtop>=Mtop) stop=true;
				if(Rtop<=Mtop) rediter++;
			}
			Mbottom+=(*iter).count;
		}
		
	} catch(...) {
		throw fst_error("Data_Intervaller::reduce() error.");
	}
}


} // namespace
#endif // FSTDATAINTERVALLER_H ///:~
