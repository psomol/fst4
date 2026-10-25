#ifndef FSTINDEXEDVECTOR_H
#define FSTINDEXEDVECTOR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    indexed_vector.hpp
   \brief   Vector representation and operations, allows operation in selected subspace
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
#include <iostream>
#include <limits>
#include <cmath>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "indexed_matrix.hpp"

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

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Matrix;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Vector;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& tdb);

/*! \brief Vector representation and operations, allows operation in selected subspace.
    Also serves as the helper structure for passing matrix-operation intermediate data
    between higher-level matrix objects and functions. */
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Vector {
public:
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to a feature subset (the narrowing selector)
	Indexed_Vector() : _n_max(0), _n(0), _d(0) {notify("Indexed_Vector (empty) constructor");}
	Indexed_Vector(const DIMTYPE n)  : _n_max(0), _n(0), _d(0) {notify("Indexed_Vector constructor"); reset(n);} //!< allocates for \p n features via reset(); starts unnarrowed (get_n()==get_d()==n)
	Indexed_Vector(const Indexed_Vector& iv); //!< copy-constructor (deep-copies the raw buffer and the index map)
	Indexed_Vector& operator=(Indexed_Vector const &iv); //!< deep-copy assignment (reallocates only if \p iv's capacity differs)
	~Indexed_Vector() {notify("Indexed_Vector destructor");}
	void reset(const DIMTYPE n_max); //!< (re)allocates for \p n_max features and resets to full, unnarrowed dimensionality (get_n()==get_d()==n_max, identity index map)
	void redim(const DIMTYPE n=0); //!< keeps the existing allocation (reset() must have run already), resets to an unnarrowed identity index map, and sets get_n()==get_d()==\p n (\p n==0 means the full allocated capacity)

	DATATYPE& operator[](const DIMTYPE i) {assert(i<_n); assert(_data); return _data[i];} //!< raw (unindexed) element access, \p i in [0,get_n())
	DATATYPE& at_raw(const DIMTYPE i) {assert(i<_n); assert(_data); return _data[i];} //!< raw (unindexed) element access, \p i in [0,get_n()) -- same as operator[]
	DATATYPE& at(const DIMTYPE i) {assert(i<_d); assert(_index); return _data[_index[i]];} //!< indexed access through the narrow_to() subset map, \p i in [0,get_d())
	DATATYPE* get_raw() {return _data.get();} //!< raw pointer to the underlying buffer (get_n() valid entries)
	const DATATYPE& operator[](const DIMTYPE i) const {assert(i<_n); assert(_data); return _data[i];} //!< read-only raw (unindexed) element access, \p i in [0,get_n())
	const DATATYPE& at_raw(const DIMTYPE i) const {assert(i<_n); assert(_data); return _data[i];} //!< read-only raw (unindexed) element access -- same as the const operator[]
	const DATATYPE& at(const DIMTYPE i) const {assert(i<_d); assert(_index); return _data[_index[i]];} //!< read-only indexed access through the narrow_to() subset map, \p i in [0,get_d())
	const DATATYPE* get_raw() const {return _data.get();} //!< read-only raw pointer to the underlying buffer (get_n() valid entries)

	//! \brief fill _index[] -> access to data is then mapped to sub-vector; get_d() becomes the subset size.
	//! \pre reset()/redim() has run and get_n() is at least the subset size; \p sub selects at least one feature.
	//! Only at() follows the map - the raw accessors and the *_raw() operations keep addressing all get_n() entries.
	void narrow_to(const PSubset sub);
	void denarrow(); //!< reset _index[] -> access is reset to the full vector (identity map, get_d()==get_n())
	DIMTYPE get_n() const {return _n;} //!< the full ("raw") dimensionality set by reset()/redim()
	DIMTYPE get_d() const {return _d;} //!< the current virtual (narrow_to()'d) dimensionality, get_d()<=get_n()

	// "raw" operations, ignoring index[]
	void set_all_raw_to(const DATATYPE val); //!< sets all get_n() raw entries to \p val (ignores the index map)
	void subtract_raw(const Indexed_Vector& vr); //!< this -= \p vr, elementwise over the raw get_n() entries (requires vr.get_n()==get_n())
	void copy_raw(const Indexed_Vector& vr); //!< this = \p vr, elementwise over the raw get_n() entries (requires vr.get_n()==get_n())

	friend class Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>;
	friend std::ostream& operator<< <>(std::ostream& os, const Indexed_Vector& tdb); //!< grants operator<< access to print this object's contents (see the out-of-class definition below)
	
protected:
	// NOTE: vector size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max
	//       _d     - "virtual subset" dimensionality, _d<=_n
	//       (this logic replicates that in Indexed_Matrix, where it is more important)
	DIMTYPE _n_max; //!< allocated capacity: _data/_index can hold up to _n_max elements
	DIMTYPE _n; //!< full ("raw") dimensionality currently in use, _n<=_n_max; see get_n()
	std::unique_ptr<DATATYPE[]> _data; //!< raw vector data buffer (_n_max entries allocated)
	// sub-vector access tools
	DIMTYPE _d; //!< current virtual (narrow_to()'d) sub-vector dimensionality, _d<=_n; see get_d()
	std::unique_ptr<DIMTYPE[]> _index; //!< maps virtual sub-vector _d dimensions to original _n dimensions (identity after reset()/redim()/denarrow(), set by narrow_to())
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::Indexed_Vector(const Indexed_Vector& iv) : // copy-constructor
	_n_max(iv._n_max),
	_n(iv._n),
	_d(iv._d)
{
	notify("Indexed_Vector copy-constructor.");
	if(_n_max>0)
	{
		_data.reset(new DATATYPE[_n_max]);	memcpy((void *)_data.get(),(void *)(iv._data).get(),sizeof(DATATYPE)*_n_max);
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(iv._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
	}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::operator=(Indexed_Vector const &iv)
{
	notify("Indexed_Vector operator=.");
	if (this != &iv)
	{
		_n=iv._n;
		_d=iv._d;
		if(_n_max!=iv._n_max) 
		{
			_n_max=iv._n_max;
			if(_n_max>0)
			{
				_data.reset(new DATATYPE[_n_max]);	
				_index.reset(new DIMTYPE[_n_max]); 
			} else {
				assert(!iv._data);
				assert(!iv._index);
				_data.reset();
				_index.reset();
			}
		}
		if(_n_max>0)
		{
			memcpy((void *)_data.get(),(void *)(iv._data).get(),sizeof(DATATYPE)*_n_max);
			memcpy((void *)_index.get(),(void *)(iv._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
		}
	}
	return *this;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n_max)
{
	notify("Indexed_Vector reset.");
	assert(n_max>0);
	if(n_max!=_n_max) {
		_n_max=n_max;
		_data.reset(new DATATYPE[_n_max]);
		_index.reset(new DIMTYPE[n_max]); // selected features (subspace) - index into _data
	}
	// unconditionally, like Indexed_Matrix::reset — reset() always restores FULL dimensionality
	// and the identity index map, also when the requested size equals the current allocation
	// (it must not leave a stale redim()ed or narrow_to()'d _n/_d behind)
	_n=_d=n_max;
	for(DIMTYPE i=0;i<_n_max;i++) _index[i]=i;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::redim(const DIMTYPE n)
{
	notify("Indexed_Vector redim.");
	assert(n<=_n_max);
	assert(_data);
	assert(_index);
	if(n==0) _d=_n=_n_max; else _d=_n=n;
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub) // fill _index[] -> access to data is then mapped to sub-matrix
{
	assert(_n>=sub->get_d_raw()); // precondition: enough space to store the subset
	assert(_index);
	_d=0;
	for(DIMTYPE f : sub->selected_features()) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::denarrow() // reset _index[] -> access is reset to the full matrix
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::set_all_raw_to(const DATATYPE val)
{
	assert(_data);
	for(DIMTYPE i=0;i<_n;i++) _data[i]=val;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::subtract_raw(const Indexed_Vector& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n;i++) _data[i]-=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::copy_raw(const Indexed_Vector& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n;i++) _data[i]=mr._data[i];
}

/*! \brief Debug dump: writes tdb's active index map and its narrowed get_d()-dimensional sub-vector to \p os.
    \note The printed data row applies the _index map (tdb._data[tdb._index[j]]), the same convention as
    Indexed_Matrix's operator<< -- what is printed is genuinely the narrow_to()'d sub-vector, so the two
    printed rows are "index -> value" pairs. Debug-only: neither this operator<< nor the matrix one is
    invoked anywhere in the library, they exist for interactive debugging. */
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& tdb)
{
	assert(tdb._n>0);
	assert(tdb._d>0);
	assert(tdb._data);
	assert(tdb._index);
	os << "----------------"<<std::endl;
	os << "Indexed_Vector::" << std::endl;
	os << "_index: "; for(DIMTYPE j=0;j<tdb._d;j++) {os << tdb._index[j] << " ";} os<<std::endl;
	os << "_data (indexed): "; for(DIMTYPE j=0;j<tdb._d;j++) {os << tdb._data[tdb._index[j]] << " ";} os << std::endl;
	os << "----------------"<<std::endl;
	return os;
}


} // namespace
#endif // FSTINDEXEDVECTOR_H ///:~
