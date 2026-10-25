#ifndef FSTINDEXEDMATRIX_H
#define FSTINDEXEDMATRIX_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    indexed_matrix.hpp
   \brief   Matrix representation and operations, allows operation in selected subspace
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
#include <cstddef> // std::size_t (linearized-index arithmetic)
#ifdef FST_USE_EIGEN
#include <Eigen/Dense> // optional Eigen-backed LU/inverse/determinant (CMake option FST4_USE_EIGEN, implies EIGEN_MPL2_ONLY)
#endif
#include "error.hpp"
#include "global.hpp"
#include "indexed_vector.hpp"

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
class Indexed_Vector;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Matrix;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& mr);

//! Matrix representation and operations, allows operation in selected subspace
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Matrix {
public:
	typedef std::shared_ptr<SUBSET> PSubset; //!< shared pointer to the feature subset consumed by narrow_to()
	Indexed_Matrix() : _n_max(0), _n(0), _sign(0), _d(0) {notify("Indexed_Matrix (empty) constructor");}
	Indexed_Matrix(const DIMTYPE n, bool LUstorage=false) : _n_max(0), _n(0), _sign(0), _d(0) {notify("Indexed_Matrix constructor"); reset(n,LUstorage);} //!< allocates for an n x n matrix; \p LUstorage true also allocates the permutation/whereis buffers LUdecompose() needs (i.e. this instance is meant to serve as an LU-decomposition result buffer)
	Indexed_Matrix(const Indexed_Matrix& im); //!< copy-constructor (deep-copies the raw buffer, index map, and any allocated LU/temporary buffers)
	Indexed_Matrix& operator=(Indexed_Matrix const &im); //!< deep-copy assignment (reallocates only where capacity or LU-storage presence differs from \p im's)
	~Indexed_Matrix() {notify("Indexed_Matrix destructor");}
	void reset(const DIMTYPE n_max, bool LUstorage=false); //!< reallocates
	void redim(const DIMTYPE n=0); //!< keeps main allocations, just resets index arrays and adjusts _n; n==0 here equals _n_max

	DATATYPE& at_raw(const DIMTYPE row, const DIMTYPE col) {assert(row<_n); assert(col<_n); assert(_data); return _data[(std::size_t)row*_n+col];} //!< raw (unindexed) element access, \p row and \p col in [0,get_n())
	DATATYPE& at(const DIMTYPE row, const DIMTYPE col) {assert(row<_d); assert(col<_d); assert(_index); assert(_data); return _data[(std::size_t)_index[row]*_n+_index[col]];} //!< indexed access through the narrow_to() subset map -- both \p row and \p col are mapped, \p row/\p col in [0,get_d())
	DATATYPE* get_row_raw(const DIMTYPE row) {assert(row<_n); assert(_data); return &(_data[(std::size_t)row*_n]);} //!< pointer to the start of raw row \p row (unindexed, get_n() entries), \p row in [0,get_n())
	DATATYPE* get_row(const DIMTYPE row) {assert(row<_d); assert(_index); assert(_data); return &(_data[(std::size_t)_index[row]*_n]);} //!< pointer to the start of matrix row _index[\p row] (only the row is subset-mapped -- the returned row still spans all get_n() raw columns, unlike at()), \p row in [0,get_d())

	void narrow_to(const PSubset sub); //!< fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow(); //!< reset _index[] -> access is reset to the full matrix
	DIMTYPE get_n_max() const {return _n_max;} //!< the allocated capacity (max n the matrix can reset()/redim() to without reallocating)
	DIMTYPE get_n() const {return _n;} //!< the full ("raw") dimensionality currently in use, get_n()<=get_n_max()
	DIMTYPE get_d() const {return _d;} //!< the current virtual (narrow_to()'d) dimensionality, get_d()<=get_n()

	// "raw" operations, ignoring index[]
	void set_all_raw_to(const DATATYPE val); //!< sets all get_n()*get_n() raw entries to \p val (ignores the index map)
	void add_raw(const Indexed_Matrix& mr); //!< this += \p mr, elementwise over the raw get_n()*get_n() entries (requires mr.get_n()==get_n())
	void add_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul); //!< this += \p mul * \p mr, elementwise over the raw get_n()*get_n() entries (requires mr.get_n()==get_n())
	void copy_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul); //!< this = \p mul * \p mr, elementwise over the raw get_n()*get_n() entries (an assignment, not an accumulation; requires mr.get_n()==get_n())
	void copy_raw(const Indexed_Matrix& mr); //!< this = \p mr, elementwise over the raw get_n()*get_n() entries (requires mr.get_n()==get_n())

	// "soft" methods - the source (this) is read through the narrow_to() index map,
	// the result objects are written raw (their own _index is ignored). They share
	// one contract:
	//   * no method allocates for the caller -- every result object must be pre-allocated
	//     (Indexed_Matrix results by reset()/redim() to exactly this->get_d(),
	//     Indexed_Vector results to at least this->get_d());
	//   * the LU buffer must have been created with LUstorage==true (the second
	//     constructor argument, or reset()'s) -- the permutation buffers live there;
	//   * LUdecompose() must be called first; solve_equations(), invert(), determinant()
	//     and log_determinant() then only read that buffer. Call them on the SAME object,
	//     with the SAME narrow_to() subset in force, that produced the decomposition:
	//     the LU buffer does not carry the identity of its source matrix, and all four
	//     take their dimension get_d() from *this*, not from the buffer.

	//! Decomposes the currently narrowed get_d() x get_d() sub-matrix into the LU
	//! factorization stored in \p result (L and U packed into one raw matrix, plus the row
	//! permutation and its sign). \p result must be an LUstorage matrix with
	//! get_n()==this->get_d(); it is denarrow()'d and overwritten. Throws fst_error (code
	//! 23) when some row of the sub-matrix is entirely zero to within the smallest
	//! representable magnitude, i.e. the matrix is singular; individual pivots that merely
	//! underflow are clamped instead (see the definition below).
	void LUdecompose(Indexed_Matrix &result);
	//! Solves A*x=b for the matrix A that \p LU_matrix was decomposed from: \p coef_vec is
	//! the right-hand side b (read raw, unchanged), \p result_vec receives the solution x
	//! (raw, pre-allocated to at least get_d() entries). Back-substitution by a pivot below
	//! the smallest representable magnitude yields +/-max() for that component and a notify()
	//! message rather than an exception.
	void solve_equations(Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &result_vec, const Indexed_Matrix &LU_matrix, Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &coef_vec);
	//! Inverts the matrix that \p LU_matrix was decomposed from, into \p result (raw,
	//! pre-allocated with get_n()==this->get_d()). Implemented as get_d() solve_equations()
	//! calls against the unit vectors, so it inherits their underflow behaviour. Uses two
	//! internal work vectors of this object, allocated on first use.
	void invert(Indexed_Matrix &result, const Indexed_Matrix &LU_matrix);
	//! Determinant of the matrix that \p LU_matrix was decomposed from: the product of the
	//! LU diagonal times the permutation sign. The product over- or underflows well before
	//! the matrix itself becomes numerically hopeless -- prefer log_determinant() above
	//! roughly a hundred dimensions.
	DATATYPE determinant(const Indexed_Matrix &LU_matrix);
	//! ln|det| of the matrix that \p LU_matrix was decomposed from, as the sum of the
	//! logarithms of the LU diagonal magnitudes: the overflow-safe companion of
	//! determinant() for high dimensionalities. The sign is discarded (the normal-model
	//! callers work with positive-definite covariance matrices).
	DATATYPE log_determinant(const Indexed_Matrix &LU_matrix);

	friend std::ostream& operator<< <>(std::ostream& os, const Indexed_Matrix& mr); //!< grants operator<< access to print this object's contents (see the out-of-class definition below)
protected:
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max; //!< allocated space can hold _n_max*_n_max matrix
	DIMTYPE _n; //!< full feature set dimensionality
	std::unique_ptr<DATATYPE[]> _data; //!< raw matrix data, _n_max*_n_max entries, row by row
	// if used as LU decomposition target
	std::unique_ptr<DIMTYPE[]> _permut; //!< LU row permutation, physical row -> logical row (the inverse of _tmp_whereis); allocated only for an LUstorage matrix
	int _sign; //!< sign of the LU row permutation (+1/-1; 0 before any decomposition), the factor determinant() needs

	// sub-matrix access tools
	DIMTYPE _d; //!< virtual sub-matrix (as represented by _index) dimensionality
	std::unique_ptr<DIMTYPE[]> _index; //!< maps virtual sub-matrix _d dimensions to original _n dimensions

	// temporary needed inside LUdecompose()
	std::unique_ptr<DATATYPE[]> _tmp_rowscale; //!< reciprocals of the row maxima (implicit row scaling used for pivot choice); allocated on the first LUdecompose() call and kept
	std::unique_ptr<DIMTYPE[]> _tmp_whereis; //!< LU row permutation, logical row -> physical row in _data (the map that replaces physical row swapping); allocated only for an LUstorage matrix
	// temporaries needed inside invert()
	std::unique_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _tmp_eq_coefs; //!< right-hand side handed to solve_equations() by invert() (allocated on first use)
	std::unique_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _tmp_eq_roots; //!< solution vector returned by solve_equations() to invert() (allocated on first use)
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::Indexed_Matrix(const Indexed_Matrix& im) : // copy-constructor
	_n_max(im._n_max),
	_n(im._n),
	_sign(im._sign),
	_d(im._d)
{
	notify("Indexed_Matrix copy-constructor.");
	if(_n_max>0)
	{
		_data.reset(new DATATYPE[(std::size_t)_n_max*_n_max]); memcpy((void *)_data.get(),(void *)(im._data).get(),sizeof(DATATYPE)*_n_max*_n_max);
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(im._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
		if(im._permut)       {_permut.reset(new DIMTYPE[_n_max]); memcpy((void *)_permut.get(),(void *)(im._permut).get(),sizeof(DIMTYPE)*_n_max);}
		if(im._tmp_whereis)  {_tmp_whereis.reset(new DIMTYPE[_n_max]); memcpy((void *)_tmp_whereis.get(),(void *)(im._tmp_whereis).get(),sizeof(DIMTYPE)*_n_max);}
		if(im._tmp_rowscale)	{_tmp_rowscale.reset(new DATATYPE[_n_max]); memcpy((void *)_tmp_rowscale.get(),(void *)(im._tmp_rowscale).get(),sizeof(DATATYPE)*_n_max);}
	}
	if(im._tmp_eq_roots) _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_roots))); // solve_equations result
	if(im._tmp_eq_coefs)	_tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_coefs))); // equations' right hand side coefficients
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::operator=(Indexed_Matrix const &im)
{
	notify("Indexed_Matrix operator=.");
	assert( (im._n_max>0 && im._data && im._index) || (im._n_max==0 && !im._data && !im._index) );
	assert( !im._permut || (im._permut && im._n_max > 0) );
	assert( !im._tmp_whereis || (im._tmp_whereis && im._n_max > 0) );
	assert( !im._tmp_rowscale || (im._tmp_rowscale && im._n_max > 0) );
	if (this != &im)
	{
		_n=im._n;
		_d=im._d;
		_sign=im._sign;

		if(im._n_max>0)
		{
			if(_n_max!=im._n_max)
			{
				_data.reset(new DATATYPE[(std::size_t)im._n_max*im._n_max]);	
				_index.reset(new DIMTYPE[im._n_max]);
			}
			memcpy((void *)_data.get(),(void *)(im._data).get(),sizeof(DATATYPE)*im._n_max*im._n_max);
			memcpy((void *)_index.get(),(void *)(im._index).get(),sizeof(DIMTYPE)*im._n_max); // selected features (subspace) - index into _data

			// allocate also when the capacities match but this side is not yet an LUstorage
			// matrix - a capacity-only test would leave the buffer null and memcpy into it
			if(im._permut)       {if(!_permut || _n_max!=im._n_max) _permut.reset(new DIMTYPE[im._n_max]); memcpy((void *)_permut.get(),(void *)(im._permut).get(),sizeof(DIMTYPE)*im._n_max);} else _permut.reset();
			if(im._tmp_whereis)  {if(!_tmp_whereis || _n_max!=im._n_max) _tmp_whereis.reset(new DIMTYPE[im._n_max]); memcpy((void *)_tmp_whereis.get(),(void *)(im._tmp_whereis).get(),sizeof(DIMTYPE)*im._n_max);} else _tmp_whereis.reset();
			if(im._tmp_rowscale)	{if(!_tmp_rowscale || _n_max!=im._n_max) _tmp_rowscale.reset(new DATATYPE[im._n_max]); memcpy((void *)_tmp_rowscale.get(),(void *)(im._tmp_rowscale).get(),sizeof(DATATYPE)*im._n_max);} else _tmp_rowscale.reset();

			_n_max=im._n_max;
		} else {
			_n_max=0;
			_data.reset();
			_index.reset();
			_permut.reset();
			_tmp_whereis.reset();
			_tmp_rowscale.reset();
		}
		// use assignment operator to accelerate...
		if(im._tmp_eq_roots) _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_roots))); // solve_equations result
			else _tmp_eq_roots.reset();
		if(im._tmp_eq_coefs)	_tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_coefs))); // equations' right hand side coefficients
			else _tmp_eq_coefs.reset();
	}
	return *this;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n_max, bool LUstorage)
{
	notify("Indexed_Matrix reset.");
	assert(n_max>0);
	if(n_max!=_n_max) {
		_n=_d=_n_max=n_max;
		_data.reset(new DATATYPE[(std::size_t)_n_max*_n_max]);
		_index.reset(new DIMTYPE[_n_max]); // selected features (subspace) - index into _data
		if(LUstorage) {
			_permut.reset(new DIMTYPE[_n_max]); 
			_tmp_whereis.reset(new DIMTYPE[_n_max]); 
		} else {
			_permut.reset();
			_tmp_whereis.reset();
		}
		_tmp_rowscale.reset();
		_tmp_eq_coefs.reset();
		_tmp_eq_roots.reset();
	}
	else if(LUstorage && !_permut) { // same capacity, but LU storage newly requested: a
		_permut.reset(new DIMTYPE[_n_max]);      // capacity-only test would leave these null
		_tmp_whereis.reset(new DIMTYPE[_n_max]); // and the init loop below dereference them
	}
	_d=_n=_n_max;
	for(DIMTYPE i=0;i<_n_max;i++) _index[i]=i;
	if(LUstorage) {
		for(DIMTYPE i=0;i<_n_max;i++) _permut[i]=i;
		for(DIMTYPE i=0;i<_n_max;i++) _tmp_whereis[i]=i;
	}
	_sign=0;	
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::redim(const DIMTYPE n)
{
	notify("Indexed_Matrix redim.");
	assert(n<=_n_max);
	assert(_data);
	assert(_index);
	if(n==0) _d=_n=_n_max; else _d=_n=n;
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	if(_permut) for(DIMTYPE i=0;i<_n;i++) _permut[i]=i;
	if(_tmp_whereis) for(DIMTYPE i=0;i<_n;i++) _tmp_whereis[i]=i;
	_sign=0;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub) // fill _index[] -> access to data is then mapped to sub-matrix
{
	assert(_n>=sub->get_d_raw()); // precondition: enough space to store the subset
	assert(_index);
	_d=0;
	for(DIMTYPE f : sub->selected_features()) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::denarrow() // reset _index[] -> access is reset to the full matrix
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::set_all_raw_to(const DATATYPE val)
{
	assert(_data);
	for(std::size_t i=0;i<(std::size_t)_n*_n;i++) _data[i]=val;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::add_raw(const Indexed_Matrix& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(std::size_t i=0;i<(std::size_t)_n*_n;i++) _data[i]+=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::add_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(std::size_t i=0;i<(std::size_t)_n*_n;i++) _data[i]+=mul * mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::copy_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(std::size_t i=0;i<(std::size_t)_n*_n;i++) _data[i]=mul * mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::copy_raw(const Indexed_Matrix& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(std::size_t i=0;i<(std::size_t)_n*_n;i++) _data[i]=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::LUdecompose(Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &result)
{
	// NOTE: 'result' raw data access assumed (result._index ignored)
	//       while this->_index is used when accessing this->_data
	//
	// Crout's method with implicit row scaling and partial pivoting, with one twist:
	// rows are NEVER physically moved inside result._data. Instead two index arrays
	// carried by 'result' record where each row went --
	//   result._tmp_whereis[l] = physical row of _data holding logical row l
	//   result._permut[p]      = logical row stored in physical row p (its inverse)
	// -- so a "row interchange" is a swap of two entries in each array (plus the swap of
	// the two implicit scale factors and a flip of result._sign). Every access to the
	// working matrix therefore goes through result._tmp_whereis[.]*_d rather than the
	// plain row offset, and so must every consumer: solve_equations(), determinant() and
	// log_determinant() all read the diagonal and the triangles through _tmp_whereis.
	// On return the L (unit-diagonal, below) and U (on and above the diagonal) factors of
	// the PERMUTED matrix are packed in result._data in that logical ordering.
	notify("Indexed_Matrix::LUdecompose");
	assert(result._n==_d); // enough pre-allocated space required
	assert(result._data);
	assert(result._permut);
	assert(result._tmp_whereis);
	assert(_index);
	result.denarrow(); // no indexes needed in result, will be written in raw ordering

#ifdef FST_USE_EIGEN
	// Eigen-backed variant: 'result' holds a PLAIN copy of the narrowed source
	// (identity permutation, sign 1); the consumers - solve_equations()/invert()/
	// determinant()/log_determinant() - factor it with Eigen::PartialPivLU on demand.
	// All five methods swap together under FST_USE_EIGEN, so the internal LU
	// representation never mixes with the hand-rolled Crout layout below.
	result._d=_d;
	{
		std::size_t pROW=0;
		for(DIMTYPE i1=0;i1<_d;i1++) {
			for(DIMTYPE i2=0;i2<_d;i2++) {
				assert(pROW+i2<(std::size_t)_d*_d);
				assert((std::size_t)_index[i1]*_n+_index[i2]<(std::size_t)_n*_n);
				result._data[pROW+i2]=_data[(std::size_t)_index[i1]*_n+_index[i2]];
			}
			pROW+=_d;
		}
	}
	for(DIMTYPE j=0;j<_d;j++) {result._permut[j]=j; result._tmp_whereis[j]=j;}
	result._sign=1;
#else
	if(!_tmp_rowscale) _tmp_rowscale.reset(new DATATYPE[_n_max]); // create 'implicit scaling of rows' buffer once for the rest of Indexed_Matrix life

	DATATYPE max;
	DIMTYPE idxmax=0; // the meaningful init is per-column below; DIMTYPE is unsigned, so a -1 "none yet" sentinel is not available here
	DATATYPE pom,sum;
	DIMTYPE j,i1,i2,tmpi;
	std::size_t pROW,pROWK,pCOL;

	result._d=_d;
	pROW=0;
	for(i1=0;i1<_d;i1++) { // copy source sub-matrix to result buffer for further manipulations
		for(i2=0;i2<_d;i2++) {
			assert(pROW+i2<(std::size_t)_d*_d);
			assert((std::size_t)_index[i1]*_n+_index[i2]<(std::size_t)_n*_n);
			result._data[pROW+i2]=_data[(std::size_t)_index[i1]*_n+_index[i2]];
		}
		pROW+=_d;
	}

	result._sign=1;
	/* initialize permutation vector */
	for(j=0;j<_d;j++) {result._permut[j]=j; result._tmp_whereis[j]=j;}

	/* find scaling of rows */
	// implicit row scaling: each row's pivot candidate is later weighted by 1/(its largest
	// magnitude), so pivoting compares rows on equal terms regardless of their units
	pROW=0;
	for(i1=0;i1<_d;i1++)
	{
		max=0.0;
		for(j=0;j<_d;j++) {assert(pROW+j<(std::size_t)_d*_d); if((pom=fabs(result._data[pROW+j]))>max) max=pom;}

		// an all-(near-)zero row makes the matrix singular and the scale factor infinite:
		// the only condition this method reports by throwing (fst_error code 23), as
		// opposed to the pivot clamping further below
		if(max<std::numeric_limits<DATATYPE>::min()) throw fst_error("Indexed_Matrix::LUdecompose() value too small/matrix singular or near-singular",23);

		_tmp_rowscale[i1]=1.0/max;
		pROW+=_d;
	}

	/* Crout's method */
	for(j=0;j<_d;j++)
	{
		/* column j above the diagonal: the U entries */
		for(i1=0;i1<j;i1++)
		{
			pROW=(std::size_t)result._tmp_whereis[i1]*_d;
			assert(pROW+j<(std::size_t)_d*_d);
			sum=result._data[pROW+j];

			for(i2=0;i2<i1;i2++)
			{
				pROWK=(std::size_t)result._tmp_whereis[i2]*_d;
				assert(pROW+i2<(std::size_t)_d*_d);
				assert(pROWK+j<(std::size_t)_d*_d);
				sum-=result._data[pROW+i2]*result._data[pROWK+j];
			}
			result._data[pROW+j]=sum;
		}
		/* search for largest pivot */
		max=0.0;
		idxmax=j; // "no interchange" -- also the outcome for a column that is zero after
		          // scaling, where no candidate ever beats max; leaving idxmax at its value
		          // from the previous column would swap rows on no evidence, and the
		          // too-small-pivot clamp below is what handles the degenerate value
		for(i1=j;i1<_d;i1++)
		{
			pROW=(std::size_t)result._tmp_whereis[i1]*_d;
			assert(pROW+j<(std::size_t)_d*_d);
			sum=result._data[pROW+j];

			for(i2=0;i2<j;i2++)
			{
				pROWK=(std::size_t)result._tmp_whereis[i2]*_d;
				assert(pROW+i2<(std::size_t)_d*_d);
				assert(pROWK+j<(std::size_t)_d*_d);
				sum-=result._data[pROW+i2]*result._data[pROWK+j];
			}
			result._data[pROW+j]=sum;
			pom=_tmp_rowscale[i1]*fabs(sum); // implicit scaling applies to the comparison only
			if(pom>max) {max=pom; idxmax=i1;}
		}

		/* interchange rows ? */
		if(j!=idxmax)
		{
			// NOTE: no data is moved -- the interchange is recorded in _tmp_whereis
			//       and its inverse _permut (see the block comment at the top)

			assert(idxmax<_d);
			assert(j<_d);
			
			pom=_tmp_rowscale[idxmax];
			_tmp_rowscale[idxmax]=_tmp_rowscale[j];
			_tmp_rowscale[j]=pom;

			i1=result._tmp_whereis[j];
			i2=result._tmp_whereis[idxmax];

			assert(i1<_d);
			assert(i2<_d);

			tmpi=result._permut[i1];
			result._permut[i1]=result._permut[i2];
			result._permut[i2]=tmpi;

			tmpi=result._tmp_whereis[result._permut[i1]];
			result._tmp_whereis[result._permut[i1]]=result._tmp_whereis[result._permut[i2]];
			result._tmp_whereis[result._permut[i2]]=tmpi;

			result._sign=-result._sign;
		}

		pCOL=(std::size_t)result._tmp_whereis[j]*_d;

		/* divide the column below the diagonal by the pivot: the L entries */
		if(j<(_d-1))
		{
			assert(pCOL+j<(std::size_t)_d*_d);
			// A pivot at or below the smallest representable magnitude is NOT an error here:
			// its reciprocal is replaced by the largest representable number of the pivot's
			// sign and only a notify() message is issued. The decomposition then completes,
			// but it describes a (numerically) singular matrix -- expect huge or infinite
			// entries downstream, a determinant() that over/underflows, and an invert()
			// result that is meaningless. Callers that cannot tolerate that (the normal-model
			// criteria) treat such a subset as unusable.
			if(fabs(result._data[pCOL+j])<=std::numeric_limits<DATATYPE>::min()) {
				if(result._data[pCOL+j]<0.0) pom=-std::numeric_limits<DATATYPE>::max();
				else pom=std::numeric_limits<DATATYPE>::max();
				notify("NOTE: division by too small a number - result set to maximum representable number.");
			} else
			pom=1.0/result._data[pCOL+j];

			for(i1=j+1;i1<_d;i1++)
			{
				pROW=(std::size_t)result._tmp_whereis[i1]*_d;
				assert(pROW+j<(std::size_t)_d*_d);
				result._data[pROW+j] *= pom;
			}
		}
	}
#endif // FST_USE_EIGEN
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::solve_equations(Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &result_vec, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_matrix, Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &coef_vec)
{
	// NOTE: result_vec, coef_vec, and LU_matrix raw data access assumed (*->_index ignored)
	//       while this->_index is used when accessing this->_data
	notify("Indexed_Matrix::solve_equations");
	assert(result_vec._data);
	assert(result_vec._n>=_d); // enough pre-allocated space required
	assert(coef_vec._data);
	assert(coef_vec._n>=_d); // enough pre-allocated space required
	assert(LU_matrix._n==_d);
	assert(LU_matrix._permut);
	assert(LU_matrix._tmp_whereis);

#ifdef FST_USE_EIGEN
	// Eigen-backed variant: LU_matrix holds the plain narrowed copy (see LUdecompose)
	typedef Eigen::Matrix<DATATYPE,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> FST_EMat;
	typedef Eigen::Matrix<DATATYPE,Eigen::Dynamic,1> FST_EVec;
	Eigen::Map<const FST_EMat> A_e(LU_matrix._data.get(),(Eigen::Index)_d,(Eigen::Index)_d);
	Eigen::Map<const FST_EVec> b_e(coef_vec._data.get(),(Eigen::Index)_d);
	Eigen::Map<FST_EVec> x_e(result_vec._data.get(),(Eigen::Index)_d);
	x_e=Eigen::PartialPivLU<FST_EMat>(A_e).solve(b_e);
#else
	DIMTYPE i,j;
	DATATYPE sum;
	std::size_t pROW;

	/* apply the decomposition's row permutation to the right-hand side b */
	for(i=0;i<_d;i++) {assert(LU_matrix._tmp_whereis[i]<_d); result_vec[i]=coef_vec[LU_matrix._tmp_whereis[i]];}

	/* forward substitution through L (unit diagonal, so result_vec[0] stays as it is) */
	for(i=1;i<_d;i++)
	{
		pROW=(std::size_t)LU_matrix._tmp_whereis[i]*_d;
		sum=0.0;
		for(j=0;j<i;j++) {
			assert(pROW+j<(std::size_t)_d*_d);
			sum+=LU_matrix._data[pROW+j]*result_vec[j];
		}
		result_vec[i]-=sum;
	}

	/* back substitution through U (counting down with a 1-based i, since DIMTYPE is unsigned) */
	for(i=_d;i>0;i--)
	{
		pROW=(std::size_t)LU_matrix._tmp_whereis[i-1]*_d;
		assert(pROW+i-1<(std::size_t)_d*_d);
		sum=0.0;
		for(j=i;j<_d;j++) sum+=LU_matrix._data[pROW+j]*result_vec[j];
		if(fabs(LU_matrix._data[pROW+i-1])<=std::numeric_limits<DATATYPE>::min()) {
			if(LU_matrix._data[pROW+i-1]*(result_vec[i-1]-sum)<0.0) result_vec[i-1]=-std::numeric_limits<DATATYPE>::max();
			else result_vec[i-1]=std::numeric_limits<DATATYPE>::max();
			notify("NOTE: division by too small number - result set to maximum representable number.");
		} else {
			result_vec[i-1]=(result_vec[i-1]-sum)/LU_matrix._data[pROW+i-1];
		}
	}
#endif // FST_USE_EIGEN
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::invert(Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &result, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_Matrix)
{
	// NOTE: 'result' and 'LU_matrix' raw data access assumed (*->_index ignored)
	//       while this->_index is used when accessing this->_data
	notify("Indexed_Matrix::invert");
	assert(result._data);
	assert(result._n==_d); // enough pre-allocated space required
	assert(LU_Matrix._data);
	assert(LU_Matrix._permut);
	assert(LU_Matrix._index);
	assert(LU_Matrix._d==_d);
	
#ifdef FST_USE_EIGEN
	// Eigen-backed variant: LU_Matrix holds the plain narrowed copy (see LUdecompose)
	typedef Eigen::Matrix<DATATYPE,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> FST_EMat;
	Eigen::Map<const FST_EMat> A_e(LU_Matrix._data.get(),(Eigen::Index)_d,(Eigen::Index)_d);
	Eigen::Map<FST_EMat> R_e(result._data.get(),(Eigen::Index)_d,(Eigen::Index)_d);
	R_e=Eigen::PartialPivLU<FST_EMat>(A_e).inverse();
#else
	if(_tmp_eq_roots) assert(_tmp_eq_roots->_n>=_d);
	else _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(_n_max)); // solve_equations result
	if(_tmp_eq_coefs) assert(_tmp_eq_coefs->_n>=_d);
	else _tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(_n_max)); // equations' right hand side coefficients

	std::size_t pROW;
	DIMTYPE i,j;

	// column j of the inverse is the solution of A*x = e_j
	for(j=0;j<_d;j++)
	{
		for(i=0;i<_d;i++) (*_tmp_eq_coefs)[i]=0.0;
		(*_tmp_eq_coefs)[j]=1.0;

		solve_equations(*_tmp_eq_roots,LU_Matrix,*_tmp_eq_coefs);

		pROW=0;
		for(i=0;i<_d;i++)
		{
			result._data[pROW+j]=(*_tmp_eq_roots)[i];
			pROW+=_d;
		}
	}
#endif // FST_USE_EIGEN
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
DATATYPE Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::determinant(const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_matrix)
{
	// NOTE: 'LU_matrix' raw data access assumed (*->_index ignored)
	assert(LU_matrix._data);
	assert(LU_matrix._tmp_whereis);	
	assert(LU_matrix._d==_d);

	notify("Indexed_Matrix::determinant");
#ifdef FST_USE_EIGEN
	// Eigen-backed variant: LU_matrix holds the plain narrowed copy (see LUdecompose)
	typedef Eigen::Matrix<DATATYPE,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> FST_EMat;
	Eigen::Map<const FST_EMat> A_e(LU_matrix._data.get(),(Eigen::Index)_d,(Eigen::Index)_d);
	return (DATATYPE)Eigen::PartialPivLU<FST_EMat>(A_e).determinant();
#else
	DATATYPE det;
	std::size_t pROW;
	DIMTYPE j;

	// permutation sign times the product of the U diagonal, each element reached through
	// the logical-to-physical row map (rows were never physically swapped)
	det=LU_matrix._sign;
	for(j=0;j<_d;j++)
	{
		pROW=(std::size_t)LU_matrix._tmp_whereis[j]*_d+j;
		assert(pROW<(std::size_t)_d*_d);
		det*=LU_matrix._data[pROW];
	}
	return det;
#endif // FST_USE_EIGEN
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
DATATYPE Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::log_determinant(const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_matrix)
{
	// ln|det| as the sum of ln|LU diagonal| - does not over/underflow where the plain
	// diagonal product in determinant() would (high dimension); sign is discarded, which
	// is safe for the covariance-matrix use (positive determinants) it exists for
	// NOTE: 'LU_matrix' raw data access assumed (*->_index ignored)
	assert(LU_matrix._data);
	assert(LU_matrix._tmp_whereis);
	assert(LU_matrix._d==_d);

	notify("Indexed_Matrix::log_determinant");
#ifdef FST_USE_EIGEN
	// Eigen-backed variant: factor the plain narrowed copy, sum ln|U diagonal|
	typedef Eigen::Matrix<DATATYPE,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> FST_EMat;
	Eigen::Map<const FST_EMat> A_e(LU_matrix._data.get(),(Eigen::Index)_d,(Eigen::Index)_d);
	Eigen::PartialPivLU<FST_EMat> lu_e(A_e);
	DATATYPE logdet=0.0;
	for(DIMTYPE j=0;j<_d;j++) logdet+=std::log(std::fabs(lu_e.matrixLU()((Eigen::Index)j,(Eigen::Index)j)));
	return logdet;
#else
	DATATYPE logdet=0.0;
	std::size_t pROW;
	for(DIMTYPE j=0;j<_d;j++)
	{
		pROW=(std::size_t)LU_matrix._tmp_whereis[j]*_d+j;
		assert(pROW<(std::size_t)_d*_d);
		logdet+=std::log(std::fabs(LU_matrix._data[pROW]));
	}
	return logdet;
#endif // FST_USE_EIGEN
}

//! \brief Debug dump: writes mr's active index map and its narrowed get_d() x get_d() sub-matrix to \p os.
//! The printed data rows apply the _index map (mr._index[i]/mr._index[j]), so what is printed is genuinely
//! the narrow_to()'d sub-matrix, not a raw prefix -- the same convention as Indexed_Vector's operator<<.
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& mr)
{
	assert(mr._n>0);
	assert(mr._d>0);
	assert(mr._data);
	assert(mr._index);
	os << "----------------"<<std::endl;
	os << "Indexed_Matrix::";
	os << "_index: "; for(DIMTYPE j=0;j<mr._d;j++) {os << mr._index[j] << " ";} os<<std::endl;
	os << "_data (indexed):"<<std::endl;
	for(DIMTYPE i=0;i<mr._d;i++) {for(DIMTYPE j=0;j<mr._d;j++) {os << mr._data[(std::size_t)mr._index[i]*mr._n+mr._index[j]] << " ";} os << std::endl;}
	os << "----------------"<<std::endl;
	return os;
}

} // namespace
#endif // FSTINDEXEDMATRIX_H ///:~
