#ifndef FSTDENSETRAINTOOLS_H
#define FSTDENSETRAINTOOLS_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    dense_train_tools.hpp
   \brief   Shared dense-training helpers: standardized-design build and Cholesky factor/solve
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

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include "data_accessor.hpp"

namespace FST {

/*! Builds the internally-standardized training design matrix shared by the
    dependency-free dense trainers (logreg, linear SVM, RBF SVM, ridge).

    One pass walks classes 0..classes-1 over the TRAIN blocks of the data
    accessor's CURRENT split - the TEST blocks are never read - pushing the d
    selected raw feature values of each pattern into Z while accumulating the
    standardization moments, then calling after_pattern(pat,c) so the caller can
    append its own per-pattern payload. A second pass finalizes the moments
    (a constant feature gets scale 1 and thus stays inert) and standardizes Z in
    place to zero mean and unit variance.

    \param da data accessor, already positioned on the split to train from
    \param classes number of classes to walk (typically da->getNoOfClasses())
    \param d number of selected features = number of feature values written per pattern
    \param zstride row stride of Z: d when the caller appends nothing, d+1 when it
           appends one value per pattern (e.g. a constant bias coordinate), and
           correspondingly more for a larger payload. after_pattern() MUST push
           exactly zstride-d values into Z, otherwise the rows go out of step.
    \param index subset position -> raw feature index; its first d entries are read
    \param Z output design matrix, cleared first, N rows of zstride values, row-major
    \param mean output per-subset-position standardization shift, d entries
    \param scale output per-subset-position standardization scale, d entries (1 where
           the feature is constant)
    \param after_pattern callback invoked as after_pattern(pat,c) once per pattern,
           right after that pattern's d feature values have been pushed; \c pat points
           at the pattern's raw (unstandardized) feature vector and \c c is its class
    \return the number of training patterns N; 0 means the split has no training data,
            in which case Z, mean and scale are left in an unspecified state. */
template<typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class DATAACCESSOR, class AFTERPATTERN>
std::size_t standardize_design(const std::shared_ptr<DATAACCESSOR> &da, const DIMTYPE classes,
	const std::size_t d, const std::size_t zstride, const std::vector<DIMTYPE> &index,
	std::vector<REALTYPE> &Z, std::vector<REALTYPE> &mean, std::vector<REALTYPE> &scale,
	AFTERPATTERN after_pattern)
{
	const DIMTYPE stride=da->getNoOfFeatures();
	mean.assign(d,0.0); scale.assign(d,0.0);
	Z.clear();
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	std::size_t N=0;
	for(DIMTYPE c=0;c<classes;c++)
	{
		da->setClass(c);
		for(const auto &blk : da->blocks(TRAIN)) for(i=0,p=blk.data,s=blk.size;i<s;i++)
		{
			const auto *pat=&p[(std::size_t)i*stride];
			for(std::size_t f=0;f<d;f++) {
				const REALTYPE v=(REALTYPE)pat[index[f]];
				Z.push_back(v);
				mean[f]+=v;
			}
			after_pattern(pat,c);
			++N;
		}
	}
	if(N==0) return 0;
	for(std::size_t f=0;f<d;f++) mean[f]/=(REALTYPE)N;
	// second pass over the stored raw values: centered sums of squares. The one-pass form
	// E[x^2]-mean^2 cancels catastrophically once |mean| >> sd (features offset by 1e8 changed
	// logreg/linsvm/rbfsvm/ridge results although standardization is translation-invariant);
	// the design matrix is in memory anyway, so the extra pass costs one sweep
	for(std::size_t n=0;n<N;n++)
		for(std::size_t f=0;f<d;f++) {const REALTYPE dev=Z[n*zstride+f]-mean[f]; scale[f]+=dev*dev;}
	for(std::size_t f=0;f<d;f++)
	{
		const REALTYPE var=scale[f]/(REALTYPE)N;
		scale[f]=(var>0.0)?std::sqrt(var):1.0; // constant feature -> scale 1, stays inert
	}
	for(std::size_t n=0;n<N;n++)
		for(std::size_t f=0;f<d;f++) Z[n*zstride+f]=(Z[n*zstride+f]-mean[f])/scale[f];
	return N;
}

/*! In-place lower-Cholesky factorization of the row-major m×m SPD matrix A
    (upper triangle left untouched). Returns false when a pivot is not strictly
    positive finite — callers either fail the training or retry with an
    escalated ridge (in-place factoring destroys A, so retries reassemble). */
template<typename REALTYPE>
bool cholesky_factor(std::vector<REALTYPE> &A, const std::size_t m)
{
	for(std::size_t j=0;j<m;j++)
	{
		REALTYPE diag=A[j*m+j];
		for(std::size_t k=0;k<j;k++) diag-=A[j*m+k]*A[j*m+k];
		if(!(diag>0.0) || !std::isfinite(diag)) return false;
		diag=std::sqrt(diag);
		A[j*m+j]=diag;
		for(std::size_t i=j+1;i<m;i++)
		{
			REALTYPE v=A[i*m+j];
			for(std::size_t k=0;k<j;k++) v-=A[i*m+k]*A[j*m+k];
			A[i*m+j]=v/diag;
		}
	}
	return true;
}

/*! Solves L L^T out = rhs (or = -rhs with negate_rhs) by forward and back
    substitution using the factor left in A by cholesky_factor(). out must be
    pre-sized to m; rhs and out may not alias. */
template<typename REALTYPE>
void cholesky_solve(const std::vector<REALTYPE> &A, const std::size_t m,
	const REALTYPE *rhs, const bool negate_rhs, REALTYPE *out)
{
	for(std::size_t i=0;i<m;i++)
	{
		REALTYPE v=negate_rhs?-rhs[i]:rhs[i];
		for(std::size_t k=0;k<i;k++) v-=A[i*m+k]*out[k];
		out[i]=v/A[i*m+i];
	}
	for(std::size_t ii=m;ii>0;ii--)
	{
		const std::size_t i=ii-1;
		REALTYPE v=out[i];
		for(std::size_t k=i+1;k<m;k++) v-=A[k*m+i]*out[k];
		out[i]=v/A[i*m+i];
	}
}

} // namespace FST
#endif // FSTDENSETRAINTOOLS_H
