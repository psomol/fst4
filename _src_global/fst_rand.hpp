#ifndef FSTFSTRAND_H
#define FSTFSTRAND_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    fst_rand.hpp
   \brief   FST4-owned pseudo-random number generator (UCRT-compatible LCG),
            replacing platform-specific std::rand()/std::srand()
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

namespace FST {

/*! \section fstrand FST4-owned PRNG

    Every random draw the FST4 library makes goes through fst_rand()/fst_srand() rather than
    the C library rand()/srand(). The reason is reproducibility: the libc generator is a
    different algorithm on every platform (Windows UCRT: 32-bit LCG with 15-bit output;
    macOS libc: minstd Lehmer; glibc: additive feedback generator), so the same seed would
    produce different data splits and different search trajectories on each of them.

    fst_rand() is a fixed 32-bit linear congruential generator with 15-bit output, identical
    in every build. The guarantee it provides: a given seed yields the same sequence of draws
    -- and therefore the same randomized splits, the same Monte Carlo samples and the same
    search result -- on every platform and with every supported compiler.

    \warning This generator exists for reproducibility, not for statistical quality. A
    15-bit-output LCG has short period, strong serial correlation in the low bits and
    correlated lattice structure in higher dimensions. Do not reuse it as a general-purpose
    random source for statistical simulation, and never for anything security-related.

    Contract:
    - state is thread_local; the default state 1 makes an unseeded run equivalent to seeding
      with 1, which is what the fst4 command-line tool documents as its `--seed` default
    - output range is 0..FST_RAND_MAX == 32767 (15 bits); note that FST::FST_RAND_MAX is a
      namespace-scoped constant, deliberately distinct from the libc macro RAND_MAX, which is
      2^31-1 on macOS and Linux -- never scale an fst_rand() draw by RAND_MAX
    - a uniform index below some n comes from fst_rand_below(n), never from a bare fst_rand()%n:
      up to n==32768 it is exactly that (one draw), above it composes two draws into a 30-bit
      value so that every index remains reachable -- a bare modulus never exceeds 32767, so a
      feature index or sample index above that would never be drawn (feature universes beyond
      32768 features, classes of more than 32768 samples)
    - after fst_srand(1) the first five draws are 41, 18467, 6334, 26500, 19169
    - the optional vendored LibSVM code keeps its own libc rand() and is deliberately left
      untouched, so SVM parameter-search results are not reproducible across platforms
*/

//! LCG state, one instance per thread; each thread starts from the default state 1
inline unsigned int& _fst_rand_state() { static thread_local unsigned int s = 1; return s; }

//! Seeds the calling thread's FST4 PRNG; use instead of std::srand()
inline void fst_srand(unsigned int seed) { _fst_rand_state() = seed; }

//! Returns the next pseudo-random draw in 0..FST_RAND_MAX; use instead of std::rand()
inline int fst_rand()
{
	unsigned int& s = _fst_rand_state();
	s = s * 214013u + 2531011u;
	return (int)((s >> 16) & 0x7fff);
}

//! Maximum value fst_rand() can return; use instead of the libc macro RAND_MAX
const int FST_RAND_MAX = 32767;

/*! \brief Returns a pseudo-random index in 0..n-1 (n >= 1); use instead of fst_rand()%n.

    For n <= 32768 this is exactly fst_rand()%n -- one draw, so the sequences of every seeded
    run over ranges that size stay what they always were. For wider ranges two draws are
    composed into a 30-bit value (the first draw supplies the high 15 bits), so every index
    below n is reachable; a bare fst_rand()%n never exceeds 32767 and would silently confine
    random feature subsets, random sample draws and warm-up probes to the first 32768 indexes.
    The modulo is slightly biased in both regimes, as it always was; n beyond 2^30 is not
    supported (FST4 dimensionalities and class sizes are far smaller). */
inline unsigned long fst_rand_below(const unsigned long n)
{
	if(n<=32768UL) return (unsigned long)fst_rand()%n;
	const unsigned long r=((unsigned long)fst_rand()<<15)|(unsigned long)fst_rand();
	return r%n;
}

} // namespace FST

#endif // FSTFSTRAND_H
