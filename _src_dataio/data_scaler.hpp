#ifndef FSTDATASCALER_H
#define FSTDATASCALER_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_scaler.hpp
   \brief   Defines interface for data scaling implementations (for use in data accessors)
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

#include <iostream>
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

// NOTE: this implementation supports only scaling of one-dimensional data,
//       thus is applicable only to scale feature values individually 
//       and independently for each feature

template <typename DATATYPE>
class Data_Scaler;

template <typename DATATYPE>
std::ostream& operator<<(std::ostream& os, const Data_Scaler<DATATYPE>& ds);

//! abstract class, defines interface for data scaling implementations (for use in data accessors)
template<typename DATATYPE>
class Data_Scaler {
	//! Streams a scaler's print() to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Data_Scaler<DATATYPE>& ds);
public:
	Data_Scaler(const int dims=1) {notify("Data_Scaler constructor.");} //!< \p dims is accepted for concrete-scaler constructor symmetry; the base class does not use it
	//! \brief Copy constructor. The base class carries no state of its own; the concrete scalers'
	//! copy constructors copy both the configuration and whatever statistics have been learned so far,
	//! which is what makes a copy usable for scaling straight away. duplicate() is the virtual entry
	//! point that per-split scaling uses to obtain such a copy through a base-class pointer.
	Data_Scaler(const Data_Scaler &ds) {notify("Data_Scaler copy constructor.");}
	virtual ~Data_Scaler() {notify("Data_Scaler destructor.");}
	//! \brief Number of full data passes learn() needs before scale() is valid.
	//! 1 for scalers whose statistic is a single-pass reduction (to01's min/max); 2 for
	//! scalers whose second-order statistic depends on the first (white's stddev needs
	//! the mean from pass 1 first); 0 for a scaler that has nothing to learn.
	//! \note This is a declaration of the scaler's need, for a caller that wants to know it in
	//! advance. It is NOT what drives the passes: the caller runs
	//! `for(bool more=startFirstLoop(); more; more=startNextLoop())`, so the pass count is whatever
	//! that pair returns. A scaler reporting 0 loops may still ask for one pass and ignore its values.
	virtual int learn_loops() const =0;
	//! Begins the first learning pass: resets accumulators. Returns false to request no learning
	//! pass at all (callers then go straight to scale() with the scaler's default state).
	virtual bool startFirstLoop() =0;
	//! Finalizes the just-completed pass (e.g. dividing an accumulated sum into a mean) and
	//! begins the next one. Returns false once no further pass is needed, ending the caller's loop.
	virtual bool startNextLoop() =0;
	//! Feeds one data value into the accumulator for the current pass (see startFirstLoop()/startNextLoop()).
	//! All values of ONE feature are fed, then the pass sequence restarts for the next feature.
	virtual void learn(const DATATYPE& value) =0;

	virtual DATATYPE scale(const DATATYPE& value) =0; //!< return the scaled value
	virtual void scale_inplace(DATATYPE& value) =0; //!< scale the value in place

	/*! returns a new independent copy of this scaler (configuration and learned state), owned by
	    the caller; needed by per-split scaling, where each sharing_clone of a data accessor
	    re-learns the transform concurrently and thus needs a private scaler instance.
	    The base implementation throws — a user-derived scaler must override it to be usable for
	    per-split scaling in threaded searches (the built-in scalers all do). */
	virtual Data_Scaler* duplicate() const {throw fst_error("Data_Scaler::duplicate() not implemented by this scaler; required for per-split scaling with clones.");}

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the scaler and its configuration; base default writes nothing
};

//! Streams a scaler's print() to \p os; the operator concrete scalers are actually printed through.
template<typename DATATYPE>
std::ostream& operator<<(std::ostream& os, const Data_Scaler<DATATYPE>& ds) {
	return ds.print(os);
}

} // namespace
#endif // FSTDATASCALER_H ///:~
