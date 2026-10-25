#ifndef FSTSEARCHBRANCHANDBOUNDPREDICTOR_H
#define FSTSEARCHBRANCHANDBOUNDPREDICTOR_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    branch_and_bound_predictor.hpp
   \brief   Interface for Implementations of Prediction Mechanism in Fast Branch & Bound and Branch & Bound with Partial Prediction
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

template<class RETURNTYPE, typename DIMTYPE>
class Branch_And_Bound_Predictor;

template<class RETURNTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Branch_And_Bound_Predictor<RETURNTYPE,DIMTYPE>& babp);

/*! \brief Defines Interface for Implementations of Prediction Mechanism in Fast Branch and Bound & Branch and Bound with Partial Prediction

	\warning CONTRACT relied on by the threaded FBB/BBPP bodies: predictor state
	must be isolated per feature - predict(f) may read only state learned for
	feature f, and learn(f) may modify only feature f's state. FBB/BBPP decide
	predict-vs-compute for a whole tree level upfront on the master thread and
	replay the learns in candidate order after the batched evaluations; that
	upfront pass gives the same decisions as deciding one candidate at a time
	exactly because no feature's decision can observe another feature's same-level
	learn. A predictor that let features share state would make the two passes
	differ, and threaded runs would stop reproducing sequential ones - do not
	implement one against this interface.
*/
template<class RETURNTYPE, typename DIMTYPE>
class Branch_And_Bound_Predictor {
	//! Streams a predictor's print() (implementation-defined diagnostic state) to \p os.
	friend std::ostream& operator<< <>(std::ostream& os, const Branch_And_Bound_Predictor<RETURNTYPE,DIMTYPE>& babp);
public:
	Branch_And_Bound_Predictor() {notify("Branch_And_Bound_Predictor constructor.");}
	virtual ~Branch_And_Bound_Predictor() {notify("Branch_And_Bound_Predictor destructor.");}

	//! \brief (Re)initializes the predictor's per-feature state for \p features features; must be called before learn()/predict(), and again for every new search.
	//! \param features number of features the predictor will track (features 0..features-1); must be at least 1
	//! \pre features>=1 - a predictor tracking nothing has no use, and implementations assert against it
	virtual void initialize(const DIMTYPE features)=0;
	//! \brief Records an observed criterion \p value for \p feature, updating that feature's state only (see the class's per-feature isolation contract above).
	//! \param feature the feature this observation belongs to
	//! \param value the observed criterion value
	//! \return false if the observation could not be recorded; implementations that never reject an observation return true unconditionally
	virtual bool learn(const DIMTYPE feature, const RETURNTYPE value)=0;
	//! \brief Predicts \p feature's criterion value from its previously learn()ed observations, without evaluating the criterion.
	//! \param feature the feature to predict a value for
	//! \param value set to the predicted value on success; left unspecified on failure
	//! \return false if not enough observations are available yet for \p feature -- predict-vs-compute callers (FBB/BBPP) then fall back to actually evaluating the criterion for it
	virtual bool predict(const DIMTYPE feature, RETURNTYPE &value) const =0;

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the predictor's learned state; base default writes nothing
};

//! Streams a predictor's print() (implementation-defined diagnostic state) to \p os.
template<class RETURNTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Branch_And_Bound_Predictor<RETURNTYPE,DIMTYPE>& babp) {
	return babp.print(os);
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDPREDICTOR_H ///:~
