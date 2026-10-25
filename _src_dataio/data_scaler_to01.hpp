#ifndef FSTDATASCALERTO01_H
#define FSTDATASCALERTO01_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_scaler_to01.hpp
   \brief   Implements data normalization (of all feature values) to interval [0,1]
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

#include <iostream>
#include <cmath>
#include "error.hpp"
#include "global.hpp"
#include "data_scaler.hpp"

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

/*! \brief Implements data normalization (of all feature values) to interval [0,1]

    \note this implementation supports only scaling of one-dimensional data,
       thus is applicable only to scale feature values individually 
       and independently for each other feature

	The transform is (value-min)/(max-min), with min and max taken per feature over one pass.

    \note optionally substitutes missing values by the mean of those values that are available (separately per feature).
    Missing values are assumed to be coded by dedicated numerical value 'missing_val_code'.
    The substitution happens BEFORE the normalization, so a missing entry ends up at
    (mean-min)/(max-min) -- the mean's position within [0,1] -- not at the raw mean. Missing values
    take part in neither the min/max nor the mean.

    \note Two degenerate cases both yield 0 silently, with no warning: a feature whose values are all
    missing (nothing was learned), and a constant feature, where max==min would divide by zero.
*/
template<typename DATATYPE>
class Data_Scaler_to01 : public Data_Scaler<DATATYPE> {
public:
	Data_Scaler_to01(const int dims=1): first_learn(true), missing_values(false), _missing_val_code(0) {if(dims!=1) throw fst_error("Data_Scaler_to01() implemented for 1-dimensional data only."); notify("Data_Scaler_to01 constructor.");} //!< no missing-value substitution; \p dims must be 1
	Data_Scaler_to01(const DATATYPE missing_val_code, const int dims): first_learn(true), missing_values(true), _missing_val_code(missing_val_code) {if(dims!=1) throw fst_error("Data_Scaler_to01() implemented for 1-dimensional data only."); notify("Data_Scaler_to01 constructor.");} //!< values equal to \p missing_val_code are substituted by the per-feature mean; \p dims must be 1
	Data_Scaler_to01(const Data_Scaler_to01 &ds) : first_learn(ds.first_learn), min(ds.min), max(ds.max), missing_values(ds.missing_values), _missing_val_code(ds._missing_val_code), count(ds.count), sum(ds.sum), avg(ds.avg) {notify("Data_Scaler_to01 copy constructor.");} //!< copies configuration and any already-learned statistics
	virtual ~Data_Scaler_to01() {notify("Data_Scaler_to01 destructor.");}
	virtual int learn_loops() const override {return 1;} //!< one pass suffices: min, max and the mean are all accumulated together
	virtual bool startFirstLoop() override {max=min=0; first_learn=true; sum=0; count=0; return true;} //!< resets the per-feature accumulators and requests the single pass
	virtual bool startNextLoop() override {return false;} //!< one pass is always enough
	virtual void learn(const DATATYPE& value) override; //!< accumulates \p value into the per-feature min, max and mean, unless it is the missing-value code

	virtual DATATYPE scale(const DATATYPE& value) override; //!< returns (value-min)/(max-min); the mean's scaled position for a missing value; 0 if the feature is constant or had no available values
	virtual void scale_inplace(DATATYPE& value) override; //!< the in-place form of scale()

	virtual Data_Scaler_to01* duplicate() const override {return new Data_Scaler_to01(*this);}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Scaler_to01("; if(missing_values) os << "missing_value=" << _missing_val_code; os << ")"; return os;}
	
protected:
	bool first_learn; //!< true until the first learn() call of the current pass has initialized min/max
	DATATYPE min; //!< minimum non-missing value seen for the current feature
	DATATYPE max; //!< maximum non-missing value seen for the current feature

	// missing value substitution support
	bool missing_values; //!< whether values equal to _missing_val_code are substituted rather than learned/scaled normally
	const DATATYPE _missing_val_code; //!< the numerical code that marks a missing value (only meaningful when missing_values is true)
	long count; //!< number of non-missing values fed to learn() for the current feature
	DATATYPE sum; //!< running sum of non-missing values, used to derive avg
	DATATYPE avg; //!< running mean of non-missing values (sum/count); substituted, then scaled, for missing entries
};

template<typename DATATYPE>
void Data_Scaler_to01<DATATYPE>::learn(const DATATYPE& value)
{
	if(!missing_values || value!=_missing_val_code) {
		if(first_learn) {min=max=value; first_learn=false;} else {
			if(value>max) max=value;
			if(value<min) min=value;
		}
		sum+=value; count++;
		avg=sum/(DATATYPE)count;
	}
}

template<typename DATATYPE>
DATATYPE Data_Scaler_to01<DATATYPE>::scale(const DATATYPE& value) 
{
	if(count==0) // all values missing
		return 0;
	else if(missing_values && value==_missing_val_code)
	{
		if(max>min) return (avg-min)/(max-min); else return 0;
	} else {
		if(max>min) return (value-min)/(max-min); else return 0;
	}
}
	
template<typename DATATYPE>
void Data_Scaler_to01<DATATYPE>::scale_inplace(DATATYPE& value) 
{
	if(count==0) // all values missing
		value=0;
	else if(missing_values && value==_missing_val_code)
	{
		if(max>min) value=(avg-min)/(max-min); else value=0;
	} else {
		if(max>min) value=(value-min)/(max-min); else value=0;
	}
}

} // namespace
#endif // FSTDATASCALERTO01_H ///:~
