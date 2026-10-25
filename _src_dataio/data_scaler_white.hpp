#ifndef FSTDATASCALERWHITE_H
#define FSTDATASCALERWHITE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_scaler_white.hpp
   \brief   Implements data whitening (as result, each feature mean value is 0 with normalized stddev)
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
#include <cassert>
#include <sstream>
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

/*! \brief Implements data whitening (as result, each feature mean value is 0 with normalized stddev)

    \note this implementation supports only scaling of one-dimensional data,
       thus is applicable only to scale feature values individually 
       and independently for each other feature

	The transform is (value-mean)/stddev, learned per feature in two passes: the first accumulates
	the mean, the second the squared deviations from it. The variance is divided by the number of
	values (the population convention), not by that number minus one.

	\note Optionally substitutes missing values by the mean of those values that are available
    (separately per feature). Missing values are assumed to be coded by dedicated numerical value
    'missing_val_code'. Because the output is already centred, the mean of the whitened feature is 0
    by construction -- which is exactly the value scale() returns for a missing entry. Missing values
    take part in neither the mean nor the standard deviation.

    \note A feature whose standard deviation comes out as 0 -- a constant feature, or one with no
    available values at all -- scales to 0 everywhere, silently.

    \note better implementation may be needed in case of large sample sizes
       to prevent mean computation bias
*/
template<typename DATATYPE, typename IDXTYPE>
class Data_Scaler_white : public Data_Scaler<DATATYPE> {
public:
	Data_Scaler_white(const int dims=1): learn_loop(0), count1(0), mean(0.0), count2(0), stddev(0.0), missing_values(false), _missing_val_code(0) {if(dims!=1) throw fst_error("Data_Scaler_white() implemented for 1-dimensional data only."); notify("Data_Scaler_white constructor.");} //!< no missing-value substitution; \p dims must be 1
	Data_Scaler_white(const DATATYPE missing_val_code, const int dims): learn_loop(0), count1(0), mean(0.0), count2(0), stddev(0.0), missing_values(true), _missing_val_code(missing_val_code) {if(dims!=1) throw fst_error("Data_Scaler_white() implemented for 1-dimensional data only."); notify("Data_Scaler_white constructor.");} //!< values equal to \p missing_val_code are substituted (scaled value 0) rather than learned/scaled normally; \p dims must be 1
	Data_Scaler_white(const Data_Scaler_white &ds) : learn_loop(ds.learn_loop), count1(ds.count1), mean(ds.mean), count2(ds.count2), stddev(ds.stddev), missing_values(ds.missing_values), _missing_val_code(ds._missing_val_code)  {notify("Data_Scaler_white copy constructor.");} //!< copies configuration and any already-learned statistics
	virtual ~Data_Scaler_white() {notify("Data_Scaler_white destructor.");}
	virtual int learn_loops() const override {return 2;} //!< two passes: the variance of pass 2 needs the mean from pass 1
	virtual bool startFirstLoop() override; //!< resets the mean accumulator and begins pass 1
	virtual bool startNextLoop() override; //!< finalizes the mean (after pass 1) or the standard deviation (after pass 2); returns false once both passes are done
	virtual void learn(const DATATYPE& value) override; //!< accumulates \p value into the mean (pass 1) or its squared deviation from that mean (pass 2), unless it is the missing-value code

	virtual DATATYPE scale(const DATATYPE& value) override; //!< returns (value-mean)/stddev; 0 for a missing value (which is the mean in whitened coordinates) and 0 for a feature with zero standard deviation
	virtual void scale_inplace(DATATYPE& value) override; //!< the in-place form of scale()

	virtual Data_Scaler_white* duplicate() const override {return new Data_Scaler_white(*this);}

	virtual std::ostream& print(std::ostream& os) const override {os << "Data_Scaler_white("; if(missing_values) os << "missing_value=" << _missing_val_code; os << ")"; return os;}
	
protected:
	int learn_loop; //!< which of the two learn() passes is active: 1 = accumulating the mean, 2 = accumulating the variance against that mean
	IDXTYPE count1; //!< number of non-missing values fed to learn() during pass 1 (mean accumulation)
	DATATYPE mean; //!< running sum during pass 1, finalized to the per-feature mean by startNextLoop()
	IDXTYPE count2; //!< number of non-missing values fed to learn() during pass 2 (variance accumulation)
	DATATYPE stddev; //!< running sum of squared deviations during pass 2, finalized to the per-feature standard deviation by startNextLoop()

	// missing value substitution support
	bool missing_values; //!< whether values equal to _missing_val_code are substituted (scaled to 0) rather than learned/scaled normally
	const DATATYPE _missing_val_code; //!< the numerical code that marks a missing value (only meaningful when missing_values is true)
};

template<typename DATATYPE, typename IDXTYPE>
bool Data_Scaler_white<DATATYPE,IDXTYPE>::startFirstLoop() 
{
	learn_loop=1; count1=0; mean=0.0;
	return true;
}
	
template<typename DATATYPE, typename IDXTYPE>
bool Data_Scaler_white<DATATYPE,IDXTYPE>::startNextLoop() 
{
	if(learn_loop==1) {
		 if(count1>0) mean/=count1; else mean=0.0;
		 learn_loop=2; count2=0; stddev=0.0; return true;
	}
	else if(learn_loop==2) {
		 // population variance: the sum of squared deviations is divided by the number of values,
		 // not by that number minus one
		 if(count2>0) stddev=sqrt(stddev/count2); else stddev=0.0; return false;
	}
	else {
		std::ostringstream sos; sos << "Data_Scaler_white::startNextLoop() wrong previous loop. (count1=" << count1 << ", count2=" << count2 << ")";
		throw fst_error(sos.str());
	}
}

template<typename DATATYPE, typename IDXTYPE>
void Data_Scaler_white<DATATYPE,IDXTYPE>::learn(const DATATYPE& value)
{
	if(learn_loop==1) {
		if(!missing_values || value!=_missing_val_code) {
			++count1;
			mean+=value;
		}
	} else if(learn_loop==2) {
		if(!missing_values || value!=_missing_val_code) {
			++count2;
			stddev+=(value-mean)*(value-mean);
		}
	} else throw fst_error("Data_Scaler_white::learn() wrong loop.");
}

template<typename DATATYPE, typename IDXTYPE>
DATATYPE Data_Scaler_white<DATATYPE,IDXTYPE>::scale(const DATATYPE& value) 
{
	assert(learn_loop==2 && count1==count2);
	if(missing_values && value==_missing_val_code) return 0; // substitute the mean, which in whitened coordinates is 0
	if(stddev>0.0) return (value-mean)/stddev; else return 0;
}
	
template<typename DATATYPE, typename IDXTYPE>
void Data_Scaler_white<DATATYPE,IDXTYPE>::scale_inplace(DATATYPE& value) 
{
	assert(learn_loop==2 && count1==count2);
	if(missing_values && value==_missing_val_code) {value=0; return;} // substitute the mean, which in whitened coordinates is 0
	if(stddev>0.0) value=(value-mean)/stddev; else value=0;
}

} // namespace
#endif // FSTDATASCALERWHITE_H ///:~
