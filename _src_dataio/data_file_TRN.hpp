#ifndef FSTDATAFILETRN_H
#define FSTDATAFILETRN_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_file_TRN.hpp
   \brief   Enables data accessor objects to read TRN type of data files
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

   \warning Please note that this code plays only supportive role
            and does not represent coding standards followed in core
            FST4 codes.
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
#include <iostream> 
#include <vector>
#include <cstdio>
#include <cstring>
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

/*! \brief TRN data format filter

    Reads FST's own plain-text data format. A TRN file is a header followed by a data section.

    Header. One keyword per line, the keyword being the line's first word:
    \li \c \#datafile -- file-type marker, takes no argument. REQUIRED, and must precede every
        other keyword.
    \li \c \#title &lt;text&gt; -- optional; the rest of the line is the data set's title.
        Longer than 499 characters it is truncated. Omitted, the title reads "no title".
    \li \c \#features &lt;N&gt; -- REQUIRED; N is the data dimensionality, at least 1.
    \li \c \#classes &lt;K&gt; &lt;n1&gt;,&lt;n2&gt;,... -- REQUIRED; K is the number of classes,
        followed by exactly K per-class sample counts. Commas and whitespace separate the counts
        interchangeably. Every count must be at least 1.
    \li \c \#data -- optional; marks the end of the header. If it is absent, the header ends at the
        first line that is neither blank, a comment, nor a keyword, and that line is read as the
        first row of data.

    A line whose first non-blank character is a semicolon is a comment, and blank lines are ignored;
    both may appear anywhere in the header. No keyword may appear twice. Keyword order is otherwise
    free, except that \c \#datafile comes first.

    Data section. Whitespace-separated decimal numbers, read strictly in order: all samples of class
    1, then all samples of class 2, and so on; within a class, one sample after another, each sample
    being \c \#features consecutive values. Line breaks carry no meaning -- only the sequence of
    numbers matters, although one sample per line is the usual layout. There is NO class-label
    column: class membership is positional and comes entirely from the counts on the \c \#classes
    line, so the file must hold at least features * (n1+n2+...+nK) values.

    A minimal example (2 features, 2 classes of 3 and 2 samples):
    \verbatim
    #datafile
    #title A tiny example
    ; comments start with a semicolon
    #features 2
    #classes  2  3,2
    #data
    0.1 0.4
    0.2 0.5
    0.3 0.6
    1.1 1.4
    1.2 1.5
    \endverbatim

    \note The reader is written around C stdio and fixed-size line buffers: a header line longer
    than 999 characters is truncated, and the per-class counts are parsed through \c int, so a class
    larger than INT_MAX is not representable. Neither limit is reachable with realistic data.
    Malformed headers throw fst_error; see GetFileParams() for what each error code means. */
template<typename DATATYPE, typename IDXTYPE>
class Data_File_TRN {
public:
	Data_File_TRN() {_filename=""; _title=""; _no_of_features=0; _no_of_classes=0; _valid=false; _file=nullptr;}
	~Data_File_TRN() {close();}
	void open(const std::string& filename); //!< opens TRN file and reads header
	void close() {if(_file!=nullptr) {fclose(_file); _file=nullptr;}} //!< closes TRN file if it is open
	const unsigned int getNoOfFeatures() const {return _no_of_features;} //!< data dimensionality, as read from the header's \#features line
	const unsigned int getNoOfClasses() const {return _no_of_classes;} //!< number of classes, as read from the header's \#classes line
	const IDXTYPE getClassSize(const unsigned int cidx) const {if(cidx<_no_of_classes) return _class_sizes[cidx]; else return 0;} //!< size of class \p cidx, as read from the header's \#classes line; 0 if \p cidx is out of range
	//! Reads the next value from the data section into \p target; returns false (without throwing) on EOF/parse failure, or if the header was never successfully read (\c open() failed or was not called).
	bool read_val(DATATYPE& target); // does not throw
	//! Reads and returns the next value from the data section; throws fst_error on EOF/parse failure, or if the header was never successfully read.
	DATATYPE read_val();// throws
private:
	Data_File_TRN(const Data_File_TRN&); //!< declared, not defined -> prevents pass-by-value/copy (predates the C++11 \c =delete idiom used elsewhere in FST4)
protected:
	/*! \brief parses the TRN header (opens #_filename, reads \#datafile/\#title/\#features/\#classes keywords, leaves the file positioned at the start of the data section in #_datastart); throws fst_error on any malformed header

	    Every failure throws fst_error with the same message text and a distinguishing code:
	    1 the file cannot be opened; 5 a keyword occurs twice; 6 a keyword or a data row appears
	    before \c \#datafile and \c \#features have both been seen; 7 the \c \#features argument is not
	    a number; 8 the \c \#classes argument is not a number, or the \c \#classes line is missing
	    altogether; 9 one of the per-class counts is not a number; 10 an unrecognized line appears
	    before the header is complete; 11 the file holds nothing but blank and comment lines;
	    12 the file ends inside the header (neither a data row nor a \c \#data line follows it);
	    88 the number of per-class counts differs from the declared class count; 100 fewer than one
	    class; 101 fewer than one feature; 102 a class with fewer than one sample. */
	void GetFileParams();
	std::string _filename;	//!< input data file name, as passed to open()
	std::string _title;	//!< optional data title (part of the data file); defaults to "no title" if the header omits \#title
	unsigned int _no_of_features;	//!< number of features (read from the header)
	unsigned int _no_of_classes;	//!< number of classes (read from the header)
	std::vector<IDXTYPE> _class_sizes;	//!< per-class sample counts, indexed by class index
	IDXTYPE _datastart;	//!< file offset of the start of data past the header
	int _valid;	//!< indicates the state, when header is successfully read and file open and ready to be read for actual data.
	FILE *_file;   //!< the open TRN file handle, or nullptr if none is open
};

template<typename DATATYPE, typename IDXTYPE>
void Data_File_TRN<DATATYPE,IDXTYPE>::open(const std::string& filename)
{
	_filename=filename;
	_class_sizes.clear();
	_no_of_features=0; _no_of_classes=0;
	GetFileParams();
}

template<typename DATATYPE, typename IDXTYPE>
void Data_File_TRN<DATATYPE,IDXTYPE>::GetFileParams()
{
  const unsigned int ONUMCOM=5;        // max no. of header keywords
  const unsigned int OHEADSTRLEN=1000; // max header line size
  const unsigned int OFILENAMELENGTH=500;
  const std::string ONOTITLE="no title";
  const char *command[ONUMCOM]={"#datafile",
			"#title",
			"#features",
			"#classes",
			"#data"};
  char s[OHEADSTRLEN];   /* each header line is read into this */
  char word[OHEADSTRLEN];/* the first word of that line is copied here */
  char *s2;             /* points at the start of the header keyword, if present */
  char *s3;             /* points at the data following the keyword */
  int empty=0;          /* empty file? (convention inherited from Optima) - FALSE */
  int header_done=0;          /* whole-header-read indicator */
  unsigned int i;
  int scanned;
  /* Keyword bookkeeping. Each keyword of command[] owns one bit, in table order:
       1 = #datafile, 2 = #title, 4 = #features, 8 = #classes, 16 = #data.
     "seen" is the running OR of the keywords seen so far, so (seen&1) reads as "#datafile has
     been seen", (seen&4) as "#features has been seen", and so on; a keyword whose bit is already
     set in seen is a duplicate. "keyword" holds the bit of the keyword found on the current line,
     or 0 when the line is not a keyword at all -- which, once #datafile and #features are in,
     means the header is over and the line is the first row of data. "mask" is the walking mask
     that assigns those bit values while the command[] table is scanned. */
  int seen=0;		/* bit record of which header keywords occurred so far */
  int keyword=0;         /* bit value of the recognized keyword */
  int mask;		/* bit mask for reading/writing bits of seen */

  _valid = false;
  // error check
  if((_file=fopen(_filename.c_str(),"r"))==nullptr) {throw fst_error("Data_File_TRN file read problem.",1);}
  empty=1;  /* TRUE - empty file */
  _title=ONOTITLE;

  while((!feof(_file))&&(!header_done))
  {
    _datastart=ftell(_file);
    if(fgets(s,OHEADSTRLEN,_file)==nullptr) break; /* end of file (or a read error) inside the header */

    scanned=sscanf(s,"%s",word); 			/* did it read anything at all? */
    if((scanned)&&(scanned!=EOF)&&(word[0]!=';')&&(word[0]!=0))
    {
      empty=0; /* FALSE */
      mask=1;
      keyword=0;
      s3=nullptr;
      for(i=0;(i<ONUMCOM)&&(!keyword);i++)         /* is it one of *command[]? */
      {
        s2=strstr(word,command[i]);
        if(s2!=nullptr)
        {
          keyword=mask;
          s3=strstr(s,word)+strlen(command[i]); /* point s3 into s past the keyword */
        }
        mask<<=1;
      }
      if(s3==nullptr)
      {
        // error check
        // an unrecognized line is acceptable only once #datafile and #features are in, in which
        // case it is the first data row and case 0 below ends the header
        if(!((seen&1)&&(seen&4)))
           {throw fst_error("Data_File_TRN file read problem.",10);}
      }
      else for(i=0;s3[i]!=0;i++)if(s3[i]==',')s3[i]=' ';

      // error check: keyword seen twice
      if((keyword)&&(seen&keyword)) {throw fst_error("Data_File_TRN file read problem.",5);}
      else
      {
        switch(keyword)
        {              // error check
          case 0:  if((seen&1)&&(seen&4))	/* data row (without the #data keyword) */
               {
                 header_done=1;
                 // the #data keyword is optional: this first data row has already been
                 // consumed by the header scan, so seek back to its start (_datastart
                 // is recorded before each header line is read)
                 fseek(_file,(long)_datastart,SEEK_SET);
               }
               else {throw fst_error("Data_File_TRN file read problem.",6);}
               break;
                       // error check
          case 2:  if(!(seen&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                       /* title */
                 s2=strstr(s3,"\n"); if(s2!=nullptr) s2[0]=0; // erase the newline
                 while((s3[0]!=0)&&((s3[0]==' ')||(s3[0]=='\t'))) s3++;
                 // error check
                 if(strlen(s3)>=OFILENAMELENGTH) // a longer title is silently truncated
                     s3[OFILENAMELENGTH-1]=0;
                 _title=s3;
               }
               break;
                       // error check
          case 4:  if(!(seen&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                       /* features */
                 int tmp_scan=0; // sscanf %d needs an int-sized target
                 scanned=sscanf(s3,"%d",&tmp_scan); _no_of_features=tmp_scan;
                 // error check
                 if((!scanned)||(scanned==EOF))
                 {
                   _no_of_features=-1;
                   throw fst_error("Data_File_TRN file read problem.",7);
                 }
               }
               break;
                       // error check
          case 8:  if(!(seen&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                        /*classes */
                 int tmp_scan=0; // sscanf %d needs an int-sized target
                 scanned=sscanf(s3,"%d",&tmp_scan); _no_of_classes=tmp_scan;
                 // error check
                 if((!scanned)||(scanned==EOF))
                 {
                   _no_of_classes=-1;
                   throw fst_error("Data_File_TRN file read problem.",8);
                 }
                 else
                 {
                   if(_no_of_classes>0)
                   {
                   	_class_sizes.clear();
                   }
                   sscanf(s3,"%s",word);
                   s3=strstr(s3,word)+strlen(word);
                   for(i=0;i<_no_of_classes;i++)
                   {
                   	IDXTYPE temp;
                     int tmp_scan=0; // scan into an int and convert: %d into a template-typed IDXTYPE would be wrong for any IDXTYPE that is not int-sized
                     scanned=sscanf(s3,"%d",&tmp_scan); temp=(IDXTYPE)tmp_scan;
                     // error check
                     if((!scanned)||(scanned==EOF))
                     {
                       throw fst_error("Data_File_TRN file read problem.",9);
                     }
                     else
                     {
                       _class_sizes.push_back(temp);
                       sscanf(s3,"%s",word);
                       s3=strstr(s3,word)+strlen(word);
                     }
                   }
                 }
               }
               break;
                       // error check    
          case 16: if((seen&1)&&(seen&4))	/*data*/
               {
                 header_done=1;
                 _datastart=ftell(_file);
               }
               else  {throw fst_error("Data_File_TRN file read problem.",6);} 
               break;
          default: ;
        }
        seen|=keyword;
      }
    }
  }
  // error check: empty file
  if(empty) {throw fst_error("Data_File_TRN file read problem.",11);}
  // error check: the file ended inside the header (no data row and no #data line)
  if(!header_done) {throw fst_error("Data_File_TRN file read problem.",12);}

  if(!(seen&8)) // class count undefined
  {
      _no_of_classes = 0; // undefined
      throw fst_error("Data_File_TRN file read problem.",8); // report via the error message that this is not a usual data file
  }
  if(_no_of_classes!=_class_sizes.size()) {throw fst_error("Data_File_TRN file read problem.",88);}
  if(_no_of_classes<1) {throw fst_error("Data_File_TRN file read problem.",100);}
  if(_no_of_features<1) {throw fst_error("Data_File_TRN file read problem.",101);}
  for(i=0;i<_no_of_classes;i++) if(_class_sizes[i]<1) {throw fst_error("Data_File_TRN file read problem.",102);}
 
  _valid = true;
}

template<typename DATATYPE, typename IDXTYPE>
bool Data_File_TRN<DATATYPE,IDXTYPE>::read_val(DATATYPE& target)
{
	if(_valid) {
		double dbltmp; int err;
		err=fscanf(_file,"%lf",&dbltmp); 
		if(err>0) {
			target=dbltmp;
			return true;
		}
	}
	return false;
}

template<typename DATATYPE, typename IDXTYPE>
DATATYPE Data_File_TRN<DATATYPE,IDXTYPE>::read_val()
{
	if(_valid) {
		double dbltmp; int err;
		err=fscanf(_file,"%lf",&dbltmp); 
		if(err>0) return DATATYPE(dbltmp); else throw fst_error("Data_File_TRN::read_val() problem.",200);
	}
	throw fst_error("Data_File_TRN::read_val() problem.",201);
}

} // namespace
#endif // FSTDATAFILETRN_H ///:~
