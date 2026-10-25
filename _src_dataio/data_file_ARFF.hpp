#ifndef FSTDATAFILEARFF_H
#define FSTDATAFILEARFF_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    data_file_ARFF.hpp
   \brief   Enables data accessor objects to read ARFF (Weka) type of data files
   \author  Jan Hora (hora@utia.cas.cz) with collaborators, see Contacts at https://fst.utia.cz
   \author  Version 4 was developed with Anthropic Claude models as coding assistants:
            Claude Fable 5 for the 2026 development (with minor contributions by
            Claude Opus 5), Claude Fable 5.1 for the pre-publication reviews of the
            full codebase, and Claude Opus 4.5 for the December 2025 audit of the
            FST3 codebase; design, decisions, review and acceptance are human work
   \date    September 2012 (last updated July 2026)
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

#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <vector>
#include <iostream>
#include "error.hpp"
#include "global.hpp"

namespace FST {

	//////////////////////////////////////////////   Local utilities


	//! \brief ARFF data format filter - representation of an attribute (one \@ATTRIBUTE column)
	class Data_File_ARFF_Feature {
	public:
		static const int TYPE_NUMERIC = 1;      //!< loadable: a single-precision float slot
		static const int TYPE_NOMINAL = 2;      //!< loadable: a `{v1,v2,...}` enumeration, stored as a short index into #values
		static const int TYPE_STRING = 3;       //!< recognized but unloadable (ARFF STRING); the column is skipped, see Data_File_ARFF::skippedColumns
		static const int TYPE_UNSUPPORTED = 4;	//!< recognized but unloadable (DATE, RELATIONAL, typos...); the column is skipped, see Data_File_ARFF::skippedColumns

		std::string name;                  //!< attribute name, as written after \@ATTRIBUTE
		std::string typeName;		//!< the type text as written in the file (skip reporting)
		int type;                          //!< one of TYPE_NUMERIC/TYPE_NOMINAL/TYPE_STRING/TYPE_UNSUPPORTED
		int offset;                        //!< byte offset of this feature's value within a Data_File_ARFF_Record's raw data buffer
		int size;                          //!< size in bytes of this feature's value slot (sizeof(float) or sizeof(short))
		bool isClass;                      //!< true if this is the designated CLASS attribute (excluded from Data_File_ARFF::features, tracked separately)
		std::vector<std::string> values;			//!< the enumeration's value names, for TYPE_NOMINAL (empty otherwise)
	};

	//! \brief ARFF data format filter - representation of a record (one \@DATA row)
	class Data_File_ARFF_Record {
		void *data;    //!< raw feature-value buffer, recordSize bytes, laid out per Data_File_ARFF_Feature::offset/size (owned; freed in the destructor)
		int classId;   //!< index into the class attribute's values, or -1 if not yet set

	public:
		Data_File_ARFF_Record(int recordSize) {this->data = malloc(recordSize); classId = -1;} //!< allocates (uninitialized) an \p recordSize-byte data buffer
		~Data_File_ARFF_Record();

		void *getData() { return this->data;} //!< the raw feature-value buffer (see #data)
		int getClassId() { return this->classId;} //!< this record's class index, or -1 if unset
		void setClassId(int pId) {this->classId = pId;} //!< sets this record's class index
	};

	inline Data_File_ARFF_Record::~Data_File_ARFF_Record() {
		if (this->data != nullptr) free(this->data);
		this->data = nullptr;
	}



	/////////////////////////////////////////////////    Main data_file_arff class
	/*! \brief ARFF data format filter

	    Reads the subset of Weka's ARFF format that FST4 can use.

	    Supported. \@RELATION, \@ATTRIBUTE and \@DATA sections in the usual order; lines starting with
	    '%' are comments and blank lines are ignored; keywords are matched case-insensitively; data
	    rows may be dense (comma-separated, values optionally single- or double-quoted) or sparse
	    (`{index value, index value, ...}`, where an omitted attribute means 0 and an omitted class
	    means class 0). Lines may be of any length.

	    Attribute types. NUMERIC, REAL and INTEGER load as floating-point features. A nominal type
	    (`{v1,v2,...}`) loads as its 0-based index into the declared value list; a row's value must
	    match one of the declared names exactly (after trimming), or the row fails.

	    Attribute types that are skipped. STRING, DATE, RELATIONAL and any unrecognized type cannot
	    be represented as a feature. Such a column is reported on standard output and then skipped:
	    its tokens are parsed and discarded, and the file continues loading. \warning A skipped column
	    does NOT keep an index. Feature indices are assigned to the REMAINING columns in file order,
	    so every column after a skipped one shifts down by one, and the mapping between FST4 feature
	    numbers and columns in the file is no longer the identity. The load summary states how many
	    columns were skipped. If every column is skipped, loading fails.

	    The class attribute. The class is the nominal attribute named CLASS (case-insensitive, with
	    or without surrounding single quotes). It is excluded from the features, and its value list
	    defines the classes, in declaration order. A file without such an attribute loads, but reports
	    zero classes and cannot be split by class -- so the FST4 accessors require one.

	    Errors. A malformed \@ATTRIBUTE definition, an unparseable data value, and a sparse row with an
	    out-of-range column index all abort the load; the reader reports the reason on standard output
	    and prepareFile() returns false. An unopenable file throws fst_error. */
	class Data_File_ARFF {
	private:
		std::string _fileName;  //!< input .arff file name, as passed to the constructor
		std::vector<Data_File_ARFF_Record*> data;  //!< loaded records, owned (deleted in the destructor); reordered in place by sortRecordsByClass()
		std::vector<Data_File_ARFF_Feature*> attributes;	//!< LOADED attribute columns only (unsupported types are skipped), INCLUDING the CLASS attribute if present; owned (deleted in the destructor)
		std::vector<Data_File_ARFF_Feature*> features;  //!< the subset of #attributes that are actual features (i.e. excluding the CLASS attribute); populated by prepareAttributes()
		//  graceful degradation over unsupported column types: columnMap has one entry per
		//  @ATTRIBUTE line in file order — the index into `attributes` for loaded columns,
		//  -1 for skipped ones. Row parsers walk file columns through this map, so a
		//  STRING/DATE/unknown-typed column does not abort the load; each skip is reported
		//  loudly and the load summary states that feature indices follow the REMAINING
		//  columns in file order.
		//! \brief One \@ATTRIBUTE column that was recognized but could not be loaded (STRING/DATE/unknown type), for the load-summary report.
		struct SkippedColumn {
			std::string name;      //!< the attribute's name
			std::string typeName;  //!< the type text as written in the file
			int fileColumn;        //!< 0-based column index among all \@ATTRIBUTE lines (not just loaded ones)
		};
		std::vector<SkippedColumn> skippedColumns;  //!< every column skipped due to an unsupported type, in file order; see #columnMap
		std::vector<int> columnMap;  //!< one entry per \@ATTRIBUTE line in file order: the index into #attributes for a loaded column, or -1 for a skipped one (see #skippedColumns)
		int *classSize;  //!< per-class record counts, indexed by class id (size #classCount); owned, allocated in prepareAttributes(), freed in the destructor

		std::string relationName;  //!< the \@RELATION name
		int attributesCount;  //!< number of loaded (non-skipped) attribute columns, i.e. attributes.size()
		int classAttribute;  //!< index into #attributes of the nominal attribute named CLASS/'CLASS', or -1 if none was found
		int classCount;  //!< number of classes (the CLASS attribute's value count), or -1 if #classAttribute is -1

		int recordSize;  //!< size in bytes of one record's raw data buffer (sum of all loaded attributes' Data_File_ARFF_Feature::size); set by prepareAttributes()
		bool loaded;  //!< whether prepareFile() completed without error; getters that need loaded data throw fst_error while false
	public:
		Data_File_ARFF(std::string fileName);  //!< constructs an (unloaded) reader for \p fileName; call prepareFile() to actually read it
		~Data_File_ARFF();

		//! \brief Reads and parses the whole .arff file: the header (\@RELATION/\@ATTRIBUTE lines) then every \@DATA row.
		//! Columns of an unsupported type (STRING, DATE, or any other non-numeric/non-nominal type) are skipped
		//! gracefully with a loud per-column message; a malformed \@ATTRIBUTE definition is still a hard failure.
		//! Loading only hard-fails on unsupported columns if NONE of them turn out loadable (every column skipped).
		//! \return whether loading succeeded (also stored in #loaded); throws fst_error if the file cannot be opened.
		bool prepareFile();				// prepares the file contents

		//! \brief Reorders #data in place, grouping records by class id (bucket sort over #classCount buckets, then concatenated in class order).
		//! \throws fst_error if the file was not loaded or #classCount<1 (no class attribute was found).
		void sortRecordsByClass();

		//! \brief Number of classes, i.e. the CLASS attribute's value count.
		//! \return 0 if no CLASS attribute was found (logged, not an error); the CLASS attribute's Data_File_ARFF_Feature::values size otherwise.
		//! \throws fst_error if the file was not loaded, or (defensively -- prepareAttributes() makes this unreachable) if the CLASS attribute is not nominal.
		unsigned int getNoOfClasses();			// resturn number of classes
		unsigned int getNoOfFeatures() {return features.size();} //!< number of non-class loadable features (#features, i.e. #attributes minus the CLASS attribute)
		unsigned int getNoOfAttributes() {return attributes.size();} //!< number of loaded attribute columns (#attributes), INCLUDING the CLASS attribute if present
		//! \brief Number of loaded records (#data's size).
		//! \throws fst_error if the file was not loaded.
		unsigned int getTotalSize();				// returns the total number of records
		//! \brief Number of records belonging to class \p clsId.
		//! \return the count from #classSize.
		//! \throws fst_error if the file was not loaded, if #classSize is null (no class attribute), or if \p clsId is out of [0,#classCount).
		unsigned int getClassSize(int clsId);	// returns number of records for given class
		//! \brief Name (nominal value text) of class \p clsId.
		//! \return "-n/a-" if #classSize is null or \p clsId is out of [0,#classCount); the CLASS attribute's value name otherwise.
		//! \throws fst_error if the file was not loaded.
		std::string getClassName(int clsId); // returns the nam of given class

		Data_File_ARFF_Record* getRecordAt(long i) {if (i<0 || i>= (long)data.size()) return nullptr; else return data[i];} //!< the loaded record at index \p i, or nullptr if \p i is out of range
		//  returns the value of given feature of given record
		//! \brief The value of #features[feature] in record \p record, read out of that record's raw data buffer.
		//! \note A nominal feature's value is returned as its (small, exactly float-representable) Data_File_ARFF_Feature::values index, cast from the short it is stored as.
		//! \warning Unchecked: \p record must be in [0,getTotalSize()) and \p feature in [0,getNoOfFeatures()),
		//! and the file must be loaded. Out-of-range arguments read out of bounds rather than failing.
		float getFeatureValue(long record, int feature) {
			char *v = ((char *)(data[record]->getData()))+features[feature]->offset;
			if (features[feature]->type == Data_File_ARFF_Feature::TYPE_NOMINAL) return (float)(*(short *)v); // nominal values are stored as short (see readAttrValue)
			return *(float *)v;
		};

		void printInfo(); //!< prints a full debug dump (every record's feature values, the attribute list with nominal value tables, and per-class sizes) to stdout; just a "File not loaded" message if #loaded is false
	private:
		Data_File_ARFF_Feature *createAttribute(char *info);  //!< parses one \@ATTRIBUTE definition's `name type` text (\p info) into a new Data_File_ARFF_Feature (TYPE_STRING/TYPE_UNSUPPORTED for unloadable types); returns nullptr on a malformed definition
		//! \brief Reads and stores one data row (dense or sparse, dispatched on a leading '{'); increments the row's class's #classSize entry if the row has a class id.
		//! \return false if the row is empty or fails to parse (no record is stored).
		bool readNextRow(char *row);
		//! \brief Post-\@DATA setup: assigns each loaded attribute's Data_File_ARFF_Feature::offset within a record buffer, splits off the CLASS attribute into #classAttribute/#classCount (allocating #classSize), and sets #recordSize.
		void prepareAttributes();
		//! \brief Parses one dense (comma-separated) data row into a new record, walking file columns through #columnMap (skipped columns are consumed and discarded).
		//! \return the parsed record, or nullptr (freeing the partial record) if a token fails to parse.
		Data_File_ARFF_Record *readNormalRow(char *row);
		//! \brief Parses one sparse (`{idx value, idx value, ...}`) data row into a new record (unset attributes default to 0; an omitted class attribute defaults to class 0), remapping each file column index through #columnMap.
		//! \return the parsed record, or nullptr (freeing the partial record) if a token fails to parse.
		Data_File_ARFF_Record *readSparseRow(char *row);
		//! \brief Parses \p value into \p feature's slot of \p rec's data buffer (float for TYPE_NUMERIC, a Data_File_ARFF_Feature::values index for TYPE_NOMINAL); also sets \p rec's class id if \p feature isClass.
		//! \return false (logging a message) if \p value doesn't parse as the feature's type, or isn't among a nominal feature's Data_File_ARFF_Feature::values.
		bool readAttrValue(Data_File_ARFF_Feature *feature, Data_File_ARFF_Record *rec, char *value);

		///  reads one full line of arbitrary length, growing the buffer as needed;
		///  returns false at EOF with nothing read. There is no line-length cap: a wide
		///  dense row (thousands of attributes easily exceed any fixed buffer) is read whole
		///  rather than being split into fragments that would parse as garbage rows.
		bool readLine( FILE *f, std::vector<char> &buff ) {
			size_t len = 0; buff[0] = 0;
			while (fgets(buff.data()+len, (int)(buff.size()-len), f) != nullptr) {
				len += strlen(buff.data()+len);
				if ((len > 0 && buff[len-1] == '\n') || feof(f)) return true;
				if (len >= buff.size()-2) buff.resize(2*buff.size());
			}
			return len > 0;
		};

		///  trims a string (removes spaces, tabs, and cr / lf
		char *trim( char *szSource ) {
			char *pszEOS = szSource + strlen( szSource ) - 1;
			while( pszEOS >= szSource && (*pszEOS == ' ' || *pszEOS == 9 || *pszEOS == 13 || *pszEOS == 10)) *pszEOS-- = '\0';
			pszEOS = szSource;
			while( *pszEOS == ' ' || *pszEOS == 9  || *pszEOS == 13 || *pszEOS == 10) pszEOS++;
			return pszEOS;
		};

		///  replaces nonstandard strupr
		char *mystrupr( char *szSource ) {
			char *pszEOS = szSource;
			while( *pszEOS != 0 ) {*pszEOS=toupper(*pszEOS); pszEOS++;}
			return szSource;
		};
	};


	inline Data_File_ARFF::Data_File_ARFF(std::string fileName) {
		this->_fileName = fileName;
		this->classSize = nullptr;
		attributesCount = 0;
		classAttribute = -1;
		classCount = -1;
		loaded = false;
	};

	inline Data_File_ARFF::~Data_File_ARFF() {
		for (unsigned i=0;i<attributes.size(); i++) delete attributes[i];
		this->attributes.clear();
		this->features.clear();
		for (unsigned i=0;i<data.size(); i++) delete data[i];
		this->data.clear();
		if (classSize != nullptr) delete[] classSize;
		classSize = nullptr;
	}

	inline bool Data_File_ARFF::prepareFile() {
		FILE *f =fopen(_fileName.c_str(),"rt");
		if(f==nullptr) {throw FST::fst_error("Data_File_ARFF file read problem.",1);}
		std::vector<char> buff(8192);	//  grown by readLine() as needed — no line-length cap

		bool readingData = false;
		bool isError = false;
		int rowsReaded = 0;

		while (!isError && readLine(f, buff)) {
			// NOTE: trim() right-trims in place but LEFT-trims only through its return value,
			// so the result must be used — reading buff.data() instead would leave an indented
			// @DATA/@ATTRIBUTE line unrecognized
			char *line = trim(buff.data());

			if (!line[0]) continue;		//  empty row
			if (line[0]=='%') continue;			// comment
			if (readingData) {
				if (readNextRow(line)) rowsReaded++;
				else isError = true;
			} else if (line[0]=='@') {
				//  get the second part of string
				char *val = line+2; while(*val!=' ' && *val!=0 && *val!=9) val++; val[0] = 0; val++;
				trim(mystrupr(line));

				if (strcmp(line, "@RELATION")==0) this->relationName = val;
				else if (strcmp(line, "@DATA")==0) {
					readingData = true;
					prepareAttributes();
				} else if (strcmp(line, "@ATTRIBUTE")==0) {
					Data_File_ARFF_Feature *feature = this->createAttribute(val);
					if (feature == nullptr) {	//  malformed definition — a damaged header stays fatal
						std::cout << "Attribute description not understood : " << line << " " << val << std::endl;
						isError = true; break;
					} else if (feature->type == Data_File_ARFF_Feature::TYPE_STRING ||
					           feature->type == Data_File_ARFF_Feature::TYPE_UNSUPPORTED) {
						//  unsupported type: skip the COLUMN, keep loading everything else
						const int fileColumn = (int)columnMap.size();
						std::cout << "Skipping attribute '" << feature->name.c_str() << "' (file column "
						          << fileColumn+1 << "): unsupported type " << feature->typeName.c_str()
						          << " - FST4 features are numeric/nominal" << std::endl;
						skippedColumns.push_back(SkippedColumn{feature->name, feature->typeName, fileColumn});
						columnMap.push_back(-1);
						delete feature;
					} else {					//  loadable feature
						columnMap.push_back((int)attributes.size());
						this->attributes.push_back(feature);
						this->attributesCount ++;
					}
				}
			} else {
				//  a line outside the header that is neither a comment nor data: reported,
				//  but tolerated — the load continues with the next line
				if (line[0] != 0) {
					std::cout << "Error reading file - unexpected row : " << line << std::endl;
				}
			}

		}

		fclose(f);

		//  graceful degradation ends where nothing usable remains
		if (!isError && !skippedColumns.empty() && attributes.empty()) {
			std::cout << "No loadable attribute columns remain (all " << columnMap.size()
			          << " have unsupported types)" << std::endl;
			throw FST::fst_error("Data_File_ARFF: no loadable attributes (all column types unsupported).",1);
		}

		loaded = !isError;
		if (!loaded) return false; // the offending definition/row was reported above; the summary
		                           // below reads through getters that refuse an unloaded file
		std::cout <<  "Relation: " << this->relationName.c_str() << std::endl;
		std::cout <<  "Classes count: " << getNoOfClasses() << std::endl;
		std::cout <<  "Records count: " << rowsReaded << std::endl;
		if (!skippedColumns.empty()) {
			std::cout << "Skipped " << skippedColumns.size() << " of " << columnMap.size()
			          << " attribute columns (unsupported types, reported above); loaded feature"
			          << " indices follow the remaining columns in file order" << std::endl;
		}

		return !isError;
	}

	inline void Data_File_ARFF::printInfo() {
		if (!loaded) {std::cout << "File not loaded" << std::endl; return;};
		///////////  Debug info
		for (unsigned d=0; d < data.size(); d++) {
			char *dt = (char *)data[d]->getData();
			std::cout << "Record: cls=" << data[d]->getClassId() << " " ;
			for (unsigned i=0; i < attributes.size(); i++) {
				char *v = dt + attributes[i]->offset;
				if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NOMINAL) 
					std::cout << " " << *(short *)v;
				if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NUMERIC) 
					std::cout << " " << *(float*)v;
			}
			std::cout << std::endl;
		}

		std::cout <<  "Relation: " << this->relationName.c_str() << std::endl;
		std::cout <<  "Feature count: " << this->attributes.size() << std::endl;

		for (unsigned i=0; i < attributes.size(); i++) {
			std::cout << "  " << attributes[i]->name.c_str() << " : " << attributes[i]->type << std::endl;
			if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NOMINAL) {
				std::vector<std::string> vals = attributes[i]->values;
				for (unsigned j=0; j < vals.size(); j++) 
					std::cout << "    - " << vals[j].c_str() << std::endl;
			}
		}

		std::cout <<  "Classes count: " << getNoOfClasses() << " sizes: ";
		for (unsigned int i=0; i<getNoOfClasses() ; i++)
			std::cout << getClassName(i).c_str() << ": " << getClassSize(i) << "     ";
		std::cout << std::endl;


		std::cout <<  "Records count: " << data.size() << std::endl;

	}

	inline void Data_File_ARFF::prepareAttributes() {
		int size = 0;
		this->classAttribute = -1;

		for (unsigned fi = 0; fi < attributes.size(); fi ++ ) {
			Data_File_ARFF_Feature *feature = attributes[fi];
			feature->offset = size;
			size += feature->size;

			std::vector<char> tmp(feature->name.c_str(), feature->name.c_str()+feature->name.length()+1); // no VLA: MSVC hard error C2131
			mystrupr(tmp.data());
			if ((strcmp(tmp.data(), "CLASS")==0 || strcmp(tmp.data(), "'CLASS'")==0) && feature->type==Data_File_ARFF_Feature::TYPE_NOMINAL) {
				this->classAttribute = fi;
				feature->isClass = true;
			} else {
				feature->isClass = false;
				features.push_back(feature);
			}
		}

		if (classAttribute>=0) {
			classCount = attributes[classAttribute]->values.size();
			classSize = new int[classCount];
			for (int i=0;i<classCount;i++) classSize[i] = 0;
			std::cout << "Classes count : " << classCount << std::endl;
		}
		this->recordSize = size;
	}

	inline Data_File_ARFF_Feature *Data_File_ARFF::createAttribute(char *info) {
		info = trim(info);
		if (*info == 0) {std::cout << "empty definition!" << std::endl;return nullptr;}

		//  get the second part: the name ends at the first blank -- unless it is quoted ('my attr'
		//  or "my attr", which is how Weka writes names containing blanks), in which case it ends
		//  at the closing quote and the quotes are not part of the name
		char *val;
		if (*info == '\'' || *info == '"') {
			const char qc = *info; info++;
			val = info; while (*val != qc && *val != 0) val++;
			if (*val == 0) {std::cout << "syntax error (unterminated quoted attribute name)! - " << info << std::endl; return nullptr;}
			*val = 0; val++;
		} else {
			val = info+1; while(*val!=' ' && *val!=0 && *val!=9) val++;
			if (*val) {*val = 0; val++;}
		}
		val = trim(val);
		// info => name
		// val => type

		int type = 0;
		int size = 0;

		Data_File_ARFF_Feature *feature = nullptr;

		if (val[0]==0)  {std::cout << "empty datatype!" << std::endl;return nullptr;}
		if (val[0]=='{') {		///  Nominal type
			val++;
			char *p = val;
			std::vector<std::string> values;
			while (*val != '}' && *val !=0) {
				if (*val == ',') {
					*val = 0;
					std::vector<char> tmp(p, p+strlen(p)+1); // no VLA: MSVC hard error C2131
					values.push_back((std::string)trim(tmp.data()));
					p = ++val;
				}
				val++;
			}
			if (*val == 0) {std::cout << "syntax error! - " << info << std::endl;return nullptr;}
			*val = 0;
			std::vector<char> tmp(p, p+strlen(p)+1); // no VLA: MSVC hard error C2131
			values.push_back((std::string)trim(tmp.data()));

			feature = new Data_File_ARFF_Feature();
			feature->name = info;
			feature->size = sizeof(short);	//  short
			feature->type = Data_File_ARFF_Feature::TYPE_NOMINAL;
			feature->values = values;

		} else {
			val = mystrupr(val);
			if (strcmp(val, "REAL")==0 ||  strcmp(val, "NUMERIC")==0 || strcmp(val, "INTEGER")==0) {
				type = Data_File_ARFF_Feature::TYPE_NUMERIC;
				size = sizeof(float);
			} else if (strcmp(val, "STRING")==0) {
				// an unsupported-type MARKER, not a loadable feature: FST4 has no way to turn
				// free text into a numeric feature, so the caller skips the whole column
				type = Data_File_ARFF_Feature::TYPE_STRING;
				size = 0;
			} else {
				// unknown/unsupported type (DATE, RELATIONAL, typos...) — marker, same contract
				type = Data_File_ARFF_Feature::TYPE_UNSUPPORTED;
				size = 0;
			}
			feature = new Data_File_ARFF_Feature();
			feature->name = info;
			feature->typeName = val;
			feature->size = size;
			feature->type = type;
		}
		return feature;
	}

	inline bool Data_File_ARFF::readNextRow(char *row) {
		if (! *row) return false;		//  empty row;
		row = trim(row);
		Data_File_ARFF_Record *rec = nullptr;
		if (*row == '{') { 
			rec = this->readSparseRow(row);
		} else {
			rec = this->readNormalRow(row);
		}

		if (rec == nullptr) return false;
		
		if (classSize != nullptr && rec->getClassId()>=0) classSize[rec->getClassId()] ++;
		this->data.push_back(rec);
		return true;
	}

	inline bool Data_File_ARFF::readAttrValue(Data_File_ARFF_Feature *feature, Data_File_ARFF_Record *rec, char *value) {
		//  p points to single value for my feature
		char *fdata = (char *)rec->getData()+feature->offset;
		if (feature->type == Data_File_ARFF_Feature::TYPE_NUMERIC) {
			if (sscanf(value, "%f", (float *)fdata)!=1) {
				std::cout << "Error reading feature data - feature " << feature->name << ":"<< value << std::endl;return false;
			}
			if (feature->isClass) rec->setClassId((int)*((float*)fdata)); // slot holds a float (see sscanf above)
		} else if (feature->type == Data_File_ARFF_Feature::TYPE_NOMINAL) {
			std::vector<std::string> vals = feature->values;
			int val = -1;
			for (unsigned j=0; j < vals.size(); j++) 
				if (vals[j].compare(value)==0) {val = j; break;};
			if (val == -1) {
				std::cout << "Nominal value not found - " << feature->name.c_str() << " " << value << std::endl; return false;
			};
			*(short *)(fdata) = (short)val;

			if (feature->isClass) rec->setClassId(val);
		} else {
			std::cout << "Unknown attribute type - " << feature->name.c_str() << feature->type << std::endl; return false;
		}
		return true;
	}

	inline Data_File_ARFF_Record *Data_File_ARFF::readNormalRow(char *row) {
		Data_File_ARFF_Record *rec = new Data_File_ARFF_Record(recordSize);
		bool ok = true;

		//  iterate FILE columns; columnMap routes each token to its loaded
		//  attribute or discards it (skipped unsupported column)
		for (unsigned ci = 0; ok && ci < columnMap.size(); ci ++ ) {
			if (*row == 0) {	//  problem, row ended to soon
				std::cout << "Error reading data record - column " << ci << std::endl; ok = false; break;
			}
			//  quote-aware token scan: honor a leading ' or " so commas embedded in
			//  quoted values (typical for skipped STRING columns) don't split the token
			char *p = row;
			char *q = row; while (*q == ' ' || *q == 9) q++;
			if (*q == '\'' || *q == '"') {const char qc = *q; q++; while (*q != 0 && *q != qc) q++; if (*q) q++; row = q;}
			while (*row != 0 && *row != ',') row++;
			char tmpch = *row; *row = 0;

			const int ai = columnMap[ci];
			if (ai >= 0) {
				// tokens must be trimmed before nominal lookup — a space after a comma
				// would otherwise fail the exact string compare against the declared values
				char *tok = trim(p);
				// a value may be single- or double-quoted (Weka quotes nominal values freely,
				// and quoted numbers occur too): hand readAttrValue() the bare content
				{ const size_t L = strlen(tok); if (L >= 2 && (tok[0] == '\'' || tok[0] == '"') && tok[L-1] == tok[0]) {tok[L-1] = 0; tok = trim(tok+1);} }
				ok = ok && this->readAttrValue(attributes[ai], rec, tok);
			}	//  ai<0: unsupported column — token consumed and discarded

			*row = tmpch;
			if (*row == ',') row++;
		}

		if (! ok ) {delete rec; return nullptr;};
		return rec;
	}

	inline Data_File_ARFF_Record *Data_File_ARFF::readSparseRow(char *row) {

		Data_File_ARFF_Record *rec = new Data_File_ARFF_Record(recordSize);
		void *data = rec->getData();

		memset(data, 0, recordSize);
		bool ok = true;

		char *id;
		char *val;
		int attID = 0;

		row++;
		while (*row != 0 && *row != '}' && ok) {
			//  skip initial spaces
			while (*row == 9 || *row == ' ') row++;
			// start reading attrnum
			id = row;	
			while (*row != 9 && *row != ' ' && *row!=0) row++;
			if (*row == 0) {ok= false; break;};	//  line can't end here
			*row = 0; row++;
			// => id contains attr id

			//  skip spaces
			while (*row == 9 || *row == ' ') row++;
			val = row;
			if (*row == '"' || *row == '\'') { //  value is in commas
				row++;
				while (*row != 0 && *row != *val) row++; // scan to the matching closing quote (*val is the opening one)
				if (*row == 0) {ok= false; break;};	//  line can't end here

				*row = 0; val++; row++;
				while (*row == 9 || *row == ' ' || *row == ',') row++;
			} else {
				while (*row != ',' && *row != '}' && *row != 0) row++;
				if (*row == 0) {ok= false; break;};	//  line can't end here
				if (*row == '}') {*row=0; row++; *row = '}';}
				else {*row = 0; row++;};
				val = trim(val);
			}
			// => val contains attr value
			//	  row contains either ',' or '}'

			attID = atoi(id);	//  a FILE column index — remap through columnMap
			if (attID<0 || attID >= (int)columnMap.size()) {
				std::cout << "Invalid attribute index - " << attID << std::endl; ok = false; break;
			}

			const int ai = columnMap[attID];
			if (ai >= 0) {
				ok = ok && readAttrValue(attributes[ai], rec, val);
			}	//  ai<0: unsupported column — value ignored (reported once at header time)

			while (*row == 9 || *row == ' ') row++;
		}

		if (!ok) {delete rec; return nullptr;}

		// in sparse format an omitted attribute means value 0; if the class attribute
		// was omitted the record belongs to class 0 (its data slot is already zeroed
		// by the memset above)
		if (classAttribute >= 0 && rec->getClassId() == -1) rec->setClassId(0);

		return rec;
	}

	inline unsigned int Data_File_ARFF::getNoOfClasses() {
		if (!loaded) {throw fst_error("File not loaded");};

		if (classAttribute == -1) {std::cout << "class attribute not found" << std::endl; return 0;};
		if (attributes[classAttribute]->type != Data_File_ARFF_Feature::TYPE_NOMINAL) {
			// defensive only: prepareAttributes() never assigns classAttribute to a
			// non-nominal attribute, so this state is unreachable through load();
			// throw rather than return -1, which would wrap to a huge unsigned value
			throw fst_error("Class attribute is not nominal");
		}
		return attributes[classAttribute]->values.size();
	}

	inline unsigned int Data_File_ARFF::getTotalSize() {
		if (!loaded) {throw fst_error("File not loaded");};
		return data.size();
	}

	inline unsigned int Data_File_ARFF::getClassSize(int clsId) {
		if (!loaded) {throw fst_error("File not loaded");};
		// throw rather than return -1, which would wrap to a huge unsigned value;
		// all in-tree callers iterate over [0,getNoOfClasses()) so never trigger this
		if (classSize == nullptr) {throw fst_error("getClassSize: no class attribute loaded");};
		if (clsId < 0 || clsId >= classCount) {throw fst_error("getClassSize: class id out of range");};
		return classSize[clsId];
	}

	inline std::string Data_File_ARFF::getClassName(int clsId) {
		if (!loaded) {throw fst_error("File not loaded");};
		if (classSize == nullptr || clsId < 0 || clsId >= classCount) { return "-n/a-";};
		return attributes[classAttribute]->values[clsId];
	}


	inline void Data_File_ARFF::sortRecordsByClass() {
		if (!loaded) {throw fst_error("File not loaded");};
		if (classCount < 1) {throw fst_error("Classes not identified");};
		std::vector<Data_File_ARFF_Record*> **clsData = new std::vector<Data_File_ARFF_Record*>*[classCount];

		for (int c=0;c<classCount;c++) {
			clsData[c]= new std::vector<Data_File_ARFF_Record*>;
		}

		for (unsigned i=0;i<data.size();i++) {
			Data_File_ARFF_Record *rec = data[i];
			clsData[rec->getClassId()]->push_back(rec);
		}

		data.clear();
		for (int c=0;c<classCount;c++) {
			std::vector<Data_File_ARFF_Record*> *v =clsData[c];
			for (unsigned d=0;d<v->size();d++) {
				Data_File_ARFF_Record *rec = v->at(d);
				data.push_back(rec);
			}
			v->clear();
			delete v;
		}

		delete[] clsData;
	}

}

#endif
