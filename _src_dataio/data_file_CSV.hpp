#ifndef FSTDATAFILECSV_H
#define FSTDATAFILECSV_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    data_file_CSV.hpp
   \brief   Enables data accessor objects to read CSV (delimited text, e.g. Excel export) data files
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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "error.hpp"
#include "global.hpp"

namespace FST {

	/*! \brief CSV loader options
	    \note carried by the accessor so sharing clones stay self-contained */
	struct Data_File_CSV_Options {
		std::string class_column;	//!< which column carries the class label: "" = the last column; a decimal integer = that 0-based file column index; anything else = a header column name (which implies a header row)
		std::string delimiter;		//!< field separator: "auto" (detect from the first line), "comma", "semicolon" or "tab"
		std::string header;			//!< whether the first line is a header row: "auto" (detect), "yes" or "no"
		bool missing_to_active;		//!< when true, missing FEATURE tokens are replaced by #missing_to and the row is kept instead of skipped; a missing CLASS label still skips the row, since a row without a label cannot be assigned to any class (the fst4 tool sets this from `--csv-missing-to`)
		float missing_to;			//!< the numeric code that replaces a missing feature value when #missing_to_active. The mapping is purely a substitution: to have the code imputed away, configure the accessor's missing-value substitution with the SAME code; leave that unset to keep the code in the data as an explicit missingness indicator
		bool missing_to_unpaired;	//!< set by the caller when no missing-value substitution is configured; the load summary then names the pairing option, so that mapping-without-imputing reads as a deliberate choice rather than an oversight
		Data_File_CSV_Options() : delimiter("auto"), header("auto"), missing_to_active(false), missing_to(0), missing_to_unpaired(false) {}
	};

	/*! \brief CSV (delimited text) data format filter — the simplest spreadsheet-to-FST4 path

	    Dialect handling: delimiter auto-detected on the first line (comma/semicolon/tab, counted
	    outside double quotes); decimal COMMAS accepted whenever the delimiter is not the comma,
	    so a European-locale spreadsheet export loads without any options; a UTF-8 byte-order mark
	    at the start of the file is stripped; double-quote quoting with "" as the escaped quote;
	    header row auto-detected (any non-numeric non-class field in line 1 means a header).
	    Class labels are arbitrary tokens mapped to ids in first-appearance order.

	    The file has no comment syntax: every line that is not blank (or pure whitespace) is either
	    the header or a data row, and must have exactly the same number of columns as the first line.

	    Missing values. A field reading "", ?, NA, N/A, NaN or NULL (case-insensitively) is missing.
	    By default the whole row is skipped, with a message naming the row and column, and the load
	    summary reports how many rows went that way. With Data_File_CSV_Options::missing_to_active
	    a missing FEATURE value is instead replaced by a numeric code and the row is kept; a missing
	    CLASS label still skips the row. A malformed row -- wrong column count, or a feature token
	    that is neither a number nor a missing marker -- is fatal and names the line number. */
	class Data_File_CSV {
	private:
		std::string _fileName;  //!< input .csv/.tsv file name, as passed to the constructor
		Data_File_CSV_Options opts;  //!< dialect/class-column/missing-value options, as passed to the constructor

		char delim;								//!< resolved delimiter character (comma/semicolon/tab), set by resolveDialect()
		bool hasHeader;							//!< resolved header presence, set by resolveClassAndHeader()
		int classColumn;						//!< resolved 0-based file column of the class label, set by resolveClassAndHeader()
		unsigned int columns;					//!< total file columns (features + class), set by resolveClassAndHeader()
		std::vector<std::string> headerNames;	//!< the header row's tokens, empty when hasHeader is false

		std::vector<std::vector<float> > rows;	//!< loaded records; each row holds feature values only, in non-class file-column order
		std::vector<int> classIds;				//!< each record's class id, parallel to #rows
		std::vector<std::string> classNames;	//!< class label text, indexed by class id (assigned in first-appearance order)
		std::vector<unsigned int> classSizes;  //!< per-class record counts, indexed by class id, parallel to #classNames
		unsigned int skippedRows;  //!< number of data rows skipped due to a missing value (see the class description's missing-value paragraph)
		unsigned int mappedCells;				//!< number of missing feature values replaced by Data_File_CSV_Options::missing_to
		unsigned int mappedRows;				//!< data rows carrying at least one replaced value
		bool loaded;  //!< whether prepareFile() completed without error; getters that need loaded data throw fst_error while false
	public:
		//! \brief Constructs an (unloaded) reader for \p fileName under options \p _opts; call prepareFile() to actually read it.
		Data_File_CSV(std::string fileName, const Data_File_CSV_Options &_opts) : _fileName(fileName), opts(_opts),
			delim(','), hasHeader(false), classColumn(-1), columns(0), skippedRows(0), mappedCells(0), mappedRows(0), loaded(false) {}

		//! \brief Reads and parses the whole CSV/TSV file: BOM-strip + blank-line skip, dialect/class-column/header resolution off line 1
		//! (resolveDialect()/resolveClassAndHeader()), then every data row (splitLine(), missing-value handling, parseNumeric()).
		//! \return true (a malformed row, wrong column count, or unreadable file all throw fst_error instead); sets #loaded.
		bool prepareFile();				//  reads and parses the whole file
		//! \brief Regroups #rows/#classIds into class-id order (stable within each class), mirroring Data_File_ARFF::sortRecordsByClass().
		//! \throws fst_error if the file was not loaded or no classes were identified.
		void sortRecordsByClass();		//  regroups records into class-id order (stable)

		unsigned int getNoOfClasses() {if (!loaded) throw fst_error("File not loaded"); return (unsigned int)classNames.size();} //!< number of classes (#classNames's size); throws fst_error if not loaded
		unsigned int getNoOfFeatures() {if (!loaded) throw fst_error("File not loaded"); return columns-1;} //!< data dimensionality (#columns minus the class column); throws fst_error if not loaded
		unsigned int getTotalSize() {if (!loaded) throw fst_error("File not loaded"); return (unsigned int)rows.size();} //!< number of loaded records (#rows's size); throws fst_error if not loaded
		unsigned int getClassSize(int clsId) {if (!loaded) throw fst_error("File not loaded"); if (clsId<0 || clsId>=(int)classSizes.size()) throw fst_error("getClassSize: class id out of range"); return classSizes[clsId];} //!< number of records in class \p clsId; throws fst_error if not loaded or \p clsId is out of range (rather than returning -1 wrapped to a huge unsigned value; in-tree callers iterate over [0,getNoOfClasses()) so never trigger this)
		std::string getClassName(int clsId) {if (!loaded) throw fst_error("File not loaded"); if (clsId<0 || clsId>=(int)classNames.size()) return "-n/a-"; return classNames[clsId];} //!< class \p clsId's label text, or "-n/a-" if \p clsId is out of range; throws fst_error if not loaded
		//! \brief the (already-parsed) value of feature \p feature in record \p record
		//! \warning Unchecked: \p record must be in [0,getTotalSize()) and \p feature in
		//! [0,getNoOfFeatures()), and the file must be loaded. Out-of-range arguments are undefined behaviour.
		float getFeatureValue(long record, int feature) {return rows[record][feature];}
	private:
		///  reads one full line of arbitrary length, growing the buffer as needed
		///  (the ARFF readLine pattern — no line-length cap); false at EOF with nothing read
		bool readLine( FILE *f, std::vector<char> &buff ) {
			size_t len = 0; buff[0] = 0;
			while (fgets(buff.data()+len, (int)(buff.size()-len), f) != nullptr) {
				len += strlen(buff.data()+len);
				if ((len > 0 && buff[len-1] == '\n') || feof(f)) return true;
				if (len >= buff.size()-2) buff.resize(2*buff.size());
			}
			return len > 0;
		}

		//! \brief Strips leading/trailing spaces, tabs and CRs from \p s (a single already-split field token).
		static std::string trimTok(const std::string &s) {
			size_t b = 0, e = s.size();
			while (b < e && (s[b]==' ' || s[b]=='\t' || s[b]=='\r')) b++;
			while (e > b && (s[e-1]==' ' || s[e-1]=='\t' || s[e-1]=='\r')) e--;
			return s.substr(b, e-b);
		}

		///  quote-aware field split: the delimiter inside "..." does not split,
		///  "" inside quotes is a literal quote (the Excel convention)
		std::vector<std::string> splitLine(const char *line) const {
			std::vector<std::string> out; std::string cur; bool inq = false;
			for (const char *p = line; *p; p++) {
				const char c = *p;
				if (inq) { if (c=='"') { if (p[1]=='"') {cur += '"'; p++;} else inq = false; } else cur += c; }
				else if (c=='"') inq = true;
				else if (c==delim) { out.push_back(trimTok(cur)); cur.clear(); }
				else cur += c;
			}
			out.push_back(trimTok(cur));
			return out;
		}

		//! \brief Whether \p tok (case-insensitively) is one of the recognized missing-value markers: empty, "?", "NA", "N/A", "NaN", "NULL".
		bool isMissing(const std::string &tok) const {
			std::string u; u.reserve(tok.size());
			for (size_t i = 0; i < tok.size(); i++) u += (char)toupper((unsigned char)tok[i]);
			return u=="" || u=="?" || u=="NA" || u=="N/A" || u=="NAN" || u=="NULL";
		}

		///  strict full-token numeric parse: the whole token must be consumed (trailing spaces and
		///  tabs aside), so "1.5x" or "1 2" fail. Commas are read as the decimal point, but only
		///  when the delimiter is not the comma AND the token contains no '.' at all -- a token
		///  that already has a decimal point keeps its commas, which then make the parse fail
		///  rather than silently reinterpreting a thousands separator.
		bool parseNumeric(const std::string &tok, float &out) const {
			if (tok.empty()) return false;
			std::string t = tok;
			if (delim != ',' && t.find('.') == std::string::npos)
				for (size_t i = 0; i < t.size(); i++) if (t[i]==',') t[i]='.';
			const char *s = t.c_str(); char *end = nullptr;
			const double v = strtod(s, &end);
			if (end == s) return false;
			while (*end==' ' || *end=='\t') end++;
			if (*end) return false;
			out = (float)v; return true;
		}

		//! \brief Resolves #delim from opts.delimiter, or auto-detects it from \p firstLine (candidates counted outside double-quoted spans).
		//! \note Auto tie-break is deliberately comma-LAST (tab > semicolon > comma): in a European-locale export
		//! every data line carries about as many decimal commas as delimiters, and those inflate only the comma
		//! count, never the tab or semicolon count -- so a candidate that merely ties the comma count should win.
		void resolveDialect(const char *firstLine);
		//! \brief Resolves #columns, #classColumn and #hasHeader from the first line's split tokens \p toks (also #headerNames when a header is present).
		//! Class column: opts.class_column empty = last column, a parseable integer = that 0-based index, otherwise = a header column name
		//! (looked up case-insensitively once the header is confirmed present). Header presence: forced by opts.header ("yes"/"no") or by naming
		//! the class column by name; otherwise auto-detected as present iff any non-class field on line 1 fails a numeric-or-missing parse.
		//! \throws fst_error if \p toks has fewer than 2 columns, a numeric class-column index is out of range, or a by-name class column isn't found in the header.
		void resolveClassAndHeader(const std::vector<std::string> &toks);
	};


	inline void Data_File_CSV::resolveDialect(const char *firstLine) {
		if (opts.delimiter=="comma") delim = ',';
		else if (opts.delimiter=="semicolon") delim = ';';
		else if (opts.delimiter=="tab") delim = '\t';
		else {	//  auto: count candidates outside double quotes, comma-last tie-break (see the \note
			//  on the declaration for why a tie must not go to the comma)
			int cnt[3] = {0,0,0}; bool inq = false;	//  [0]=tab [1]=semicolon [2]=comma
			for (const char *p = firstLine; *p; p++) {
				if (*p=='"') inq = !inq;
				else if (!inq) { if (*p=='\t') cnt[0]++; else if (*p==';') cnt[1]++; else if (*p==',') cnt[2]++; }
			}
			int best = 0; if (cnt[1] > cnt[best]) best = 1; if (cnt[2] > cnt[best]) best = 2;
			delim = (best==0) ? '\t' : (best==1) ? ';' : ',';
		}
	}

	inline void Data_File_CSV::resolveClassAndHeader(const std::vector<std::string> &toks) {
		columns = (unsigned int)toks.size();
		if (columns < 2) throw fst_error("Data_File_CSV: fewer than 2 columns found on the first line (need at least one feature column and the class column; check --csv-delimiter).",1);

		//  class column spec: "" = last, integer = 0-based index, otherwise = header name
		bool byName = false;
		if (opts.class_column.empty()) classColumn = (int)columns-1;
		else {
			char *end = nullptr;
			const long ix = strtol(opts.class_column.c_str(), &end, 10);
			if (end != opts.class_column.c_str() && *end == 0) {
				if (ix < 0 || ix >= (long)columns) throw fst_error("Data_File_CSV: --csv-class index "+opts.class_column+" out of range (file has "+std::to_string((long long)columns)+" columns, 0-based).",1);
				classColumn = (int)ix;
			} else byName = true;	//  resolved below, once the header row is confirmed
		}

		//  header presence
		if (byName && opts.header=="no") throw fst_error("Data_File_CSV: --csv-class names a header column but --csv-header is 'no'.",1);
		if (byName || opts.header=="yes") hasHeader = true;
		else if (opts.header=="no") hasHeader = false;
		else {	//  auto: line 1 is a header iff any non-class field fails numeric-or-missing parse
			hasHeader = false;
			for (unsigned int ci = 0; ci < columns; ci++) {
				if ((int)ci == classColumn) continue;
				float v;
				if (!isMissing(toks[ci]) && !parseNumeric(toks[ci], v)) { hasHeader = true; break; }
			}
		}

		if (hasHeader) {
			headerNames = toks;
			if (byName) {
				classColumn = -1;
				for (unsigned int ci = 0; ci < columns; ci++) {
					std::string a, b;
					for (size_t i = 0; i < toks[ci].size(); i++) a += (char)toupper((unsigned char)toks[ci][i]);
					for (size_t i = 0; i < opts.class_column.size(); i++) b += (char)toupper((unsigned char)opts.class_column[i]);
					if (a == b) { classColumn = (int)ci; break; }
				}
				if (classColumn < 0) {
					std::cout << "Header columns:"; for (unsigned int ci = 0; ci < columns; ci++) std::cout << " '" << headerNames[ci].c_str() << "'"; std::cout << std::endl;
					throw fst_error("Data_File_CSV: --csv-class column '"+opts.class_column+"' not found in the header (available names listed above).",1);
				}
			}
		}
	}

	inline bool Data_File_CSV::prepareFile() {
		FILE *f = fopen(_fileName.c_str(), "rt");
		if (f == nullptr) throw fst_error("Data_File_CSV file read problem.",1);
		std::vector<char> buff(8192);	//  grown by readLine() as needed — no line-length cap

		bool first = true;
		long lineNo = 0, dataRowNo = 0;
		while (readLine(f, buff)) {
			lineNo++;
			char *line = buff.data();
			if (first && (unsigned char)line[0]==0xEF && (unsigned char)line[1]==0xBB && (unsigned char)line[2]==0xBF) line += 3;	//  Excel UTF-8 BOM
			size_t len = strlen(line);
			while (len > 0 && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len] = 0;
			{	//  a line of pure whitespace/delimiter residue counts as empty (Excel emits trailing ",,,"
				//  lines for deleted rows); on line 1 the delimiter is not resolved yet, so every candidate
				//  delimiter counts as residue there
				bool blank = true;
				for (size_t i = 0; i < len; i++) {
					const char ch = line[i];
					if (ch==' ' || ch=='\t') continue;
					if (!first && ch==delim) continue;
					if (first && (ch==',' || ch==';')) continue;
					blank = false; break;
				}
				if (blank) continue;
			}

			if (first) {
				first = false;
				resolveDialect(line);
				std::vector<std::string> toks = splitLine(line);
				resolveClassAndHeader(toks);
				if (hasHeader) continue;	//  line 1 consumed as the header row
				//  no header: line 1 is data — fall through and parse it below
			}

			std::vector<std::string> toks = splitLine(line);
			if (toks.size() != columns) {
				fclose(f);
				throw fst_error("Data_File_CSV: line "+std::to_string((long long)lineNo)+" has "+std::to_string((long long)toks.size())+" columns, expected "+std::to_string((long long)columns)+" (check --csv-delimiter and quoting).",1);
			}
			dataRowNo++;

			//  missing-value scan first: by default the whole row is skipped, loudly; with
			//  missing_to_active the missing FEATURE tokens are replaced by the code in the parse
			//  loop below and the row is kept — only a missing CLASS label still skips (a row
			//  without a label cannot be assigned to any class)
			{
				int missCol = -1;
				for (unsigned int ci = 0; ci < columns; ci++) if (isMissing(toks[ci])) { missCol = (int)ci; break; }
				if (missCol >= 0 && !opts.missing_to_active) {
					std::cout << "Skipping data row " << dataRowNo << " (line " << lineNo << "): missing value in column " << missCol+1 << std::endl;
					skippedRows++;
					continue;
				}
				if (missCol >= 0 && isMissing(toks[classColumn])) {
					std::cout << "Skipping data row " << dataRowNo << " (line " << lineNo << "): missing class label in column " << classColumn+1 << " (--csv-missing-to maps feature values only)" << std::endl;
					skippedRows++;
					continue;
				}
			}

			std::vector<float> vals; vals.reserve(columns-1);
			unsigned int rowMapped = 0;
			for (unsigned int ci = 0; ci < columns; ci++) {
				if ((int)ci == classColumn) continue;
				float v;
				if (opts.missing_to_active && isMissing(toks[ci])) { v = opts.missing_to; rowMapped++; }	//  keep the row, code the cell
				else if (!parseNumeric(toks[ci], v)) {	//  malformed stays fatal (not a missing marker — those were skipped above or mapped here)
					fclose(f);
					throw fst_error("Data_File_CSV: line "+std::to_string((long long)lineNo)+" column "+std::to_string((long long)(ci+1))+": feature value '"+toks[ci]+"' is not numeric.",1);
				}
				vals.push_back(v);
			}
			if (rowMapped) { mappedCells += rowMapped; mappedRows++; }

			//  class label: exact trimmed token, id by first appearance
			const std::string &cls = toks[classColumn];
			int id = -1;
			for (unsigned int k = 0; k < classNames.size(); k++) if (classNames[k] == cls) { id = (int)k; break; }
			if (id < 0) { id = (int)classNames.size(); classNames.push_back(cls); classSizes.push_back(0); }
			classSizes[id]++;
			rows.push_back(std::move(vals));
			classIds.push_back(id);
		}
		fclose(f);

		if (rows.empty()) throw fst_error("Data_File_CSV: no data rows loaded"+std::string(skippedRows ? " (all rows had missing values)" : "")+".",1);

		loaded = true;
		std::cout << "CSV file: " << _fileName.c_str()
		          << " (delimiter " << (delim=='\t' ? "TAB" : delim==';' ? "';'" : "','")
		          << ", header " << (hasHeader ? "yes" : "no")
		          << ", class column " << classColumn+1;
		if (hasHeader) std::cout << " '" << headerNames[classColumn].c_str() << "'";
		std::cout << " of " << columns << ")" << std::endl;
		std::cout << "Classes count: " << classNames.size() << std::endl;
		for (unsigned int k = 0; k < classNames.size(); k++)
			std::cout << "  class " << k << " = '" << classNames[k].c_str() << "' (" << classSizes[k] << " records)" << std::endl;
		std::cout << "Records count: " << rows.size() << std::endl;
		if (skippedRows)
			std::cout << "Skipped " << skippedRows << " of " << dataRowNo << " data rows (missing values, reported above)" << std::endl;
		if (mappedRows) {
			std::cout << "Mapped " << mappedCells << " missing value(s) in " << mappedRows << " of " << dataRowNo << " data rows to code " << opts.missing_to << " (--csv-missing-to)";
			//  unpaired = the code stays in the data as a missingness indicator; legitimate (a tree
			//  wrapper can split on it) but easy to reach by forgetting, so name the pairing option
			if (opts.missing_to_unpaired) std::cout << "; the code is left in the data - add --missing-value " << opts.missing_to << " to substitute per-feature training averages instead";
			std::cout << std::endl;
		}
		std::cout << "Features count: " << columns-1 << " (feature indices follow the non-class columns in file order)" << std::endl;

		return true;
	}

	inline void Data_File_CSV::sortRecordsByClass() {
		if (!loaded) throw fst_error("File not loaded");
		if (classNames.empty()) throw fst_error("Classes not identified");
		std::vector<std::vector<float> > newRows; newRows.reserve(rows.size());
		std::vector<int> newIds; newIds.reserve(classIds.size());
		for (unsigned int c = 0; c < classNames.size(); c++)
			for (unsigned int i = 0; i < rows.size(); i++)
				if (classIds[i] == (int)c) { newRows.push_back(std::move(rows[i])); newIds.push_back((int)c); }
		rows.swap(newRows);
		classIds.swap(newIds);
	}

}

#endif
