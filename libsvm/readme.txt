Feature Selection Toolbox 4 optionally takes use of the
LibSVM library to enable SVM based wrappers (i.e., feature
selection directly optimizing SVM classification accuracy).

LibSVM is NOT distributed with FST4 (licensing policy: FST4
sources contain no third-party code). To enable SVM support,
place into this directory the files svm.cpp and svm.h from
LibSVM version 3.37 (or a later drop-in compatible release),
downloadable from:

http://www.csie.ntu.edu.tw/~cjlin/libsvm/

The CMake build auto-detects libsvm/svm.cpp: when present, the
fst4 CLI is built with -DFST_SVM and the SVM components become
available; when absent, configuration proceeds without SVM
support (equivalent to -DFST4_WITH_SVM=OFF) and the SVM demo
scenarios are skipped. FST4 touches only LibSVM's stable public
API (see _src_criteria/classifier_svm.hpp), which is why newer
LibSVM versions are expected to remain drop-in compatible
(verified for 2.91 through 3.37).

Remark 1: Note for golden-test reproducibility: the tests/
golden outputs for SVM scenarios were captured with LibSVM 3.37.
A different LibSVM version may produce different SVM results
and thus golden mismatches in the SVM scenarios (those whose
manifest command uses --criterion wrapper-svm).

Remark 2: LibSVM by default prints detailed logs in the
course of classifier training. Together with FST4 messages
this leads to cluttered output, and the SVM golden scenarios
assume the logging is off. After dropping in svm.cpp, disable
the function static void info(const char *fmt,...) in
svm.cpp by changing the respective
#if 1
directive in front of static void info(const char *fmt,...) to
#if 0
(valid at least in versions 2.91 through 3.37 of LibSVM)
