# FST4 — Feature Selection Toolbox 4
#
# Build rules:
#   - the library itself is header-only; _src_global/global.cpp is an empty
#     compatibility stub, still compiled (plain and FST_THREADED) and linked into
#     every program
#   - Release flags follow a two-preset policy: the default is a fixed,
#     CPU-independent "golden" flag set — the one the regression baselines in
#     tests/ are captured under (byte-exact output comparison depends on it, so
#     do NOT change it without recapturing every baseline);
#     -DFST4_NATIVE_OPT=ON opts into machine-specific native optimization, which
#     is NOT valid for the regression gate
#   - Debug builds define DEBUG and carry a "d" executable suffix
#   - LibSVM-dependent targets appear only when libsvm/svm.cpp is present
#     (FST4_WITH_SVM); fst4 itself degrades to an SVM-less build
#
# Header dependencies are tracked automatically per translation unit, so
# incremental builds are exact.
#
# Typical use on Windows (from PowerShell, with MinGW-W64 on PATH; the "MinGW
# Makefiles" generator refuses to configure when Git Bash's sh.exe is on PATH):
#   cmake -S . -B build-win -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-win --parallel
#   FST_BIN=build-win/bin sh tests/golden.sh check     (from Git Bash)

cmake_minimum_required(VERSION 3.16)
project(FST4 VERSION 4.2.0 LANGUAGES CXX)

# --- language & flags --------------------------------------------------------

# C++17, no GNU extensions.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The regression baselines are captured from optimized builds; default to Release so
# a bare configure cannot silently produce non-comparable binaries.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Two-preset flag policy.
# OFF (default) = the "golden" preset: fixed, CPU-independent codegen, identical on
# every architecture — machine-portable binaries within a platform+arch, and the flag
# set the regression baselines in tests/ are captured under, so a default build is
# valid for the regression gate with no special knowledge. ON = native CPU
# optimization for speed experiments — machine-specific and NOT gate-valid
# (tests/golden.sh refuses capture from such a build and warns on check).
# Reproducible-by-default is deliberate; GCC bug 54412 (below) also makes
# -march=native actively hazardous on Windows.
option(FST4_NATIVE_OPT "native CPU optimization: faster, machine-specific, not golden-gate-valid (OFF = the fixed golden codegen)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(FST4_NATIVE_OPT)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
      # The native flag set incl. the sse,387 x87 mix — kept exactly, since results of
      # earlier native-optimized runs are reproducible only under these flags.
      set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -funroll-loops -mfpmath=sse,387 -DNDEBUG")
      if(WIN32)
        # GCC bug 54412: on Windows/SEH gcc cannot realign the stack beyond 16 bytes,
        # yet -march=native emits 32-byte-aligned ymm spills without realignment —
        # crashes by caller stack parity.
        # Capping vector width at 128 bits removes
        # every 256-bit stack slot. Needed ONLY under -march=native — the golden
        # preset never emits ymm spills, which is one of the reasons it exists.
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mprefer-vector-width=128")
      endif()
    else()
      # native on non-x86 (Apple Silicon etc.): -mcpu is the arm spelling; FP
      # contraction left at the compiler default (fast) — this preset trades
      # reproducibility-across-machines for speed by definition.
      set(CMAKE_CXX_FLAGS_RELEASE "-O2 -mcpu=native -funroll-loops -DNDEBUG")
    endif()
  else()
    # The "golden" preset, identical on every architecture: no -march/-mcpu=native
    # (CPU-independent codegen), no x87 (-mfpmath=sse is the x86-64 default), and
    # -ffp-contract=off pins FMA contraction, which would otherwise differ per arch
    # and per compiler. Every shipped baseline (tests/golden-*) is captured under
    # this preset.
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -funroll-loops -ffp-contract=off -DNDEBUG")
  endif()
  set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
  if(WIN32)
    # fst4.cpp at -O0 exceeds the COFF 32k-section limit (the MSVC /bigobj twin).
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wa,-mbig-obj")
  endif()
elseif(MSVC)
  # The templated headers overflow COFF section limits without /bigobj.
  add_compile_options(/bigobj)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
endif()

# Debug builds define DEBUG (in addition to the absence of NDEBUG).
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

# --- LibSVM (optional drop-in; not distributed with FST4, see libsvm/readme.txt) ---

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libsvm/svm.cpp)
  option(FST4_WITH_SVM "Build LibSVM-backed targets (drop-in libsvm/)" ON)
else()
  set(FST4_WITH_SVM OFF)
  message(STATUS "FST4: libsvm/svm.cpp not found - SVM-dependent targets disabled")
endif()

# --- optional Eigen-backed Indexed_Matrix numerics ----------------------------

# Swaps the hand-rolled Crout LU/inverse/determinant in indexed_matrix.hpp for
# Eigen::PartialPivLU (targets the documented Divergence instability and
# cross-validates the hand-rolled numerics). Results differ in low-order FP bits,
# so golden byte-comparison does NOT apply to Eigen builds — keep OFF for gate runs.
option(FST4_USE_EIGEN "Eigen-backed Indexed_Matrix LU/inverse/determinant" OFF)
set(FST4_EIGEN_DIR "" CACHE PATH "Eigen source root containing Eigen/Dense (header-only; e.g. /opt/eigen-5.0.1 or C:/tools/eigen-5.0.1)")
if(FST4_USE_EIGEN AND NOT EXISTS "${FST4_EIGEN_DIR}/Eigen/Dense")
  message(FATAL_ERROR "FST4_USE_EIGEN requires -DFST4_EIGEN_DIR=<path> pointing at an Eigen source root (Eigen/Dense not found under '${FST4_EIGEN_DIR}')")
endif()

# --- header library ----------------------------------------------------------

# The header-only library, plus the vendored CLI11 and the optional LibSVM drop-in.
add_library(fst4_headers INTERFACE)
target_include_directories(fst4_headers INTERFACE
  _src_global _src_dataio _src_criteria _src_search)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libsvm)
  target_include_directories(fst4_headers INTERFACE libsvm)
endif()
target_include_directories(fst4_headers INTERFACE cli11)
if(FST4_USE_EIGEN)
  target_include_directories(fst4_headers INTERFACE ${FST4_EIGEN_DIR})
  # EIGEN_MPL2_ONLY: fail the build if anything outside Eigen's MPL2 core is pulled in
  target_compile_definitions(fst4_headers INTERFACE FST_USE_EIGEN EIGEN_MPL2_ONLY)
endif()

# global.cpp is an empty compatibility stub (the library is header-only; notify()
# is inline in global.hpp); it is still built in a plain and a threaded variant and
# linked into every program.
add_library(fst4_global OBJECT _src_global/global.cpp)
target_link_libraries(fst4_global PRIVATE fst4_headers)

add_library(fst4_global_threaded OBJECT _src_global/global.cpp)
target_compile_definitions(fst4_global_threaded PRIVATE FST_THREADED)
target_link_libraries(fst4_global_threaded PRIVATE fst4_headers Threads::Threads)

if(FST4_WITH_SVM)
  # Compiled without FST_THREADED even for threaded consumers (LibSVM does not use it).
  add_library(fst4_libsvm OBJECT libsvm/svm.cpp)
  target_link_libraries(fst4_libsvm PRIVATE fst4_headers)
endif()

# --- executables -------------------------------------------------------------

# fst4_add_program(<name> [THREADED] [SVM]): one root-level <name>.cpp linked
# with the matching global.cpp variant (and LibSVM when requested). SVM targets
# are silently skipped when FST4_WITH_SVM is off. SOURCE <path> overrides the
# default <name>.cpp location.
function(fst4_add_program name)
  cmake_parse_arguments(ARG "THREADED;SVM" "SOURCE" "" ${ARGN})
  if(ARG_SVM AND NOT FST4_WITH_SVM)
    return()
  endif()
  if(NOT ARG_SOURCE)
    set(ARG_SOURCE ${name}.cpp)
  endif()
  if(ARG_THREADED)
    add_executable(${name} ${ARG_SOURCE} $<TARGET_OBJECTS:fst4_global_threaded>)
    target_compile_definitions(${name} PRIVATE FST_THREADED)
    target_link_libraries(${name} PRIVATE Threads::Threads)
  else()
    add_executable(${name} ${ARG_SOURCE} $<TARGET_OBJECTS:fst4_global>)
  endif()
  target_link_libraries(${name} PRIVATE fst4_headers)
  if(ARG_SVM)
    target_sources(${name} PRIVATE $<TARGET_OBJECTS:fst4_libsvm>)
  endif()
  # Debug builds produce demoNNd etc.
  set_target_properties(${name} PROPERTIES DEBUG_POSTFIX d)
endfunction()

# Compiled demo exemplars: the library-as-library usage examples. All other demos
# ship as configuration files of the fst4 tool (run: fst4 --config demos/demoNN.cfg).
fst4_add_program(demo10 SOURCE demos/demo10.cpp)              # simplest filter workflow (SFS + gmahalanobis)
fst4_add_program(demo52t THREADED SOURCE demos/demo52t.cpp)   # threaded wrapper + tracker regularization
fst4_add_program(demo55 SOURCE demos/demo55.cpp)              # two-scenario stability/similarity choreography

# smoke tests (type-erasure and generalized-step probes; part of the regression gate)
fst4_add_program(smoke_type_erasure THREADED SOURCE tests/smoke_type_erasure.cpp)
fst4_add_program(smoke_generalized_step THREADED SOURCE tests/smoke_generalized_step.cpp)

# fst4 CLI: threaded so both evaluators are runtime-selectable; FST_SVM enables
# the wrapper-svm factory when LibSVM is available.
fst4_add_program(fst4 THREADED)
if(FST4_WITH_SVM)
  target_compile_definitions(fst4 PRIVATE FST_SVM)
  target_sources(fst4 PRIVATE $<TARGET_OBJECTS:fst4_libsvm>)
endif()

# --- aggregate targets ---------------------------------------------------------

# demos = the three compiled exemplars; demos_all = everything the regression gate
# (tests/golden.sh) needs.
set(FST4_GROUP_demos demo10 demo52t demo55)
set(FST4_GROUP_smoke_tests smoke_type_erasure smoke_generalized_step)

foreach(group demos smoke_tests)
  set(members)
  foreach(t IN LISTS FST4_GROUP_${group})
    if(TARGET ${t})
      list(APPEND members ${t})
    endif()
  endforeach()
  add_custom_target(${group} DEPENDS ${members})
endforeach()

add_custom_target(demos_all DEPENDS demos smoke_tests fst4)
