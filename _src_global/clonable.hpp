#ifndef FSTCLONABLE_H
#define FSTCLONABLE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    clonable.hpp
   \brief   Defines interface for classes that may need to get cloned
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

#include <cassert>
#include <iostream>
#include <memory>
#include <typeinfo>
#include <type_traits> // for the contract static_asserts in Clonable_Impl
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

class Clonable;

std::ostream& operator<<(std::ostream& os, const Clonable& cl);

/*! \brief Abstract class, defines interface for classes that may need to get cloned
    
    (defined especially to support threading in Sequential_Step specializations)
    \note Note that clonable objects may appear in threaded code - their output
    thus should be guarded using syncout class (see global.*pp)
*/  
class Clonable { // abstract class
	//! Streams a clonable object's print() to \p os.
	friend std::ostream& operator<<(std::ostream& os, const Clonable& cl);
public:
	Clonable(): _clone(false), _sharing(false) {}
	virtual ~Clonable() {}

	/*! create 1:1 independent clone of the current object.
	    The caller owns the returned object. Families that cannot support a given clone kind
	    implement it to throw fst_error; Clonable_Impl's CLONE_INHERIT slot defers to such a body. */
	virtual Clonable* clone() const =0;

	/*! create equivalent clone of the current object,
	    permitting read-only access to structures in the source
	    object (allows referencing instead of copying of large memory structures).
	    may be faster and save space but requires more caution with respect to concurrency
	    Use example: Data_Accessor memory data representation cloning */
	virtual Clonable* sharing_clone() const =0;

	/*! create clone of the current object, ignoring
	    internal temporary structures to save speed.
	    Does not replicate exact object state. The clone
	    must be used carefully in a way that ensures
	    internal structure re-initialization
	    Use example: Data_Splitter cloning or
	    Classifier_LIBSVM cloning due to inability to clone
	    external structures defined in LibSVM */
	virtual Clonable* stateless_clone() const =0;

	/*! check whether this instance is a clone */
	bool is_clone() const {return _clone;}
	bool is_sharing() const {return _sharing;} //!< true if this instance is a sharing_clone() result (referencing, not owning, the source's structures)
	bool is_sharing_clone() const {return _sharing && _clone;} //!< true if this instance is BOTH a clone and sharing -- i.e. specifically a sharing_clone() result, not a clone()/stateless_clone() one

	virtual std::ostream& print(std::ostream& os) const {return os;} //!< writes a human-readable identification of the object; base default writes nothing
protected:
	void set_cloned(const bool cloned=true) {_clone=cloned;} //!< marks/unmarks this instance as a clone; call from clone()/stateless_clone() overrides (Clonable_Impl does this automatically)
	void set_sharing(const bool sharing=true) {_sharing=sharing;} //!< marks/unmarks this instance as sharing (see is_sharing())
	void set_sharing_cloned(const bool cloned=true, const bool sharing=true) {_clone=cloned; _sharing=sharing;} //!< sets both flags at once; call from sharing_clone() overrides (Clonable_Impl does this automatically)
private:
	/*! cloning indicator, should be set inside clone() and weak_clone() */
	bool _clone;
	bool _sharing; //!< set by set_sharing()/set_sharing_cloned(); read by is_sharing()/is_sharing_clone()
};

//! Streams a clonable object's print() to \p os.
inline std::ostream& operator<<(std::ostream& os, const Clonable& cl) {
	return cl.print(os);
}

//! How a Clonable_Impl slot (clone/sharing_clone/stateless_clone) is implemented
enum CloneKind {
	CLONE_INHERIT,      //!< do not add behavior; defer to the Base class body (typically a family-level throwing default)
	CLONE_THROW,        //!< throw fst_error (for classes whose Base has no default body to defer to)
	CLONE_DEEP,         //!< new Derived(*this), then set_cloned()
	CLONE_WEAK,         //!< new Derived(*this,0) via the weak/referencing copy-constructor, then set_cloned()
	CLONE_WEAK_SHARING, //!< new Derived(*this,0) via the weak/referencing copy-constructor, then set_sharing_cloned()
	CLONE_TO_STATELESS  //!< forward to stateless_clone() (Classifier_LIBSVM contract)
};

/*! \brief CRTP mixin implementing the Clonable protocol for a concrete class

    Usage: class X : public Clonable_Impl<X<args...>, Family_Base<args...> > {...};
    Concrete classes never write clone()/sharing_clone()/stateless_clone() bodies of their own --
    this mixin implements all three uniformly, which is what keeps the clone/sharing flags
    consistent across the library. The three slot parameters select the mechanism per clone
    kind; the default is a deep clone via Derived's copy-constructor with the other two kinds
    deferred to the family base (whose defaults throw).

    Notes:
    \li Derived's copy-constructor remains responsible for correct member-wise
        deep-copy semantics (e.g. cloning wrapped Clonable members instead of sharing them).
    \li If Derived's copy-constructor is non-public, add 'friend parent;' (where parent
        is the class-local typedef of this mixin instantiation) to Derived.
    \li A class deriving from a Clonable_Impl-using concrete class must insert its own
        Clonable_Impl layer, otherwise clone() would slice; this is asserted at runtime
        in debug builds.
    \li set_cloned()/set_sharing_cloned() is applied uniformly, so is_clone()/is_sharing()
        answer correctly for every clone produced through this mixin.

    \tparam Derived the concrete class being equipped (the CRTP self type)
    \tparam Base the family's abstract base, which must itself derive from Clonable
    \tparam DEEP_KIND mechanism backing clone()
    \tparam SHARING_KIND mechanism backing sharing_clone()
    \tparam STATELESS_KIND mechanism backing stateless_clone() */
template<class Derived, class Base,
         CloneKind DEEP_KIND = CLONE_DEEP,
         CloneKind SHARING_KIND = CLONE_INHERIT,
         CloneKind STATELESS_KIND = CLONE_INHERIT>
class Clonable_Impl : public Base {
	static_assert(std::is_base_of<Clonable,Base>::value,
	              "Clonable_Impl: Base must derive from Clonable (family-base contract)");
public:
	Base* clone() const override {return make<DEEP_KIND,SLOT_CLONE>();} //!< dispatches to the DEEP_KIND mechanism (default CLONE_DEEP: Derived's copy-constructor)
	Base* sharing_clone() const override {return make<SHARING_KIND,SLOT_SHARING>();} //!< dispatches to the SHARING_KIND mechanism (default CLONE_INHERIT: defers to Base, which throws by family convention)
	Base* stateless_clone() const override {return make<STATELESS_KIND,SLOT_STATELESS>();} //!< dispatches to the STATELESS_KIND mechanism (default CLONE_INHERIT: defers to Base, which throws by family convention)
protected:
	using Base::Base; //!< inherits Base's constructors, so Derived need not repeat them
private:
	enum Slot {SLOT_CLONE, SLOT_SHARING, SLOT_STATELESS}; //!< which of the three clone() overrides is dispatching, so make()'s CLONE_INHERIT branch defers to the matching Base:: method

	//! Casts to Derived, asserting the dynamic type actually IS Derived (catches a more-derived
	//! class that forgot its own Clonable_Impl layer, which would otherwise slice on clone).
	const Derived& self() const {
		assert(typeid(*this)==typeid(Derived)); // a more-derived class is missing its own Clonable_Impl layer (would slice)
		return static_cast<const Derived&>(*this);
	}
	//! \brief Constructs the clone per CloneKind \p K, for the override slot \p S (see Slot).
	//! Only the branch selected by \p K is ever instantiated (`if constexpr`), so e.g. a class
	//! with no weak copy-constructor never instantiates the CLONE_WEAK path just because some
	//! OTHER class's Clonable_Impl instantiation elsewhere uses it.
	template<CloneKind K, Slot S> Base* make() const {
		static_assert(std::is_base_of<Clonable_Impl,Derived>::value,
		              "Clonable_Impl: Derived must derive from its own Clonable_Impl (CRTP contract)");
		if constexpr(K==CLONE_DEEP) {
			Derived *c=new Derived(self()); c->set_cloned(); return c;
		} else if constexpr(K==CLONE_WEAK) {
			Derived *c=new Derived(self(),0); c->set_cloned(); return c;
		} else if constexpr(K==CLONE_WEAK_SHARING) {
			Derived *c=new Derived(self(),0); c->set_sharing_cloned(); return c;
		} else if constexpr(K==CLONE_TO_STATELESS) {
			return this->stateless_clone();
		} else if constexpr(K==CLONE_THROW) {
			throw fst_error("Clonable_Impl: this clone kind is not supported by this class.");
		} else {
			// CLONE_INHERIT defers to the Base body, which by family convention throws; the
			// casts are unreachable at runtime and exist only to satisfy the return type
			if constexpr(S==SLOT_CLONE) return static_cast<Base*>(Base::clone());
			else if constexpr(S==SLOT_SHARING) return static_cast<Base*>(Base::sharing_clone());
			else return static_cast<Base*>(Base::stateless_clone());
		}
	}
};

/*! unique_ptr-returning clone helpers preserving the argument's static type.
    Replace the covariant raw-pointer idiom 'shared_ptr<T> p(x->clone())' with
    'p=clone_unique(*x)'; the static_cast is safe because the dynamic type of a
    clone is the dynamic type of its source, which is at least T. */
template<class T>
std::unique_ptr<T> clone_unique(const T& src) {
	return std::unique_ptr<T>(static_cast<T*>(src.clone()));
}
//! sharing_clone() counterpart of clone_unique() -- see its comment above.
template<class T>
std::unique_ptr<T> sharing_clone_unique(const T& src) {
	return std::unique_ptr<T>(static_cast<T*>(src.sharing_clone()));
}
//! stateless_clone() counterpart of clone_unique() -- see its comment above.
template<class T>
std::unique_ptr<T> stateless_clone_unique(const T& src) {
	return std::unique_ptr<T>(static_cast<T*>(src.stateless_clone()));
}

} // namespace
#endif // FSTCLONABLE_H ///:~
