#ifndef FSTTHREADPOOL_H
#define FSTTHREADPOOL_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------
	
   \file    thread_pool.hpp
   \brief   Implements thread scheduler that assigns jobs up to maximum number of threads allowed
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

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include "global.hpp"

namespace FST {

/*! \brief Implements thread scheduler that assigns jobs up to maximum number of threads allowed

    The capacity is a runtime constructor argument, so the same binary can be driven with any
    thread count. The pool is deliberately minimal: go() launches a job and blocks the caller
    while the pool is at capacity, and the only synchronization primitive offered is join_all(),
    which waits for every in-flight job. There is no per-job handle, no future and no result
    channel -- a job communicates by writing into structures the caller owns and outlives it.

    \warning An exception escaping a job function propagates out of the worker thread and
    terminates the process (std::terminate). A job must catch everything it can throw. */
class ThreadPool
{
public:
	/*! \brief Constructs a pool that allows up to \p max_threads concurrently running worker threads.
	    \param max_threads pool capacity; must be greater than zero (asserted) */
	explicit ThreadPool(const unsigned int max_threads=2) : _max_threads(max_threads), threads(max_threads), _running_threads(0)
	{
		assert(max_threads>0);
		for(unsigned int i=0;i<_max_threads;i++) threads[i].tid=-1;
	}
	//! Joins every in-flight job before destroying the pool.
	~ThreadPool() {notify("ThreadPool destructor, idx=",tid_counter().load()); join_all();}

	unsigned int get_max_threads() const {return _max_threads;} //!< returns the pool capacity (\p max_threads as given to the constructor)

	/*! \brief Launches \p threadfunc in a free pool slot, blocking the caller until one becomes available if the pool is currently at capacity.
	    Any previous thread that occupied the chosen slot is joined first (a slot's std::thread handle
	    is reused only after its predecessor has finished and been joined).
	    \param threadfunc the job to run on the worker thread
	    \return the 0-based slot index the job was launched into. This is a diagnostic value,
	    not a job handle: the slot is recycled as soon as the job ends and there is no way to
	    wait on that one job. Use join_all() to synchronize with all outstanding jobs. */
	unsigned int go(const std::function<void()> &threadfunc)
	{
		std::unique_lock<std::mutex> lock_launcher(mutex_launcher);
		if(_running_threads==_max_threads)
		{
			notify("Max allowed threads running, waiting for one to end...");
			while (_running_threads==_max_threads) cond_launcher.wait(lock_launcher);
		}
		unsigned int free_tidx=0;
		while(free_tidx<_max_threads && threads[free_tidx].tid!=-1) free_tidx++;
		assert(free_tidx<_max_threads);
		{
			notify("Launching thread, tid=",tid_counter().load(),"...");
			{
				_running_threads++;
			}
			if(threads[free_tidx].m_thread)
			{
				notify("Joining past ",free_tidx,"th thread before lanching new one.");
				threads[free_tidx].m_thread->join();
			}
			{
				threads[free_tidx].tid=tid_counter()++;
				threads[free_tidx].m_thread.reset(new std::thread(std::bind(&ThreadPool::launch_thread,this,threadfunc,free_tidx)));
				notify("Thread tid=",threads[free_tidx].tid," (",free_tidx,"th) launched...");
			}
		}
		return free_tidx;
	}

	//! \brief Joins every currently running worker thread and resets the pool to empty (all slots free).
	//! Safe to call from the destructor or explicitly mid-lifetime to synchronize with in-flight jobs.
	void join_all()
	{
		notify("Calling join_all...");
		for(unsigned int i=0; i<_max_threads; i++)
		{
			// take ownership of the thread handle under the lock, but join outside it -
			// the finishing worker needs mutex_launcher in launch_thread()
			ThreadPtr t;
			{
				std::unique_lock<std::mutex> lock(mutex_launcher);
				assert((threads[i].m_thread && threads[i].tid>=0) || (threads[i].tid==-1));
				t.swap(threads[i].m_thread);
			}
			if(t)
			{
				notify("join_all: Joining ",i,"th thread.");
				t->join();
				std::unique_lock<std::mutex> lock(mutex_launcher);
				threads[i].tid=-1;
			}
		}
		{
			std::unique_lock<std::mutex> lock(mutex_launcher);
			_running_threads=0;
		}
		notify("join_all finished...");
	}
	
	//! \brief Returns the number of pool slots with an active thread right now.
	//! The counter is read without taking the pool mutex, so the value is a snapshot that may
	//! already be stale when it is returned; use it for reporting, never to make a decision.
	unsigned int get_running_threads() const {return _running_threads;}

private:
	// the next procedure must be re-entrant
	/*! \brief Worker-thread trampoline: runs \p threadfunc to completion, then marks slot \p threadsid free and wakes one waiter in go().
	    \param threadfunc the job to execute
	    \param threadsid the pool slot this worker occupies (re-entrant across concurrently running workers) */
	void launch_thread(const std::function<void()> &threadfunc, unsigned int threadsid)
	{
		notify("Starting launch_thread(tid=",threads[threadsid].tid,", ",threadsid,"th) ...");
		threadfunc();
		notify("Finishing launch_thread(tid=",threads[threadsid].tid,", ",threadsid,"th).");
		{
			std::unique_lock<std::mutex> lock(mutex_launcher);
			_running_threads--;
			threads[threadsid].tid=-1;
		}
		{
			cond_launcher.notify_one();
		}
	}
private:
	typedef std::unique_ptr<std::thread> ThreadPtr; //!< owning handle to a pool worker's std::thread
	//! Structure to keep status of threads in the pool
	typedef struct {
		ThreadPtr m_thread; //!< owns the worker thread's std::thread handle, or null if this slot is free
		int tid;            //!< monotonically increasing id assigned when a thread is launched into this slot, -1 if the slot is free
	} ThreadsArray;
	const unsigned int _max_threads; //!< pool capacity, fixed at construction
	std::vector<ThreadsArray> threads; //!< one slot per allowed concurrent thread; sized once in the constructor, never resized (no reallocation races)

private:
	//! process-wide tid sequence for notify() diagnostics (function-local static avoids header ODR issues)
	static std::atomic<unsigned int>& tid_counter() {static std::atomic<unsigned int> c(0); return c;}
	unsigned int _running_threads; //!< current count of slots with an active thread (guarded by mutex_launcher)

	std::mutex mutex_launcher; //!< guards threads[], _running_threads and slot bookkeeping in go()/join_all()/launch_thread()
	std::condition_variable cond_launcher; //!< signaled by launch_thread() when a slot frees up, to wake a go() call waiting for capacity
};

} // namespace
#endif // FSTTHREADPOOL_H ///:~
