#ifndef FSTEXTERNBRIDGE_H
#define FSTEXTERNBRIDGE_H

/*!======================================================================
   Feature Selection Toolbox 4 source code
   ---------------------------------------

   \file    extern_bridge.hpp
   \brief   Persistent-child-process line-protocol plumbing shared by the external-tool bridges (Classifier_Extern, Criterion_Extern_Value)
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

#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <sstream>
#include <mutex>
#include "error.hpp"
#include "global.hpp"

// ---- the ONLY platform-sensitive process plumbing in FST4, kept in this single header
//      so that both external-tool bridges share one implementation instead of duplicating
//      the fork/CreateProcess code ----
#ifdef _WIN32
	#ifndef NOMINMAX
		#define NOMINMAX // keep windows.h from defining min/max macros that break std::min/max and numeric_limits textually after this include
	#endif
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	#include <windows.h>
	#include <io.h>
	#include <fcntl.h>
#else
	#include <unistd.h>
	#include <sys/wait.h>
	#include <signal.h>
	#include <fcntl.h>
#endif

namespace FST {

//! process-wide spawn serialization for the extern bridges (shared across all
//! users and template instantiations; the function-local static avoids header ODR issues)
inline std::mutex& extern_spawn_mutex() {static std::mutex m; return m;}

/*! \brief Persistent-child-process line-protocol plumbing shared by the external-tool
    bridges: Classifier_Extern (TRAIN/TEST label protocol) and Criterion_Extern_Value
    (VALUE protocol).

    Owns the child process handles and the raw line I/O; the protocol blocks
    themselves (headers, data rows, response parsing) live in the owning component.
    Error messages carry the owning component's name, so a diagnostic always says
    which bridge failed.

    SINGLE OWNER BY DESIGN: exactly one instance per owning component instance, and
    the class is non-copyable. A copy would duplicate the FILE* pipe ends and the
    child handle without duplicating the child, so two objects would write into one
    child's stdin, interleave their protocol blocks, and each try to close and reap
    the same process in its destructor. Clones of the owners therefore construct
    FRESH bridges with their own child process - pipes are never shared - which is
    also what keeps threaded evaluation isolation-safe.

    All LOUD CAVEATS of the extern mechanism apply: the command is user-supplied code
    executed through the system shell, reproducibility guarantees end at the bridge
    (whatever the external tool does is outside FST4's control), there is no timeout
    machinery, and see the SIGPIPE note on ensure_process(). */
class Extern_Bridge {
public:
	explicit Extern_Bridge(const char *component) : _component(component) {init_process_handles();} //!< \p component names the owning component (e.g. "Classifier_Extern"), prefixing every diagnostic; does not spawn anything yet
	~Extern_Bridge() {shutdown_process();}
	Extern_Bridge(const Extern_Bridge&) = delete; // owners construct fresh bridges for their clones instead
	Extern_Bridge& operator=(const Extern_Bridge&) = delete;

	void set_command(const std::string &cmd) {assert(!running()); _cmd=cmd;} //!< sets the external command line; must be called before the child process is spawned (ensure_process())
	const std::string& get_command() const {return _cmd;} //!< returns the configured command line
	bool running() const {return _to_child!=0;} //!< true once ensure_process() has successfully spawned a child that has not since been shut down
	std::FILE* to_child() {assert(_to_child); return _to_child;} //!< for owner-side row streaming (%.17g formatting stays with the owner)
	void flush_to_child() {assert(_to_child); std::fflush(_to_child);} //!< flushes buffered writes to the child's stdin (call after a batch of send_line()/owner-side writes, before reading a response)
	void die_on_write() const {throw fst_error(_component+": the external command '"+_cmd+"' stopped accepting input (died?)");} //!< shared loud-failure path for owner-side writes

	/*! \brief Lazily spawns the child process if not already running() (a no-op otherwise).
	    Requires get_command() to be non-empty. Spawns are serialized process-wide via
	    extern_spawn_mutex() -- without this, a concurrently spawning clone's child (threaded
	    search = one child per criterion/classifier clone) could inherit this clone's pipe ends
	    in the window before they are marked non-inheritable, which would keep a sibling
	    child's stdin open past this instance's shutdown_process() and deadlock the destructor's
	    wait. Prints a one-time process-wide NOTE that the command executes user-supplied
	    code through the shell. Platform-specific pipe setup follows: CreateProcess via
	    "cmd.exe /C <cmd>" on Windows, fork()+"/bin/sh -c <cmd>" on POSIX.
	    \warning SIGPIPE side effect on POSIX. A child that stops reading must surface as an
	    fst_error rather than killing fst4, so writes to the pipe must not raise SIGPIPE. On
	    macOS/BSD this is arranged per file descriptor (F_SETNOSIGPIPE) and nothing else is
	    affected. On platforms without that option - Linux among them - the first spawn calls
	    signal(SIGPIPE,SIG_IGN), which changes the disposition for the WHOLE PROCESS and is
	    never restored. Embedders whose own code relies on SIGPIPE terminating the process (or
	    on their own handler) must take that into account before using an external bridge.
	    \throw fst_error on any setup failure (pipe/fork/CreateProcess, or a launch failure). */
	void ensure_process()
	{
		if(running()) return;
		if(_cmd.empty()) throw fst_error(_component+": no command set (see --extern-cmd)");
		// serialize spawns process-wide: without this, a concurrently spawning clone's child
		// (threaded search = one child per criterion clone) can inherit this clone's pipe ends
		// in the window before they are marked non-inheritable below - an inherited write end
		// keeps a sibling child's stdin open past shutdown_process()'s fclose, the child never
		// sees EOF, and the destructor's waitpid deadlocks
		std::unique_lock<std::mutex> spawn_lock(extern_spawn_mutex());
		{ // one-time runtime notice: someone running an untrusted third-party config
		  // should see that it executes code; the CLI's prose plumbing swallows this under
		  // `--verbosity` silent and routes it to stderr under machine output formats
			static bool notice_fired=false; // guarded by the spawn lock
			if(!notice_fired) {
				notice_fired=true;
				std::ostringstream sos; sos << "NOTE: --extern-cmd executes user-supplied code through the system shell: " << _cmd << std::endl;
				syncout::print(std::cout,sos);
			}
		}
#ifdef _WIN32
		// ---- Windows plumbing (CreateProcess through cmd.exe) ----
		SECURITY_ATTRIBUTES sa; sa.nLength=sizeof(sa); sa.bInheritHandle=TRUE; sa.lpSecurityDescriptor=NULL;
		HANDLE in_rd=0,in_wr=0,out_rd=0,out_wr=0;
		if(!CreatePipe(&in_rd,&in_wr,&sa,0) || !CreatePipe(&out_rd,&out_wr,&sa,0))
			throw fst_error(_component+": CreatePipe failed");
		SetHandleInformation(in_wr,HANDLE_FLAG_INHERIT,0);  // our ends must not leak to the child
		SetHandleInformation(out_rd,HANDLE_FLAG_INHERIT,0);
		STARTUPINFOA si; ZeroMemory(&si,sizeof(si)); si.cb=sizeof(si);
		si.dwFlags=STARTF_USESTDHANDLES;
		si.hStdInput=in_rd; si.hStdOutput=out_wr; si.hStdError=GetStdHandle(STD_ERROR_HANDLE);
		PROCESS_INFORMATION pi; ZeroMemory(&pi,sizeof(pi));
		std::string full=std::string("cmd.exe /C ")+_cmd;
		std::vector<char> cmdbuf(full.begin(),full.end()); cmdbuf.push_back('\0');
		if(!CreateProcessA(NULL,cmdbuf.data(),NULL,NULL,TRUE,0,NULL,NULL,&si,&pi))
			{CloseHandle(in_rd);CloseHandle(in_wr);CloseHandle(out_rd);CloseHandle(out_wr);
			 throw fst_error(_component+": cannot launch '"+_cmd+"'");}
		CloseHandle(pi.hThread);
		CloseHandle(in_rd); CloseHandle(out_wr); // child's ends
		_child_process=pi.hProcess;
		_to_child=_fdopen(_open_osfhandle((intptr_t)in_wr,_O_WRONLY|_O_BINARY),"wb");
		_from_child=_fdopen(_open_osfhandle((intptr_t)out_rd,_O_RDONLY|_O_BINARY),"rb");
		if(!_to_child || !_from_child) {shutdown_process(); throw fst_error(_component+": cannot attach pipes to '"+_cmd+"'");}
#else
		// ---- POSIX plumbing (fork + /bin/sh -c) ----
		int to_pipe[2], from_pipe[2];
		if(pipe(to_pipe)!=0 || pipe(from_pipe)!=0) throw fst_error(_component+": pipe() failed");
		fcntl(to_pipe[1],F_SETFD,FD_CLOEXEC);   // our ends must not leak into later children
		fcntl(from_pipe[0],F_SETFD,FD_CLOEXEC); // (the Windows twin: SetHandleInformation above)
		// SIGPIPE discipline: a child that stops reading (crashed mid-search) must surface as
		// the send-path fst_errors, not kill fst4 silently (default SIGPIPE disposition
		// terminates the process before fputs/fputc can even return EOF - observed as exit 141
		// with zero output, intermittently masked by kernel pipe buffering on small payloads)
#ifdef F_SETNOSIGPIPE
		fcntl(to_pipe[1],F_SETNOSIGPIPE,1); // macOS/BSD: per-fd, write fails with EPIPE instead of raising SIGPIPE
#else
		// no per-fd control on this platform, so SIGPIPE is ignored PROCESS-WIDE and stays
		// ignored for the rest of the run - a side effect on the whole embedding program,
		// documented as a warning on ensure_process() above
		signal(SIGPIPE,SIG_IGN);
#endif
		const pid_t pid=fork();
		if(pid<0) throw fst_error(_component+": fork() failed");
		if(pid==0) { // child: wire the pipes to stdin/stdout and exec the command via the shell
			dup2(to_pipe[0],0); dup2(from_pipe[1],1);
			close(to_pipe[0]); close(to_pipe[1]); close(from_pipe[0]); close(from_pipe[1]);
			execl("/bin/sh","sh","-c",_cmd.c_str(),(char*)0);
			_exit(127); // exec failed
		}
		close(to_pipe[0]); close(from_pipe[1]);
		_child_pid=pid;
		_to_child=fdopen(to_pipe[1],"w");
		_from_child=fdopen(from_pipe[0],"r");
		if(!_to_child || !_from_child) {shutdown_process(); throw fst_error(_component+": cannot attach pipes to '"+_cmd+"'");}
#endif
	}

	/*! \brief Terminates the child, if any, and returns the bridge to a not-running() state.
	    Closing our end of stdin/stdout (fclose) signals EOF to the child, the conventional
	    terminate request; the call then WAITS for the child to actually exit (waitpid() on
	    POSIX -- unbounded, so a child that ignores stdin-EOF hangs this call and thus the
	    destructor; WaitForSingleObject with a 2s timeout followed by a forced TerminateProcess
	    on Windows). Idempotent: every handle is individually null-checked, so repeated calls
	    (e.g. destructor after an explicit error-path shutdown) are safe. Called from the
	    destructor and from every "the child misbehaved" error path (a desynced or dead child
	    is dropped rather than reused; the next ensure_process() respawns cleanly). */
	void shutdown_process()
	{
		if(_to_child) {std::fclose(_to_child); _to_child=0;} // EOF on the child's stdin = terminate request
		if(_from_child) {std::fclose(_from_child); _from_child=0;}
#ifdef _WIN32
		if(_child_process) {WaitForSingleObject(_child_process,2000); TerminateProcess(_child_process,0); CloseHandle(_child_process); _child_process=0;}
#else
		if(_child_pid>0) {int status=0; waitpid(_child_pid,&status,0); _child_pid=-1;}
#endif
	}

	//! \brief Writes \p line followed by a newline to the child's stdin. Requires running().
	//! Throws (via die_on_write()) if either write fails, i.e. the child stopped reading.
	//! Does NOT flush -- callers that need the data to arrive before reading a response must
	//! call flush_to_child() themselves (e.g. after a batch of send_line()/row writes).
	void send_line(const std::string &line)
	{
		assert(_to_child);
		if(std::fputs(line.c_str(),_to_child)==EOF || std::fputc('\n',_to_child)==EOF)
			die_on_write();
	}

	/*! \brief Reads one line from the child's stdout, with the trailing "\n" (and a preceding
	    "\r", for CRLF-emitting children) stripped. Requires running(). Throws fst_error if the
	    child produced no response at all (EOF/died mid-read) or if more than 1 MB accumulates
	    without a newline -- a runaway/non-protocol child must fail loudly rather than exhaust
	    memory; that case also calls shutdown_process() first so the dead/misbehaving child is
	    not reused. */
	std::string read_line()
	{
		assert(_from_child);
		std::string out;
		char buf[256];
		for(;;)
		{
			if(!std::fgets(buf,sizeof(buf),_from_child))
				throw fst_error(_component+": the external command '"+_cmd+"' produced no response (died?)");
			out+=buf;
			if(!out.empty() && out.back()=='\n') {out.pop_back(); if(!out.empty() && out.back()=='\r') out.pop_back(); return out;}
			if(out.size()>1048576) // a child streaming bytes without newline must fail loudly, not exhaust memory
				{shutdown_process(); throw fst_error(_component+": the external command '"+_cmd+"' produced a response line exceeding 1 MB (runaway output?)");}
		}
	}

private:
	std::string _component; //!< owning component name, prefixes every diagnostic
	std::string _cmd;       //!< the configured external command line (see set_command())
	std::FILE *_to_child;   //!< write end of the pipe to the child's stdin, or nullptr when not running()
	std::FILE *_from_child; //!< read end of the pipe from the child's stdout, or nullptr when not running()
#ifdef _WIN32
	HANDLE _child_process;
#else
	pid_t _child_pid;       //!< POSIX pid of the running child, or -1 when not running()
#endif
	//! \brief Resets the handles/pid to their "not running" sentinel values (nullptr / -1).
	//! Called from the constructor to establish that initial state; NOT called by
	//! shutdown_process() itself (which null/reset each handle individually as it closes it).
	void init_process_handles()
	{
		_to_child=0; _from_child=0;
#ifdef _WIN32
		_child_process=0;
#else
		_child_pid=-1;
#endif
	}
};

} // namespace
#endif // FSTEXTERNBRIDGE_H ///:~
