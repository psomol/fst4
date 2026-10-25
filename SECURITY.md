# Security policy

## Supported versions

Security fixes go into the latest release (currently FST4 4.2.0) and the
`main` branch. The FST3 and FST1 releases are not maintained.

## What to know before running FST4 on input you did not create

FST4 is a research tool. It runs locally, under the privileges of the user
who starts it, and makes no network connections. Three points deserve
attention:

- **`--extern-cmd` executes a command.** With `--criterion wrapper-extern`
  or `--criterion extern-value`, the string given to `--extern-cmd` is
  launched through the system shell, by design: it is the user's own
  classifier or criterion program. Never pass a command string that comes
  from an untrusted source. A config file (`--config`) can set this option
  too, so running a config file you did not write means executing the
  commands it names: read config files before running them.
- **The data readers parse untrusted files.** The TRN, ARFF and CSV readers
  are hand-written C/C++ parsers. They validate structure and reject
  malformed input, but like every parser they are an attack surface. Treat
  data files from unknown sources with the usual caution on machines where
  that matters.
- **Output goes only where you point it.** FST4 writes files only on request
  (`--dump-scores`, `--curves-svg`), plus a temporary file in the system
  temporary directory when the dataset arrives on standard input
  (`--data -`).

## Reporting a vulnerability

Please do not open a public issue for a security problem. Write to
**somol@utia.cas.cz** with "FST" and "security" in the subject line, or use GitHub's private vulnerability reporting on the
repository if it is enabled. Describe the problem and how to reproduce it.
You will receive an acknowledgement; a confirmed problem is fixed in the next
release and credited to you if you wish.
