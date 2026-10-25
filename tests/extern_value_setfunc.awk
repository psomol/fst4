# extern_value_setfunc.awk — reference DATA-FREE set function for FST4's
# extern-value bridge with --extern-no-data (criterion_extern_value.hpp):
# the tool never receives data — it answers index-only VALUE
# requests from its own logic, here a synthetic additive set function
#     value(S) = sum over f in S of w(f),   w(f) = ((f*37) mod 19) + 1
# The weights are positive, so the sum is monotonic w.r.t. subset
# cardinality by construction — the reference case for pure set-function
# optimization with --extern-monotonic (Branch & Bound admissible; the
# optimum is provably the d largest weights).
#
# Protocol (one persistent process per criterion clone; no DATA ever):
#   -> VALUE <d> <f1> ... <fd>    (0-based raw feature indices)
#   <- one line: the set-function value
#   VALUE requests repeat for every candidate subset; EOF ends.
#
# Usage example (from the repo root; --features N declares the abstract universe
# with no data file at all, --data FILE would merely define its size):
#   ./fst4 --features 30 --criterion extern-value --extern-no-data \
#          --extern-monotonic --extern-cmd "awk -f tests/extern_value_setfunc.awk" \
#          --search bb-improved --target-d 5
{
	if($1=="VALUE") {
		d=$2; v=0;
		for(i=1;i<=d;i++) {f=$(i+2); v+=((f*37)%19)+1;}
		printf "%d\n", v; fflush();
	} else { print "ERROR unexpected input: " $0; fflush(); }
}
