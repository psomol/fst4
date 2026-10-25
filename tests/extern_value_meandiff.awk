# extern_value_meandiff.awk — reference external VALUE-mode criterion for
# FST4's extern-value bridge (criterion_extern_value.hpp): the sum over
# selected features of |mean(f|class 0) - mean(f|class 1)|. The sum of
# nonnegative per-feature terms cannot increase when a feature is removed, so
# the measure IS monotonic w.r.t. subset cardinality — the reference case for
# --extern-monotonic (Branch & Bound admissible). Two-class reference: classes
# beyond index 1 are ignored.
#
# Protocol (one persistent process per criterion clone; DATA once per split,
# then index-only requests — the SUFFICIENT-STATISTICS pattern: per-feature
# statistics are computed once at DATA time, every VALUE answers in O(d)):
#   -> DATA <N> <D> <K>           then N lines: <label> <v1> ... <vD>
#   <- OK
#   -> VALUE <d> <f1> ... <fd>    (0-based raw feature indices)
#   <- one line: the criterion value
#   VALUE requests repeat for every candidate subset; DATA re-arrives when the
#   split changes; EOF ends.
#
# Usage example (from the repo root):
#   ./fst4 --data data/wdbc_30.trn --criterion extern-value --extern-monotonic \
#          --extern-cmd "awk -f tests/extern_value_meandiff.awk" --search bb-improved --target-d 10
BEGIN { mode=""; }
{
	if(mode=="") {
		if($1=="DATA") {
			N=$2; D=$3; K=$4; row=0;
			for(c=0;c<2;c++) {cnt[c]=0; for(f=1;f<=D;f++) sum[c,f]=0;}
			mode="data";
		} else if($1=="VALUE") {
			d=$2; v=0;
			for(i=1;i<=d;i++) v+=absdiff[$(i+2)+1]; # indices are 0-based raw
			printf "%.17g\n", v; fflush();
		} else { print "ERROR unexpected input: " $0; fflush(); }
		next;
	}
	if(mode=="data") {
		lab=$1;
		if(lab==0 || lab==1) {
			cnt[lab]++;
			for(f=1;f<=D;f++) sum[lab,f]+=$(f+1);
		}
		row++;
		if(row>=N) {
			for(f=1;f<=D;f++) {
				m0=(cnt[0]>0 ? sum[0,f]/cnt[0] : 0);
				m1=(cnt[1]>0 ? sum[1,f]/cnt[1] : 0);
				dv=m0-m1; if(dv<0) dv=-dv;
				absdiff[f]=dv; # the per-feature sufficient statistic
			}
			print "OK"; fflush();
			mode="";
		}
		next;
	}
}
