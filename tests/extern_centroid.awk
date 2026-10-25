# extern_centroid.awk — reference external classifier for FST4's wrapper-extern
# bridge: a deterministic nearest-centroid
# classifier speaking the bridge's line protocol on stdin/stdout.
#
# Protocol (one persistent process per criterion clone):
#   -> TRAIN <N> <d> <K>          then N lines: <label> <v1> ... <vd>
#   <- OK                          (or "ERROR <message>")
#   -> TEST <M> <d>                then M lines: <v1> ... <vd>
#   <- M lines: <predicted label>
#   TRAIN/TEST blocks repeat for every candidate subset evaluation; EOF ends.
#
# Usage example (from the repo root):
#   ./fst4 --data data/wdbc_30.trn --criterion wrapper-extern \
#          --extern-cmd "awk -f tests/extern_centroid.awk" ...
BEGIN { mode=""; }
{
	if(mode=="") {
		if($1=="TRAIN") {
			N=$2; d=$3; K=$4; row=0;
			for(c=0;c<K;c++) {cnt[c]=0; for(f=1;f<=d;f++) sum[c,f]=0;}
			mode="train";
		} else if($1=="TEST") {
			M=$2; trow=0;
			mode="test";
		} else { print "ERROR unexpected input: " $0; fflush(); }
		next;
	}
	if(mode=="train") {
		lab=$1; cnt[lab]++;
		for(f=1;f<=d;f++) sum[lab,f]+=$(f+1);
		row++;
		if(row>=N) {
			for(c=0;c<K;c++) if(cnt[c]>0) for(f=1;f<=d;f++) sum[c,f]/=cnt[c];
			print "OK"; fflush();
			mode="";
		}
		next;
	}
	if(mode=="test") {
		best=-1; bestdist=0;
		for(c=0;c<K;c++) {
			if(cnt[c]==0) continue;
			dist=0;
			for(f=1;f<=d;f++) {dv=$(f)-sum[c,f]; dist+=dv*dv;}
			if(best<0 || dist<bestdist) {bestdist=dist; best=c;} # strict: ties keep the lowest class index
		}
		print best; fflush();
		trow++;
		if(trow>=M) mode="";
		next;
	}
}
