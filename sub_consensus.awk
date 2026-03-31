# sub_consensus.awk (patched for speed)

BEGIN {
    RS = "\r?\n"       # split records on \n with optional preceding \r (faster than per-line gsub)
    FS = OFS = ","

    if (ARGC < 4) {
        print "Usage: awk -f sub_consensus.awk <input.csv> <seqOut.csv> <subOut.csv>" > "/dev/stderr"
        exit 2
    }

    prev_umi = ""
    prev_kmer = ""

    seqOutFile = ARGV[ARGC-2]
    subOutFile = ARGV[ARGC-1]
    ARGV[ARGC-2] = ""   # hide from input stream (portable)
    ARGV[ARGC-1] = ""

    useFlush = 0
    init_arrays()
}

function init_arrays() {
    # Fast, portable array clears
    split("", A); split("", C); split("", G); split("", T)
    maxlen = 0
    myreads = 0
}

function flush_out(f){
    if (useFlush) close(f)    # close() forces a flush portably
}

function emit_prev_umi(    u, i, delim) {
    if (prev_umi == "") return
    u = prev_umi

    printf("%s,%d", u, (u in readSum ? readSum[u] : 0)) >> seqOutFile

    for (i = 1; i <= subCount[u]; i++) {
        delim = (i == 1 ? "," : ";")
        printf("%s%s", delim, parts[u, i]) >> seqOutFile
        delete parts[u, i]
    }
    printf("\n") >> seqOutFile
    flush_out(seqOutFile)

    delete subCount[u]
    delete readSum[u]
}

function flush_cluster(    i, consensus_fragment, total, base, ai, ci, gi, ti) {
    if (prev_umi == "") return

    if (myreads > 0) {
        consensus_fragment = ""
        for (i = 1; i <= maxlen; i++) {
            ai = A[i]+0; ci = C[i]+0; gi = G[i]+0; ti = T[i]+0
            total = ai + ci + gi + ti
            if      (total == 0)                                  base = "N"
            else if (ai > ci && ai > gi && ai > ti)               base = "A"
            else if (ci > ai && ci > gi && ci > ti)               base = "C"
            else if (gi > ai && gi > ci && gi > ti)               base = "G"
            else if (ti > ai && ti > ci && ti > gi)               base = "T"
            else                                                  base = "N"
            consensus_fragment = consensus_fragment base
        }

        subCount[prev_umi]++
        parts[prev_umi, subCount[prev_umi]] = consensus_fragment ":" myreads
        readSum[prev_umi] += myreads+0

        # Format the padded read count directly in printf
        printf("%s.%07d.sub%d,%s\n",
               prev_umi, myreads, subCount[prev_umi]-1, consensus_fragment) >> subOutFile
        flush_out(subOutFile)
    }

    init_arrays()
}

{
    if (NF < 4 || $4 == "") next

    current_umi  = $2
    current_kmer = $3
    current_seq  = $4

    if ((current_umi != prev_umi) || (current_kmer != prev_kmer)) {
        flush_cluster()
        if (prev_umi != "" && (current_umi != prev_umi)) emit_prev_umi()
        prev_umi  = current_umi
        prev_kmer = current_kmer
    }

    len = length(current_seq); if (len > maxlen) maxlen = len
    myreads++

    # Split once, then index characters (faster than substr in a loop)
    nchars = split(current_seq, ch, "")
    for (i = 1; i <= nchars; i++) {
        base = ch[i]
        if      (base == "A") ++A[i]
        else if (base == "C") ++C[i]
        else if (base == "G") ++G[i]
        else if (base == "T") ++T[i]
    }
}

END {
    flush_cluster()
    emit_prev_umi()
    flush_out(seqOutFile)
    flush_out(subOutFile)
}
