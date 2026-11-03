"""N-gram baselines under CellAI's arch_eval_v1 reset protocol.
Replicates: first 6000 records/domain (pipeline record format), seed-42 perm (v4..v7),
85/15 split, eval = first 50 docs/modality of eval split, first 256 tokens,
per-doc mean NLL (nats), then mean over docs."""
import json, math, sys, os, collections, statistics, random
import pyarrow.parquet as pq, tiktoken
SP = os.environ.get('AUDIT_WORK', 'audit_work')
enc = tiktoken.get_encoding("cl100k_base"); V = enc.n_vocab
def text_records(n):
    out=[]; pf=pq.ParquetFile(f"{SP}/dl/wt0.parquet")
    for b in pf.iter_batches(batch_size=50000, columns=["text"]):
        for t in b.column(0).to_pylist():
            t=t.strip()
            if len(t)>=50: out.append(t)
            if len(out)>=n: return out
def code_records(n, skip=0):
    out=[]; pf=pq.ParquetFile(f"{SP}/dl/csn0.parquet"); seen=0
    for b in pf.iter_batches(batch_size=20000, columns=["func_code_string","func_documentation_string"]):
        for c,d in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            c=(c or "").strip(); d=(d or "").strip()
            if len(c)<30: continue
            seen+=1
            if seen<=skip: continue
            t=((f'"""{d}"""\n' if d else "")+c)[:8000]
            if len(t)>=40: out.append(t)
            if len(out)>=n: return out
N_EXTRA = int(sys.argv[1]) if len(sys.argv)>1 else 0
texts = text_records(6000+N_EXTRA); codes = code_records(6000+N_EXTRA)
recs = {"nlp":texts[:6000], "code":codes[:6000]}
PERMF = sys.argv[2] if len(sys.argv)>2 else "leak/n_run_arch_search_v4.json"
perm = json.load(open(f"{SP}/"+PERMF))["perm1"]
order=[]
for key in perm:
    dom, idx = key.split("-")[0], int(key.split("-")[1][:5])
    order.append((dom, idx))
cut=int(len(order)*0.85)
train=[(d,i) for d,i in order[:cut] if d!="math"]; ev=[(d,i) for d,i in order[cut:] if d!="math"]
def toks(d,i, cap=None):
    t=enc.encode(recs[d][i], disallowed_special=())
    return t[:cap] if cap else t
train_docs=[toks(d,i) for d,i in train]
# Extra data (beyond the 6000 cap) for an 'unconstrained data' reference
extra_docs=[enc.encode(t, disallowed_special=()) for t in texts[6000:]+codes[6000:]] if N_EXTRA else []
def fit(docs):
    uni=collections.Counter(); bi=collections.Counter(); tri=collections.Counter()
    for t in docs:
        uni.update(t); bi.update(zip(t,t[1:])); tri.update(zip(t,t[1:],t[2:]))
    # KN statistics
    cont_uni=collections.Counter(w for (_,w) in bi)            # N1+(. w)
    n_bi_types=len(bi)
    ctx_bi=collections.Counter(); ctx_bi_types=collections.Counter()
    for (a,b),c in bi.items(): ctx_bi[a]+=c; ctx_bi_types[a]+=1
    ctx_tri=collections.Counter(); ctx_tri_types=collections.Counter()
    for (a,b,c),n in tri.items(): ctx_tri[(a,b)]+=n; ctx_tri_types[(a,b)]+=1
    # continuation counts for middle order: N1+(. a b)
    cont_bi=collections.Counter((b,c) for (a,b,c) in tri)
    cont_bi_ctx=collections.Counter(); cont_bi_ctx_types=collections.Counter()
    for (b,c),n in cont_bi.items(): cont_bi_ctx[b]+=n; cont_bi_ctx_types[b]+=1
    return dict(uni=uni,N=sum(uni.values()),bi=bi,tri=tri,cont_uni=cont_uni,nbt=n_bi_types,
                ctx_bi=ctx_bi,ctx_bi_types=ctx_bi_types,ctx_tri=ctx_tri,ctx_tri_types=ctx_tri_types,
                cont_bi=cont_bi,cont_bi_ctx=cont_bi_ctx,cont_bi_ctx_types=cont_bi_ctx_types)
D=0.75
def p_uni(M,w):  # add-0.5 smoothed unigram
    return (M["uni"][w]+0.5)/(M["N"]+0.5*V)
def p_kn_uni(M,w):  # KN lowest order, interpolated with uniform
    return (M["cont_uni"][w]+ 1.0/V*1.0) / (M["nbt"]+1.0)
def p_bi(M,a,w, lower=None):
    lower = p_kn_uni(M,w) if lower is None else lower
    c=M["ctx_bi"][a]
    if c==0: return lower
    return max(M["bi"][(a,w)]-D,0)/c + D*M["ctx_bi_types"][a]/c*lower
def p_bi_mid(M,a,w):  # continuation-count bigram used inside trigram KN
    c=M["cont_bi_ctx"][a]
    if c==0: return p_kn_uni(M,w)
    return max(M["cont_bi"][(a,w)]-D,0)/c + D*M["cont_bi_ctx_types"][a]/c*p_kn_uni(M,w)
def p_tri(M,a,b,w):
    lower=p_bi_mid(M,b,w); c=M["ctx_tri"][(a,b)]
    if c==0: return lower
    return max(M["tri"][(a,b,w)]-D,0)/c + D*M["ctx_tri_types"][(a,b)]/c*lower
def doc_nll(M,t,kind):
    s=0;n=0
    for i in range(len(t)-1):
        w=t[i+1]
        if kind=="unigram": p=p_uni(M,w)
        elif kind=="bigram_kn": p=p_bi(M,t[i],w)
        else: p=p_tri(M,t[i-1],t[i],w) if i>=1 else p_bi(M,t[i],w)
        s+=-math.log(p); n+=1
    return s/n
def evaluate(M, tag):
    res={}
    for dom in ["nlp","code"]:
        all_docs=[toks(d,i,256) for d,i in ev if d==dom]
        all_docs=[t for t in all_docs if len(t)>=2]
        for kind in ["unigram","bigram_kn","trigram_kn"]:
            vals=[doc_nll(M,t,kind) for t in all_docs]
            first50=vals[:50]
            # bootstrap spread of a 50-doc subsample
            rng=random.Random(0); boots=[statistics.mean(rng.sample(vals,50)) for _ in range(500)]
            res[(dom,kind)]=dict(n50=statistics.mean(first50), all=statistics.mean(vals), n_all=len(vals),
                                 sd50=statistics.pstdev(boots))
    out={}
    for kind in ["unigram","bigram_kn","trigram_kn"]:
        t=res[("nlp",kind)]; c=res[("code",kind)]
        out[kind]=dict(text_nll_n50=t["n50"], code_nll_n50=c["n50"], textcode_n50=(t["n50"]+c["n50"])/2,
                       text_nll_all=t["all"], code_nll_all=c["all"], textcode_all=(t["all"]+c["all"])/2,
                       n_eval_text=t["n_all"], n_eval_code=c["n_all"],
                       sd_textcode_50doc_subsample=math.sqrt(t["sd50"]**2+c["sd50"]**2)/2)
    print(tag, json.dumps(out, indent=1)); return out
results={}
M=fit(train_docs); results["same_data_as_cellai"]=evaluate(M,"same-data")
results["train_tokens"]=M["N"]
if extra_docs:
    M2=fit(train_docs+extra_docs); results["more_data"]=evaluate(M2,f"more-data(+{N_EXTRA}/domain)")
    results["more_data_train_tokens"]=M2["N"]
json.dump(results, open(f"{SP}/baselines/ngram_results_"+PERMF.split("/")[-1],"w"), indent=1)
