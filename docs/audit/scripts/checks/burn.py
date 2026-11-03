"""Fraction of scored documents that fall inside their own 4096-token warm-eval burn-in (section 9.2). Usage: AUDIT_WORK=... python burn.py [math_len]; needs dl/*.parquet and perm JSONs under AUDIT_WORK/leak/."""
import json, sys
import pyarrow.parquet as pq, tiktoken
import os; W=os.environ.get("AUDIT_WORK", "audit_work")
enc = tiktoken.get_encoding("cl100k_base")
def text_records(n):
    out=[]; pf=pq.ParquetFile(f"{W}/dl/wt0.parquet")
    for b in pf.iter_batches(batch_size=50000, columns=["text"]):
        for t in b.column(0).to_pylist():
            t=t.strip()
            if len(t)>=50: out.append(t)
            if len(out)>=n: return out
def code_records(n):
    out=[]; pf=pq.ParquetFile(f"{W}/dl/csn0.parquet")
    for b in pf.iter_batches(batch_size=20000, columns=["func_code_string","func_documentation_string"]):
        for c,d in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            c=(c or "").strip(); d=(d or "").strip()
            if len(c)<30: continue
            t=((f'"""{d}"""\n' if d else "")+c)[:8000]
            if len(t)>=40: out.append(t)
            if len(out)>=n: return out
T=text_records(6000); C=code_records(6000)
lens={"nlp":[len(enc.encode(t, disallowed_special=())) for t in T], "code":[len(enc.encode(t, disallowed_special=())) for t in C]}
MATH_LEN=int(sys.argv[1]) if len(sys.argv)>1 else 30
for perm_name in ["n_run_arch_search_v4","n_run_arch_search_v5"]:
    perm=json.load(open(f"{W}/leak/{perm_name}.json"))["perm1"]
    ev=perm[15300:]
    def L(k):
        d=k.split("-")[0]; i=int(k.split("-")[1][:5])
        return MATH_LEN if d=="math" else lens[d][i]
    hits=0; tot=0; hits_tc=0
    for lab in ["nlp","code","math"]:
        idxs=[p for p,k in enumerate(ev) if k.startswith(lab)][:50]
        for i,p in enumerate(idxs):
            start=(i+50)%len(ev); toks=0; d=start; covered=set(); scanned=0
            while toks<4096 and scanned<len(ev)+4:
                covered.add(d); toks+=L(ev[d])+1; d=(d+1)%len(ev); scanned+=1
            tot+=1
            if p in covered:
                hits+=1
                if lab!="math": hits_tc+=1
    print(perm_name, "scored docs appearing in own burn-in:", hits, "/", tot, " text+code:", hits_tc)
