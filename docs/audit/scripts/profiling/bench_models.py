from common import *
import torch.nn.functional as F
name = sys.argv[1]
r0 = rss_mb()
torch.manual_seed(0)
m = build(name)
r1 = rss_mb()
total = sum(p.numel() for p in m.parameters())
emb = m.encoder.embedding.weight.numel()
breakdown = {}
for n, p in m.named_parameters():
    top = ".".join(n.split(".")[:2]) if n.startswith("partitions") else n.split(".")[0]
    breakdown[top] = breakdown.get(top, 0) + p.numel()
D = m.params.state_size
x = torch.randn(D) * 1.0
m.train()
# forward: one cellular step (no LM head)
with torch.no_grad():
    fwd = timeit(lambda: m.cellular_step(x), reps=300, warmup=100)
    # forward incl. tied LM head (D x V matmul)
    def full():
        s = m.cellular_step(x); return s @ m.encoder.embedding.weight.t()
    fwd_head = timeit(full, reps=100, warmup=30)
# forward+backward through a 64-token segment (as in train_step_sequential), per token
ids = torch.randint(1, 100000, (65,))
def seg():
    m.zero_grad(set_to_none=True)
    if hasattr(m.partitions, "reset"): m.partitions.reset()
    embs = m.encoder.embedding(ids) * m.encoder._scale
    loss = 0
    for t in range(64):
        s = m.cellular_step(embs[t])
        loss = loss + F.cross_entropy((s @ m.encoder.embedding.weight.t()).unsqueeze(0), ids[t+1:t+2])
    t0 = time.perf_counter()
    (loss/64).backward()
    seg.bwd = (time.perf_counter() - t0) * 1000
    if hasattr(m.metaplasticity, "apply_pending_update"): m.metaplasticity.apply_pending_update()
segt = []; bwdt = []
for i in range(6):
    t0 = time.perf_counter(); seg(); dt = (time.perf_counter() - t0) * 1000
    if i >= 1: segt.append(dt); bwdt.append(seg.bwd)
segm = statistics.median(segt); bwdm = statistics.median(bwdt)
res = dict(name=name, D=D, N=m.params.num_partitions, total_params=total, embedding_params=emb,
           embedding_share=emb/total, non_embedding_params=total-emb, breakdown=breakdown,
           fwd_cellular_step=fwd, fwd_with_lm_head=fwd_head,
           tokens_per_s_inference=1000/fwd_head["median_ms"],
           train_seg64_ms=segm, train_seg64_backward_ms=bwdm,
           train_fwd_per_token_ms=(segm-bwdm)/64, train_bwd_per_token_ms=bwdm/64,
           train_tokens_per_s=64*1000/segm,
           rss_after_build_mb=r1, rss_build_delta_mb=r1-r0, peak_rss_mb=rss_mb())
print(json.dumps(res))
