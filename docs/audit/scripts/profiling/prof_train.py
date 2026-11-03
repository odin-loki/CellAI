from common import *
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
name = sys.argv[1]; T = 32
m = build(name); m.train()
ids = torch.randint(1, 100000, (T+1,))
def seg(batched_head=False):
    m.zero_grad(set_to_none=True); m.partitions.reset()
    embs = m.encoder.embedding(ids) * m.encoder._scale
    if batched_head:
        S = torch.stack([m.cellular_step(embs[t]) for t in range(T)])
        loss = F.cross_entropy(S @ m.encoder.embedding.weight.t(), ids[1:])
    else:
        loss = 0
        for t in range(T):
            s = m.cellular_step(embs[t])
            loss = loss + F.cross_entropy((s @ m.encoder.embedding.weight.t()).unsqueeze(0), ids[t+1:t+2])
        loss = loss / T
    loss.backward()
    if hasattr(m.metaplasticity, "apply_pending_update"): m.metaplasticity.apply_pending_update()
seg(); seg(True)
out = {}
for b in (False, True):
    ts = []
    for _ in range(4):
        t0 = time.perf_counter(); seg(b); ts.append((time.perf_counter()-t0)*1000)
    out["batched_head" if b else "per_token_head"] = statistics.median(ts)/T
print("PER_TOKEN_TRAIN_MS", json.dumps(out))
with profile(activities=[ProfilerActivity.CPU]) as p:
    seg(False)
tab = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
print(tab)
open(f"{OUT}/profiler_{name}.txt","w").write(f"{name}: one forward+backward over a {T}-token segment (per-token tied LM head, as in train_step_sequential)\n"+json.dumps(out)+"\n"+tab)
