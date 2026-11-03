"""Input sensitivity of the true E20 configuration and v1 with real sqrt(D)-scaled embeddings (section 8)."""
import sys
import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import torch, json
torch.set_num_threads(2)
from cellai_core.base import ModelParams
from v3.cell_ai_v3 import CellAIv3, V3Config
from v1.cell_ai import CellAI
out={}
for name, mk in [("E20", lambda: CellAIv3(ModelParams(device="cpu", state_size=512, num_partitions=8), V3Config(pde_type="spectral", hebbian_type="sparse", partition_type="multiscale", N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8))),
                 ("v1", lambda: CellAI(ModelParams(device="cpu")))]:
    m = mk(); m.eval()
    ids1 = torch.randint(1,100000,(256,), generator=torch.Generator().manual_seed(1))
    ids2 = torch.randint(1,100000,(256,), generator=torch.Generator().manual_seed(2))
    def run(ids, seed=7):
        m.partitions.reset(); m.memory_formation.reset(); torch.manual_seed(seed)
        with torch.no_grad():
            for t in ids: m.cellular_step(m.encoder.embedding.weight[t]*m.encoder._scale)
        if name=="E20":
            return m.partitions.aggregate().clone(), m.partitions.fast_pm.aggregate().clone(), m.partitions.slow_pm.aggregate().clone()
        return (m.partitions.aggregate().clone(),)
    A=run(ids1); B=run(ids2)
    out[name]=[float((a-b).norm()/a.norm()) for a,b in zip(A,B)]
    out[name+"_norms"]=[float(a.norm()) for a in A]
    if name=="v1":
        C=run(ids1, seed=8); out["v1_noise_only"]=float((A[0]-C[0]).norm()/A[0].norm())
print(json.dumps(out,indent=1))
