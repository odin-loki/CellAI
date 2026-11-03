from common import *
from v2.cell_ai_v2 import ResonanceSystem, CrystalLattice
r = ResonanceSystem(256); s = torch.randn(256)
out = {}
for phi in [0.0, 0.3, 1.0, 2.0]:
    r.phase.data.fill_(phi)
    y = r(s); out[f"phi={phi}"] = float((y - torch.cos(torch.tensor(phi))*s).abs().max())
r.phase.data.zero_(); r.zero_grad(); r(s).pow(2).sum().backward()
out["grad_phase_at_init"] = float(r.phase.grad)
L = CrystalLattice(256); M = torch.einsum("ijkd,ijk,ijke->de", L.Phi, L.T, L.Phi)
out["lattice_equals_linear_map_maxerr"] = float((L(s) - M @ s).abs().max())
out["lattice_effective_rank"] = int(torch.linalg.matrix_rank(M))
print(json.dumps(out)); json.dump(out, open(f"{OUT}/resonance_lattice_check.json","w"), indent=1)
