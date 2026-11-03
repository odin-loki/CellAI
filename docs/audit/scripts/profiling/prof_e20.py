from common import *
m = build(sys.argv[1]); m.train()
x = torch.randn(m.params.state_size)
with torch.no_grad():
    for _ in range(20): m.cellular_step(x)
    r = timeit(lambda: m.cellular_step(x), reps=100)
print(r)
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as p:
    for _ in range(20): m.cellular_step(x)
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
