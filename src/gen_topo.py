"""Generate multi-topology datasets. Usage: python3 gen_topo.py <part 0|1>"""
import numpy as np, sys, pickle
import sim, topo, gen

train, test = topo.library()
pickle.dump({'train': train, 'test': test}, open('topologies.pkl', 'wb'))
part = int(sys.argv[1])
jobs = [('tr', i, t, 1700, 500 + i) for i, t in enumerate(train)] + \
       [('va', i, t, 150, 700 + i) for i, t in enumerate(train)] + \
       [('te', i, t, 1000, 900 + i) for i, t in enumerate(test)]
for j, (kind, i, t, n, seed) in enumerate(jobs):
    if j % 2 != part:
        continue
    topo.use(t)
    d = gen.build(n, seed)
    np.savez_compressed(f'topo_{kind}{i}.npz', **d)
    print(kind, i, flush=True)
