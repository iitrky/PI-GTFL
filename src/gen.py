"""Generate fault datasets and save raw phasors."""
import numpy as np, sys, time
import sim

def build(n, seed, pen=None):
    rng = np.random.default_rng(seed)
    keys = ['Vpre', 'V1', 'V2', 'V0', 'Isub', 'Ipre_sub', 'rating']
    out = {k: [] for k in keys}
    meta = {k: [] for k in ['line', 'alpha', 'ftype', 'rf_ohm', 'load_scale', 'pen', 'ssc_mva']}
    for i in range(n):
        sc = sample_scenario = sim.sample_scenario(rng, pen=pen)
        r = sim.simulate(rng, **sc)
        for k in keys:
            out[k].append(r[k])
        for k in meta:
            meta[k].append(sc[k])
    d = {k: np.array(v) for k, v in out.items()}
    d.update({k: np.array(v) for k, v in meta.items()})
    return d

if __name__ == '__main__':
    t = time.time()
    sets = {'train': (20000, 1, None), 'val': (2000, 2, None), 'test': (5000, 3, None)}
    for p in [0.0, 0.3, 0.6, 0.9]:
        sets[f'pen{int(p*100)}'] = (1500, 10 + int(p * 10), p)
    for name, (n, seed, pen) in sets.items():
        d = build(n, seed, pen)
        np.savez_compressed(f'data_{name}.npz', **d)
        print(name, n, f'{time.time()-t:.1f}s', flush=True)
