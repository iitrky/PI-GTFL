"""Evaluate topology-randomised models on seen and unseen topologies; writes results_topo_train.json."""
import numpy as np, json, glob, pickle
import models, sim, topo, train, evaluate

TOPS = pickle.load(open('topologies.pkl', 'rb'))
VARS = ['pigtfl', 'gt_nophys', 'gt_nobias', 'gt_vanilla', 'gcn']


def ld(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_t(v, s):
    m, _ = train.build(v, s)
    for p, w in zip(m.params(), pickle.load(open(f'wt_{v}_s{s}.pkl', 'rb'))):
        p.v[...] = w
    return m


sets = [('base (seen)', sim.BASE_LINES.copy(), train.load('test'))]
names = ['Baran-Wu (unseen)', 'Random A (unseen)', 'Random B (unseen)', 'Random C (unseen)']
for i, t in enumerate(TOPS['test']):
    sets.append((names[i], t, ld(f'topo_te{i}.npz')))
R = {'per_topology': {}, 'n_ties_closed': {}}
base_key = {tuple(sorted((int(a), int(b)))) for a, b in sim.BASE_LINES[:, :2]}
for name, t, _ in sets:
    R['n_ties_closed'][name] = int(sum(tuple(sorted((int(a), int(b)))) not in base_key for a, b in t[:, :2]))
S = {v: sorted(int(f.split('_s')[-1][0]) for f in glob.glob(f'wt_{v}_s[0-9].pkl')) for v in VARS}
S = {v: s for v, s in S.items() if s}
M = {(v, s): load_t(v, s) for v, ss in S.items() for s in ss}
B = {v: evaluate.load_model(v, 0) for v in ['pigtfl', 'gt_vanilla', 'gcn']}   # base-topology-trained
for name, t, d in sets:
    topo.use(t)
    R['per_topology'][name] = {}
    for tag, kw in (('all', {}), ('drop3', {'n_drop': 3})):
        X, mask, _, meas = models.make_features(d, np.random.default_rng(12345), tve=0.005, **kw)
        r = {}
        for v, ss in S.items():
            per = [train.evaluate(M[(v, s)], d, X)[0] for s in ss]
            r[f'{v} (multi-topo)'] = {k: float(np.mean([p[k] for p in per])) for k in ('sec_acc', 'mae_km', 'within05')}
            r[f'{v} (multi-topo)']['std_sec_acc'] = float(np.std([p['sec_acc'] for p in per]))
            r[f'{v} (multi-topo)']['n_seeds'] = len(ss)
        for v, m in B.items():
            p = train.evaluate(m, d, X)[0]
            r[f'{v} (base only)'] = {k: p[k] for k in ('sec_acc', 'mae_km', 'within05')}
        if tag == 'all':
            p = evaluate.imp_eval(d, meas)[0]
            r['impedance (oracle)'] = {k: p[k] for k in ('sec_acc', 'mae_km', 'within05')}
            r['recon_pigtfl'] = evaluate.recon_eval(M[('pigtfl', 0)], d, X, mask)
        R['per_topology'][name][tag] = r
        print(name, tag, {k: (round(x['sec_acc'], 3), round(x['mae_km'], 3)) for k, x in r.items() if 'sec_acc' in x}, flush=True)
# mean over the four unseen topologies
un = [n for n in R['per_topology'] if 'unseen' in n]
R['unseen_mean'] = {}
for tag in ('all', 'drop3'):
    keys = [k for k in R['per_topology'][un[0]][tag] if k != 'recon_pigtfl']
    R['unseen_mean'][tag] = {k: {m: float(np.mean([R['per_topology'][n][tag][k][m] for n in un]))
                                 for m in ('sec_acc', 'mae_km', 'within05')} for k in keys}
R['recon_unseen_mean'] = float(np.mean([R['per_topology'][n]['all']['recon_pigtfl']['rel_err_unmeasured'] for n in un]))
print('UNSEEN MEAN', json.dumps(R['unseen_mean'], indent=0))
print('ties', R['n_ties_closed'])
json.dump(R, open('results_topo_train.json', 'w'), indent=1)
