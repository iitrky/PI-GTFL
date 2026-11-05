"""Evaluate all trained variants under every test condition; writes results.json.
Variants with several seeds are reported as mean (and std across seeds)."""
import numpy as np, pickle, json, time, os, glob
import models, sim, train

VARIANTS = ['pigtfl', 'gt_nophys', 'gt_nobias', 'gt_measonly', 'gt_vanilla', 'gcn', 'cnn']


def seeds_of(v):
    return sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(f'w_{v}_s[0-9].pkl'))


def load_model(v, seed, suffix=''):
    m, _ = train.build(v, seed)
    for p, w in zip(m.params(), pickle.load(open(f'w_{v}_s{seed}{suffix}.pkl', 'rb'))):
        p.v[...] = w
    return m


def imp_eval(d, meas):
    pr = models.impedance_locate(meas, d['ftype'], truth=(d['line'], d['alpha']))
    err = np.array([sim.point_distance(l, a, d['line'][i], d['alpha'][i]) for i, (l, a) in enumerate(pr)])
    pl = np.array([p[0] for p in pr])
    return dict(sec_acc=float((pl == d['line']).mean()), mae_km=float(err.mean()), med_km=float(np.median(err)),
                p95_km=float(np.percentile(err, 95)), within05=float((err <= 0.5).mean()), type_acc=None), err


def recon_eval(m, d, X, mask):
    """Relative error of reconstructed incremental sequence voltages at buses without a uPMU,
    and mean normalised KCL residual at injection-free buses."""
    dv = np.concatenate([m.forward(X[i:i + 500])[3].v for i in range(0, len(X), 500)]) / models.V_SCALE
    rot = np.exp(-1j * np.angle(d['Vpre'][:, :1]))
    true = np.stack([(d['V1'] - d['Vpre']) * rot, d['V2'] * rot, d['V0'] * rot], -1)
    est = dv[..., 0::2] + 1j * dv[..., 1::2]
    um = mask == 0
    rel = np.linalg.norm((est - true)[um]) / np.linalg.norm(true[um])
    res = []
    for s in range(3):
        G, B = models.YT[s]
        Yt = G + 1j * B
        r = est[..., s] @ Yt.T
        rt = true[..., s] @ Yt.T
        keep = np.ones(r.shape, bool)
        keep[np.arange(len(r)), sim.FROM[d['line']]] = False; keep[np.arange(len(r)), sim.TO[d['line']]] = False
        if s == 0:
            keep[:, sim.IBR_BUSES] = False
        res.append((np.abs(r[keep]).mean(), np.abs(rt[keep]).mean()))
    return dict(rel_err_unmeasured=float(rel), kcl_resid_est=float(np.mean([a for a, _ in res])),
                kcl_resid_true=float(np.mean([b for _, b in res])))


def main():
    te = train.load('test')
    pens = {p: train.load(f'pen{p}') for p in [0, 30, 60, 90]}
    S = {v: seeds_of(v) for v in VARIANTS}
    S = {v: s for v, s in S.items() if s}
    ms = {(v, s): load_model(v, s) for v, ss in S.items() for s in ss}
    print('seeds', S, flush=True)
    R = {'seeds': S}

    def run(tag, d, **kw):
        rng = np.random.default_rng(12345)
        X, mask, dvm, meas = models.make_features(d, rng, **kw)
        R[tag] = {}
        for v, ss in S.items():
            per = [train.evaluate(ms[(v, s)], d, X)[0] for s in ss]
            R[tag][v] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            R[tag][v]['std_sec_acc'] = float(np.std([p['sec_acc'] for p in per]))
            R[tag][v]['std_mae_km'] = float(np.std([p['mae_km'] for p in per]))
            R[tag][v]['n_seeds'] = len(ss)
        R[tag]['impedance'] = imp_eval(d, meas)[0]
        print(tag, {v: (round(r['sec_acc'], 4), round(r['mae_km'], 3)) for v, r in R[tag].items()}, flush=True)
        return X, mask, meas

    X, mask, meas = run('main', te, tve=0.005)
    errs = {v: train.evaluate(ms[(v, 0)], te, X)[1] for v in S}
    errs['impedance'] = imp_eval(te, meas)[1]
    br = {}
    for ft in sim.FAULT_TYPES:
        idx = te['ftype'] == ft
        br[ft] = {v: float(e[idx].mean()) for v, e in errs.items()}
    rf = te['rf_ohm']
    for lo, hi in [(0, 1), (1, 10), (10, 25), (25, 50)]:
        idx = (rf >= lo) & (rf < hi)
        br[f'Rf {lo}-{hi}'] = {v: float(e[idx].mean()) for v, e in errs.items()}
    R['breakdown_mae_km'] = br
    np.savez('err_main.npz', **{v: e for v, e in errs.items()})
    from scipy.stats import wilcoxon
    R['wilcoxon_vs_pigtfl'] = {v: float(wilcoxon(errs['pigtfl'], e).pvalue) for v, e in errs.items() if v != 'pigtfl'}
    for t in [0.0, 0.01, 0.02, 0.03]:
        run(f'tve{t}', te, tve=t)
    for k in [1, 2, 3, 4, 5]:
        run(f'drop{k}', te, tve=0.005, n_drop=k)
    for p, d in pens.items():
        run(f'pen{p}', d, tve=0.005)
    R['recon'] = {}
    for k in [0, 3]:
        Xk, mk, _, _ = models.make_features(te, np.random.default_rng(99), tve=0.005, n_drop=k)
        R['recon'][f'drop{k}'] = {v: recon_eval(ms[(v, 0)], te, Xk, mk)
                                  for v in ['pigtfl', 'gt_nobias', 'gt_measonly'] if v in S}
    print('recon', R['recon'], flush=True)
    R['lowlabel'] = {}
    Xl, _, _, _ = models.make_features(te, np.random.default_rng(12345), tve=0.005)
    Xl3, _, _, _ = models.make_features(te, np.random.default_rng(12345), tve=0.005, n_drop=3)
    for n in (2000, 500):
        for v in ['pigtfl', 'gt_nophys', 'gt_nobias', 'gt_vanilla', 'cnn']:
            if os.path.exists(f'w_{v}_s0_n{n}.pkl'):
                m = load_model(v, 0, f'_n{n}')
                R['lowlabel'][f'{v}_n{n}'] = {'all_pmu': train.evaluate(m, te, Xl)[0], 'drop3': train.evaluate(m, te, Xl3)[0]}
    print('lowlabel', R['lowlabel'], flush=True)
    lat = {}
    for v in S:
        m = ms[(v, 0)]; ts = []
        for _ in range(300):
            t = time.perf_counter(); m.forward(X[:1]); ts.append(time.perf_counter() - t)
        tb = time.perf_counter(); train.predict(m, X[:1000], bs=1000); tb = (time.perf_counter() - tb) / 1000
        lat[v] = dict(single_ms_median=1e3 * float(np.median(ts)), single_ms_p99=1e3 * float(np.percentile(ts, 99)),
                      batch_ms_per_sample=1e3 * tb, params=int(sum(p.v.size for p in m.params())))
    sub = {k: v[:200] for k, v in te.items()}
    _, _, _, mm = models.make_features(sub, np.random.default_rng(0))
    t = time.perf_counter(); models.impedance_locate(mm, sub['ftype']); lat['impedance'] = dict(single_ms_median=1e3 * (time.perf_counter() - t) / 200)
    R['latency'] = lat
    print(lat)
    m = ms[('pigtfl', 0)]
    R['gamma'] = [np.log1p(np.exp(l['gam'].v.ravel())).tolist() for l in m.L]
    R['beta'] = [l['bet'].v.ravel().tolist() for l in m.L]
    json.dump(R, open('results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
