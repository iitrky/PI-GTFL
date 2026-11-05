"""Train one model variant. Usage: python3 train.py <variant> [epochs]
variants: pigtfl, gt_nophys, gt_nobias, gt_vanilla, gcn, cnn"""
import numpy as np, sys, time, pickle
import ag, models, sim

def load(name):
    z = np.load(f'data_{name}.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}

def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}

TYPE_ID = {t: i for i, t in enumerate(sim.FAULT_TYPES)}

def build(variant, seed=0):
    rng = np.random.default_rng(seed)
    if variant == 'pigtfl':
        return models.GraphTransformer(rng, imp_bias=True, dv_head=True), dict(lam_p=1.0, lam_m=1.0)
    if variant == 'gt_nophys':
        return models.GraphTransformer(rng, imp_bias=True, dv_head=False), {}
    if variant == 'gt_nobias':
        return models.GraphTransformer(rng, imp_bias=False, dv_head=True), dict(lam_p=1.0, lam_m=1.0)
    if variant == 'gt_measonly':
        return models.GraphTransformer(rng, imp_bias=True, dv_head=True), dict(lam_p=0.0, lam_m=1.0)
    if variant == 'gt_vanilla':
        return models.GraphTransformer(rng, imp_bias=False, dv_head=False), {}
    if variant == 'gcn':
        return models.GCN(rng), {}
    if variant == 'cnn':
        return models.CNN1D(rng), {}
    raise ValueError(variant)

def predict(model, X, bs=512):
    L, A, Ty = [], [], []
    for i in range(0, len(X), bs):
        lg, al, ty, _ = model.forward(X[i:i + bs])
        L.append(lg.v); A.append(al.v); Ty.append(ty.v)
    return np.concatenate(L), np.concatenate(A), np.concatenate(Ty)

def evaluate(model, d, X):
    lg, al, ty = predict(model, X)
    pl = lg.argmax(1); pa = al[np.arange(len(pl)), pl]
    err = np.array([sim.point_distance(pl[i], pa[i], d['line'][i], d['alpha'][i]) for i in range(len(pl))])
    yt = np.array([TYPE_ID[t] for t in d['ftype']])
    return dict(sec_acc=float((pl == d['line']).mean()), mae_km=float(err.mean()),
                med_km=float(np.median(err)), p95_km=float(np.percentile(err, 95)),
                within05=float((err <= 0.5).mean()), type_acc=float((ty.argmax(1) == yt).mean())), err

if __name__ == '__main__':
    variant = sys.argv[1]; epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tr, va = load('train'), load('val')
    ntr = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tag = f'{variant}_s{seed}' + (f'_n{ntr}' if ntr else '')
    if ntr:
        tr = subset(tr, np.arange(ntr))
    model, cfg = build(variant, seed)
    params = model.params()
    print(variant, 'params', sum(p.v.size for p in params), flush=True)
    opt = ag.Adam(params, lr=1e-3)
    rng = np.random.default_rng(100 + seed)
    yl, ya = tr['line'], tr['alpha']; yt = np.array([TYPE_ID[t] for t in tr['ftype']])
    bs = 128 if not ntr else 64; nb = len(yl) // bs; total = epochs * nb; step = 0
    Xv, _, _, _ = models.make_features(va, np.random.default_rng(7), tve=0.005)
    best, t0, hist = -1, time.time(), []
    for ep in range(epochs):
        # fresh noise / PMU-drop augmentation every epoch
        tve = rng.uniform(0, 0.01, len(yl))
        X, mask, dvm, _ = models.make_features(tr, rng, tve=tve, drop_random=3)
        perm = rng.permutation(len(yl))
        agg = {}
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            opt.lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / total)); step += 1
            out = model.forward(X[idx])
            L, parts = models.loss_fn(out, yl[idx], ya[idx], yt[idx], mask[idx], dvm[idx], **cfg)
            L.backward(); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / nb
        m, _ = evaluate(model, va, Xv)
        hist.append(dict(epoch=ep, **agg, **{'val_' + k: v for k, v in m.items()}))
        score = m['sec_acc'] - 0.01 * m['mae_km']
        if score > best:
            best = score
            pickle.dump([p.v.copy() for p in params], open(f'w_{tag}.pkl', 'wb'))
        print(ep, {k: round(v, 4) for k, v in agg.items()}, {k: round(v, 4) for k, v in m.items()},
              f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(hist, open(f'hist_{tag}.pkl', 'wb'))
