"""Topology-randomised training: every minibatch comes from one of 12 training topologies and the
model receives that topology's graph quantities. Usage: python3 train_topo.py <variant> [epochs] [seed]"""
import numpy as np, sys, time, pickle
import ag, models, sim, topo, train

TOPS = pickle.load(open('topologies.pkl', 'rb'))
TYPE_ID = train.TYPE_ID


def load_set(kind, n):
    out = []
    for i in range(n):
        z = np.load(f'topo_{kind}{i}.npz', allow_pickle=True)
        out.append({k: z[k] for k in z.files})
    return out


def eval_sets(model, sets, tops, Xs):
    out = []
    for d, t, X in zip(sets, tops, Xs):
        topo.use(t)
        m, _ = train.evaluate(model, d, X)
        out.append(m)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}, out


if __name__ == '__main__':
    variant = sys.argv[1]; epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tr = load_set('tr', 12); va = load_set('va', 12)
    model, cfg = train.build(variant, seed)
    params = model.params()
    opt = ag.Adam(params, lr=1e-3)
    rng = np.random.default_rng(300 + seed)
    Xv = [models.make_features(d, np.random.default_rng(7), tve=0.005)[0] for d in va]
    bs = 128
    nb_per = [len(d['line']) // bs for d in tr]
    total = epochs * sum(nb_per); step = 0; best = -1; t0 = time.time(); hist = []
    for ep in range(epochs):
        feats = []
        for d in tr:
            X, mask, dvm, _ = models.make_features(d, rng, tve=rng.uniform(0, 0.01, len(d['line'])), drop_random=3)
            feats.append((X, mask, dvm, np.array([TYPE_ID[t] for t in d['ftype']])))
        batches = [(ti, p[b * bs:(b + 1) * bs]) for ti, d in enumerate(tr)
                   for p in [rng.permutation(len(d['line']))] for b in range(nb_per[ti])]
        order = rng.permutation(len(batches))
        agg = {}
        for bi in order:
            ti, idx = batches[bi]
            topo.use(TOPS['train'][ti])
            models.LAPPE = models.LAPPE * rng.choice([-1.0, 1.0], 8).astype(np.float32)  # PE sign flip
            X, mask, dvm, yt = feats[ti]; d = tr[ti]
            opt.lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / total)); step += 1
            out = model.forward(X[idx])
            L, parts = models.loss_fn(out, d['line'][idx], d['alpha'][idx], yt[idx], mask[idx], dvm[idx], **cfg)
            L.backward(); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(batches)
        topo._CACHE.clear()  # restore unflipped PE
        m, _ = eval_sets(model, va, TOPS['train'], Xv)
        hist.append(dict(epoch=ep, **agg, **{'val_' + k: v for k, v in m.items()}))
        score = m['sec_acc'] - 0.01 * m['mae_km']
        if score > best:
            best = score
            pickle.dump([p.v.copy() for p in params], open(f'wt_{variant}_s{seed}.pkl', 'wb'))
        print(ep, {k: round(v, 4) for k, v in agg.items()}, {k: round(v, 4) for k, v in m.items()},
              f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(hist, open(f'histt_{variant}_s{seed}.pkl', 'wb'))
