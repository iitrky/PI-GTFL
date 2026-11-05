"""Algorithm 2: calibrated adaptive fault-section set (split-conformal, APS-style) on PI-GTFL posteriors.
Threshold is calibrated on the validation set and evaluated on the test set."""
import numpy as np, json
import models, sim, train, evaluate
from conf_eval import auroc

def post(m, X):
    lg = np.concatenate([m.forward(X[i:i+500])[0].v for i in range(0, len(X), 500)])
    p = np.exp(lg - lg.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)

def aps_score(p, y):  # LAC nonconformity: 1 - posterior of the true section
    return 1 - p[np.arange(len(y)), y]

def aps_set(p, qhat):  # all sections with posterior >= 1 - qhat (at least the top one)
    o = np.argsort(-p, 1); ps = np.take_along_axis(p, o, 1)
    k = np.maximum((ps >= 1 - qhat).sum(1), 1)
    return o, k

m = evaluate.load_model('pigtfl', 0)
va, te = train.load('val'), train.load('test')
R = {}
for tag, kw in (('all', {}), ('drop3', {'n_drop': 3})):
    Xv, _, _, _ = models.make_features(va, np.random.default_rng(7), tve=0.005, **kw)
    Xt, _, _, _ = models.make_features(te, np.random.default_rng(12345), tve=0.005, **kw)
    pv, pt = post(m, Xv), post(m, Xt)
    r = {'auroc_pmax': auroc(-pt.max(1), pt.argmax(1) != te['line'])}
    for alpha in (0.05, 0.02):
        s = aps_score(pv, va['line']); n = len(s)
        qhat = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method='higher')
        o, k = aps_set(pt, qhat)
        y = te['line']; cover = np.array([y[i] in o[i, :k[i]] for i in range(len(y))])
        patrol = np.array([sim.LEN_KM[o[i, :k[i]]].sum() for i in range(len(y))])
        r[f'a{int(alpha*100)}'] = dict(qhat=float(qhat), coverage=float(cover.mean()), mean_set=float(k.mean()),
                                        single_share=float((k == 1).mean()),
                                        single_acc=float((o[k == 1, 0] == y[k == 1]).mean()),
                                        mean_patrol_km=float(patrol.mean()))
    r['top1_acc'] = float((pt.argmax(1) == te['line']).mean())
    r['top1_patrol_km'] = float(sim.LEN_KM[pt.argmax(1)].mean())
    R[tag] = r; print(tag, json.dumps(r))
json.dump(R, open('results_conformal.json', 'w'), indent=1)
