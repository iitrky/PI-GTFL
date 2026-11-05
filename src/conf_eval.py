"""Algorithm 2 evaluation: physics-consistency score and confidence-gated top-k candidate list."""
import numpy as np, json
import models, sim, train, evaluate

def scores(m, d, X, mask):
    lg, al, ty, dv = [], [], [], []
    for i in range(0, len(X), 500):
        o = m.forward(X[i:i+500]); lg.append(o[0].v); al.append(o[1].v); dv.append(o[3].v)
    lg = np.concatenate(lg); al = np.concatenate(al); dv = np.concatenate(dv) / models.V_SCALE
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    est = dv[..., 0::2] + 1j * dv[..., 1::2]
    pk = p.argmax(1)
    res = np.zeros(len(X))
    for s in range(3):
        G, B = models.YT[s]; r = np.abs(est[..., s] @ (G + 1j * B).T)
        keep = np.ones(r.shape, bool)
        keep[np.arange(len(r)), sim.FROM[pk]] = False; keep[np.arange(len(r)), sim.TO[pk]] = False
        if s == 0: keep[:, sim.IBR_BUSES] = False
        res += (r * keep).sum(1) / keep.sum(1)
    return p, al, res / 3

def auroc(score_bad, wrong):
    from scipy.stats import rankdata
    r = rankdata(score_bad); n1 = wrong.sum(); n0 = len(wrong) - n1
    return float((r[wrong].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

if __name__ == '__main__':
  te = train.load('test'); m = evaluate.load_model('pigtfl', 0)
  R = {}
  for tag, kw in (('all', {}), ('drop3', {'n_drop': 3})):
      X, mask, _, _ = models.make_features(te, np.random.default_rng(12345), tve=0.005, **kw)
      p, al, res = scores(m, te, X, mask)
      y = te['line']; pk = p.argmax(1); wrong = pk != y
      pmax = p.max(1)
      # combined physics-aware confidence: high prob and low residual
      zr = (res - res.mean()) / res.std(); zp = (pmax - pmax.mean()) / pmax.std()
      comb = zp - zr
      r = {'acc': float(1 - wrong.mean()),
           'auroc_pmax': auroc(-pmax, wrong), 'auroc_residual': auroc(res, wrong), 'auroc_combined': auroc(-comb, wrong)}
      for cov in (0.9, 0.8, 0.7):
          thr = np.quantile(comb, 1 - cov); sel = comb >= thr
          r[f'acc_at_{int(cov*100)}cov_combined'] = float(1 - wrong[sel].mean())
          thr2 = np.quantile(pmax, 1 - cov); sel2 = pmax >= thr2
          r[f'acc_at_{int(cov*100)}cov_pmax'] = float(1 - wrong[sel2].mean())
      top3 = np.argsort(-p, 1)[:, :3]
      r['top3_hit'] = float((top3 == y[:, None]).any(1).mean())
      # Algorithm 2 policy: single answer if comb >= tau (tau at 80 % coverage), else top-3 list
      tau = np.quantile(comb, 0.2); single = comb >= tau
      hit = np.where(single, pk == y, (top3 == y[:, None]).any(1))
      patrol = np.where(single, sim.LEN_KM[pk], sim.LEN_KM[top3].sum(1))
      r['policy_single_share'] = float(single.mean()); r['policy_hit'] = float(hit.mean())
      r['policy_mean_patrol_km'] = float(patrol.mean()); r['single_only_patrol_km'] = float(sim.LEN_KM[pk].mean())
      R[tag] = r
      print(tag, {k: round(v, 4) for k, v in r.items()})
  json.dump(R, open('results_conf.json', 'w'), indent=1)
