"""Features, PI-GTFL, GCN and CNN fault locators, and the impedance-based baseline."""
import numpy as np
import ag
from ag import T, Param, glorot, concat
import sim

N, NL = sim.N, sim.NL
F_IN = 20
NON_SUB_PMU = sim.PMU_BUSES[1:]

# ---------------- static graph quantities ----------------
A = np.eye(N)
A[sim.FROM, sim.TO] = 1; A[sim.TO, sim.FROM] = 1
deg = A.sum(1)
A_HAT = (A / np.sqrt(deg)[:, None] / np.sqrt(deg)[None, :]).astype(np.float32)
Lap = np.diag(A.sum(1) - 1) - (A - np.eye(N))
_, evec = np.linalg.eigh(Lap)
LAPPE = evec[:, 1:9].astype(np.float32)          # 8 Laplacian eigenvectors


def nominal_Y():
    S = (sim.LOADS_KW + 1j * sim.LOADS_KVAR) / 1000 / sim.S_BASE_MVA
    zs = (1 / (150 / sim.S_BASE_MVA)) * (0.1 + 1j) / abs(0.1 + 1j)
    sh1 = np.conj(S).astype(complex); sh1[0] += 1 / zs
    sh0 = np.zeros(N, complex); sh0[0] += 1 / zs
    Y1 = sim.ybus(sim.Z1_LINE, sh1)[:N, :N]
    Y0 = sim.ybus(sim.Z0_LINE, sh0)[:N, :N]
    return Y1, Y0


Y1N, Y0N = nominal_Y()
Z1N = np.linalg.inv(Y1N)
ZE = np.abs(np.diag(Z1N)[:, None] + np.diag(Z1N)[None, :] - 2 * Z1N)   # electrical distance
ZE_N = (ZE / ZE.mean()).astype(np.float32)
# diagonal-normalised admittance matrices used in the physics residual
YT = []
for Y in (Y1N, Y1N, Y0N):
    Yt = Y / np.diag(Y)[:, None]
    YT.append((Yt.real.astype(np.float32), Yt.imag.astype(np.float32)))
IBR_MASK = np.ones(N, np.float32); IBR_MASK[sim.IBR_BUSES] = 0
LINE_FEATS = np.stack([sim.LINES[:, 2], sim.LINES[:, 3], sim.LEN_KM], 1).astype(np.float32)
LINE_FEATS = LINE_FEATS / LINE_FEATS.mean(0)
INC = np.zeros((NL, N), np.float32)              # line-node incidence
INC[np.arange(NL), sim.FROM] = 1; INC[np.arange(NL), sim.TO] = 1

V_SCALE = 5.0
I_SCALE = 0.2


def make_features(d, rng, tve=0.005, n_drop=0, drop_random=None, pmus=None):
    """d: dataset dict. Returns X (B,N,F), mask (B,N), dV_meas (B,N,6) in scaled units."""
    B = len(d['line'])
    pm = sim.PMU_BUSES if pmus is None else pmus
    mask = np.zeros((B, N), np.float32); mask[:, pm] = 1
    if drop_random is not None:  # per-sample random number of drops in [0, drop_random]
        for b in range(B):
            k = rng.integers(0, drop_random + 1)
            if k:
                mask[b, rng.choice(NON_SUB_PMU, k, replace=False)] = 0
    elif n_drop:
        for b in range(B):
            mask[b, rng.choice(NON_SUB_PMU, n_drop, replace=False)] = 0
    tv = (np.full(B, tve) if np.isscalar(tve) else tve)[:, None]
    def cn(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    Vpre = d['Vpre'] + tv * cn((B, N))
    V1 = d['V1'] + tv * cn((B, N)); V2 = d['V2'] + tv * cn((B, N)); V0 = d['V0'] + tv * cn((B, N))
    Is = d['Isub'] + tv * np.abs(d['Isub']) * cn((B, 3))
    Ipre = d['Ipre_sub'] + tv[:, 0] * np.abs(d['Ipre_sub']) * cn(B)
    rot = np.exp(-1j * np.angle(Vpre[:, :1]))
    dV = np.stack([(V1 - Vpre) * rot, V2 * rot, V0 * rot], -1) * mask[..., None]
    raw = np.concatenate([dV.real, dV.imag], -1)[..., [0, 3, 1, 4, 2, 5]] * V_SCALE
    mag = np.abs(dV).sum(-1).max(1)[:, None, None] + 1e-6
    nrm = np.concatenate([(dV / mag).real, (dV / mag).imag], -1)[..., [0, 3, 1, 4, 2, 5]]
    Iv = np.stack([Is[:, 0] - Ipre, Is[:, 1], Is[:, 2]], -1) * rot[:, :1]
    X = np.zeros((B, N, F_IN), np.float32)
    X[..., 0:6] = raw; X[..., 6:12] = nrm
    X[:, 0, 12:18] = np.concatenate([Iv.real, Iv.imag], -1)[:, [0, 3, 1, 4, 2, 5]] * I_SCALE * mask[:, :1]
    X[..., 18] = mask
    X[..., 19] = d['rating'] * 2
    meas = {'Vs': (V1, V2, V0, Vpre), 'Is': Is, 'Ipre': Ipre}
    return X, mask, raw.astype(np.float32), meas


# ---------------- layers ----------------
class Lin:
    def __init__(self, rng, i, o):
        self.W = glorot(rng, i, o); self.b = Param(np.zeros(o))
    def __call__(self, x):
        return x @ self.W + self.b
    def params(self):
        return [self.W, self.b]


class LN:
    def __init__(self, d):
        self.g = Param(np.ones(d)); self.b = Param(np.zeros(d))
    def __call__(self, x):
        return x.layernorm() * self.g + self.b
    def params(self):
        return [self.g, self.b]


class Heads:
    """Edge (section) classification, per-edge distance, fault type, optional node dV head."""
    def __init__(self, rng, d, dv_head):
        self.e1 = Lin(rng, 3 * d + 3, d); self.e2 = Lin(rng, d, 2)
        self.t1 = Lin(rng, d, 4)
        self.dv = Lin(rng, d, 6) if dv_head else None
    def __call__(self, H):
        B = H.v.shape[0]
        hf = H.take(sim.FROM, 1); ht = H.take(sim.TO, 1)
        lf = T(np.broadcast_to(LINE_FEATS, (B, NL, 3)))
        e = self.e2(self.e1(concat([hf, ht, hf * ht, lf], -1)).relu())
        logit = e.take(np.array([0]), 2).reshape(B, NL)
        alpha = e.take(np.array([1]), 2).reshape(B, NL).sigmoid()
        typ = self.t1(H.mean(1))
        dv = self.dv(H) if self.dv is not None else None
        return logit, alpha, typ, dv
    def params(self):
        p = self.e1.params() + self.e2.params() + self.t1.params()
        return p + (self.dv.params() if self.dv is not None else [])


class GraphTransformer:
    def __init__(self, rng, d=64, heads=4, layers=3, imp_bias=True, dv_head=True):
        self.d, self.h, self.dk = d, heads, d // heads
        self.inp = Lin(rng, F_IN, d); self.inp2 = Lin(rng, d, d); self.pe = Lin(rng, 8, d)
        self.imp_bias = imp_bias
        self.L = []
        for _ in range(layers):
            self.L.append(dict(ln1=LN(d), q=Lin(rng, d, d), k=Lin(rng, d, d), v=Lin(rng, d, d),
                               o=Lin(rng, d, d), ln2=LN(d), f1=Lin(rng, d, 2 * d), f2=Lin(rng, 2 * d, d),
                               gam=Param(np.full((heads, 1, 1), 0.5)), bet=Param(np.full((heads, 1, 1), 1.0))))
        self.heads = Heads(rng, d, dv_head)

    def attn_bias(self, l):
        if not self.imp_bias:
            return None
        return (l['gam'].softplus() * T(-ZE_N[None])) + l['bet'] * T(A[None].astype(np.float32))

    def forward(self, X):
        B = X.shape[0]
        H = self.inp2(self.inp(T(X)).relu()) + self.pe(T(LAPPE))
        self.last_attn = []
        for l in self.L:
            z = l['ln1'](H)
            def sp(t):
                return t.reshape(B, N, self.h, self.dk).transpose(0, 2, 1, 3)
            q, k, v = sp(l['q'](z)), sp(l['k'](z)), sp(l['v'](z))
            s = (q @ k.transpose(0, 1, 3, 2)) * (1 / np.sqrt(self.dk))
            bias = self.attn_bias(l)
            if bias is not None:
                s = s + bias
            a = s.softmax(-1)
            self.last_attn.append(a.v)
            o = (a @ v).transpose(0, 2, 1, 3).reshape(B, N, self.d)
            H = H + l['o'](o)
            H = H + l['f2'](l['f1'](l['ln2'](H)).relu())
        return self.heads(H)

    def params(self):
        p = self.inp.params() + self.inp2.params() + self.pe.params() + self.heads.params()
        for l in self.L:
            for k, m in l.items():
                p += [m] if isinstance(m, Param) else m.params()
        return p


class GCN:
    def __init__(self, rng, d=64, layers=3):
        self.inp = Lin(rng, F_IN, d); self.pe = Lin(rng, 8, d)
        self.L = [(Lin(rng, d, d), LN(d)) for _ in range(layers)]
        self.heads = Heads(rng, d, False)
    def forward(self, X):
        H = self.inp(T(X)).relu() + self.pe(T(LAPPE))
        Ah = T(A_HAT)
        for lin, ln in self.L:
            H = H + (Ah @ lin(ln(H))).relu()
        return self.heads(H)
    def params(self):
        p = self.inp.params() + self.pe.params() + self.heads.params()
        for lin, ln in self.L:
            p += lin.params() + ln.params()
        return p


class CNN1D:
    """1-D CNN over the µPMU measurement sequence (ordered by bus index)."""
    def __init__(self, rng, ch=64, hid=256):
        P = len(sim.PMU_BUSES)
        self.P = P
        idx = np.arange(P)
        self.win = np.stack([np.clip(idx - 1, 0, P - 1), idx, np.clip(idx + 1, 0, P - 1)], 1)  # (P,3)
        self.c1 = Lin(rng, 3 * F_IN, ch); self.c2 = Lin(rng, 3 * ch, ch)
        self.fc = Lin(rng, P * ch, hid); self.out = Lin(rng, hid, 2 * NL + 4)
    def conv(self, x, lin):
        B, P, C = x.v.shape
        w = x.take(self.win.reshape(-1), 1).reshape(B, P, 3 * C)
        return lin(w).relu()
    def forward(self, X):
        B = X.shape[0]
        x = T(X[:, sim.PMU_BUSES, :])
        h = self.conv(self.conv(x, self.c1), self.c2)
        h = self.fc(h.reshape(B, -1)).relu()
        o = self.out(h)
        logit = o.take(np.arange(NL), 1)
        alpha = o.take(np.arange(NL, 2 * NL), 1).sigmoid()
        typ = o.take(np.arange(2 * NL, 2 * NL + 4), 1)
        return logit, alpha, typ, None
    def params(self):
        return self.c1.params() + self.c2.params() + self.fc.params() + self.out.params()


# ---------------- losses ----------------
def physics_residual(dv, logit):
    """KCL residual of the reconstructed incremental sequence voltages, down-weighted where the
    model places the fault. dv: (B,N,6) scaled; returns scalar loss T."""
    B = dv.v.shape[0]
    p_edge = logit.softmax(-1)                         # (B,NL)
    p_node = p_edge @ T(INC)                           # (B,N)
    w = (p_node * -1.0 + 1.0).relu()
    tot = None
    for s in range(3):
        a = dv.take(np.array([2 * s]), 2).reshape(B, N) * (1 / V_SCALE)
        b = dv.take(np.array([2 * s + 1]), 2).reshape(B, N) * (1 / V_SCALE)
        G, Bm = YT[s]
        rr = a @ T(G.T) - b @ T(Bm.T)
        ri = b @ T(G.T) + a @ T(Bm.T)
        r2 = (rr.square() + ri.square()) * (V_SCALE ** 2)
        m = IBR_MASK if s == 0 else np.ones(N, np.float32)
        term = (r2 * w * T(m[None])).mean()
        tot = term if tot is None else tot + term
    return tot


def loss_fn(out, y_line, y_alpha, y_type, mask, dv_meas, lam_p=0.0, lam_m=0.0):
    logit, alpha, typ, dv = out
    B = len(y_line)
    oh = np.zeros((B, NL), np.float32); oh[np.arange(B), y_line] = 1
    ce = (logit.log_softmax(-1) * T(oh)).sum() * (-1.0 / B)
    da = ((alpha - T(y_alpha[:, None].astype(np.float32))).square() * T(oh)).sum() * (1.0 / B)
    ot = np.zeros((B, 4), np.float32); ot[np.arange(B), y_type] = 1
    ct = (typ.log_softmax(-1) * T(ot)).sum() * (-1.0 / B)
    L = ce + da * 10.0 + ct * 0.3
    parts = dict(ce=float(ce.v), dist=float(da.v), typ=float(ct.v))
    if dv is not None and lam_m > 0:
        lm = ((dv - T(dv_meas)).square() * T(mask[..., None])).sum() * (1.0 / (mask.sum() * 6))
        L = L + lm * lam_m; parts['meas'] = float(lm.v)
    if dv is not None and lam_p > 0:
        lp = physics_residual(dv, logit)
        L = L + lp * lam_p; parts['phys'] = float(lp.v)
    return L, parts


# ---------------- impedance-based baseline ----------------
def _paths():
    import networkx as nx
    G = nx.Graph()
    for k in range(NL):
        G.add_edge(int(sim.FROM[k]), int(sim.TO[k]), k=k)
    cum = {0: 0j}
    order = list(nx.bfs_edges(G, 0))
    lines_of = {}
    for u, v in order:
        k = G[u][v]['k']
        cum[v] = cum[u] + sim.Z1_LINE[k]
        lines_of[k] = (u, v)
    return cum


CUMZ = _paths()


def impedance_locate(meas, ftypes, truth=None):
    """Single-ended Takagi-type (superimposed-current) reactance method with known fault type.
    Solves Im[(V - Z_cum(p) I) conj(dI)] = 0 on every section. A branched feeder yields one
    candidate per lateral (multiple estimation). If truth=(lines, alphas) is given, the candidate
    closest to the true fault is returned (oracle lateral selection, an optimistic upper bound);
    otherwise the candidate with the smallest |a - clip(a)| on the main feeder path is preferred."""
    V1, V2, V0, Vpre = meas['Vs']
    Is = meas['Is']; Ipre = meas['Ipre']
    k0 = 2.0
    preds = []
    for b, ft in enumerate(ftypes):
        v1, v2, v0 = V1[b, 0], V2[b, 0], V0[b, 0]
        i1, i2, i0 = Is[b]
        ip = Ipre[b]
        if ft == 'AG':
            V = v1 + v2 + v0; I = i1 + i2 + i0 + k0 * i0; dI = (i1 - ip) + i2 + i0
        elif ft in ('BC', 'BCG'):
            V = v1 - v2; I = i1 - i2; dI = (i1 - ip) - i2
        else:
            V = v1; I = i1; dI = i1 - ip
        cands = []
        for k in range(NL):
            u = sim.FROM[k]; z = sim.Z1_LINE[k]
            num = ((V - CUMZ[u] * I) * np.conj(dI)).imag
            den = (z * I * np.conj(dI)).imag
            a = num / den if den != 0 else -1.0
            cands.append((abs(a - np.clip(a, 0, 1)), k, float(np.clip(a, 0, 1))))
        # valid candidates: inside their section; if none, the closest one overall
        valid = [c for c in cands if c[0] < 1e-9] or [min(cands)]
        if truth is not None:
            tl, ta = truth[0][b], truth[1][b]
            c = min(valid, key=lambda c: sim.point_distance(c[1], c[2], tl, ta))
        else:
            c = min(valid, key=lambda c: (c[1] > 16, c[0]))
        preds.append((c[1], c[2]))
    return preds


# ---------------- rebuild static graph quantities after a topology change ----------------
_ZE_MEAN0 = float(ZE.mean())
_LF_MEAN0 = np.stack([sim.BASE_LINES[:, 2], sim.BASE_LINES[:, 3],
                      np.abs(sim.BASE_LINES[:, 2] + 1j * sim.BASE_LINES[:, 3]) / sim.OHM_PER_KM], 1).mean(0)


def rebuild():
    global A, A_HAT, LAPPE, Y1N, Y0N, Z1N, ZE, ZE_N, YT, INC, LINE_FEATS, CUMZ, NL
    NL = sim.NL
    A = np.eye(N); A[sim.FROM, sim.TO] = 1; A[sim.TO, sim.FROM] = 1
    dg = A.sum(1)
    A_HAT = (A / np.sqrt(dg)[:, None] / np.sqrt(dg)[None, :]).astype(np.float32)
    Lp = np.diag(A.sum(1) - 1) - (A - np.eye(N))
    _, ev = np.linalg.eigh(Lp)
    LAPPE = ev[:, 1:9].astype(np.float32)
    Y1N, Y0N = nominal_Y()
    Z1N = np.linalg.inv(Y1N)
    ZE = np.abs(np.diag(Z1N)[:, None] + np.diag(Z1N)[None, :] - 2 * Z1N)
    ZE_N = (ZE / _ZE_MEAN0).astype(np.float32)
    YT = []
    for Y in (Y1N, Y1N, Y0N):
        Yt = Y / np.diag(Y)[:, None]
        YT.append((Yt.real.astype(np.float32), Yt.imag.astype(np.float32)))
    INC = np.zeros((NL, N), np.float32); INC[np.arange(NL), sim.FROM] = 1; INC[np.arange(NL), sim.TO] = 1
    LF = np.stack([sim.LINES[:, 2], sim.LINES[:, 3], sim.LEN_KM], 1)
    LINE_FEATS = (LF / _LF_MEAN0).astype(np.float32)
    CUMZ = _paths()
