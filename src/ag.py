"""Minimal reverse-mode autodiff on numpy (float32) sufficient for the models in this study."""
import numpy as np, sys
sys.setrecursionlimit(10000)

DT = np.float32


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    def __init__(self, v, parents=(), back=None, req=False):
        self.v = np.asarray(v, DT)
        self.g = None
        self.p = parents
        self.back = back
        self.req = req or any(q.req for q in parents)

    # ---- graph ops ----
    def __add__(a, b):
        b = b if isinstance(b, T) else T(b)
        def bk(g):
            return [_unb(g, a.v.shape), _unb(g, b.v.shape)]
        return T(a.v + b.v, (a, b), bk)
    __radd__ = __add__

    def __sub__(a, b):
        b = b if isinstance(b, T) else T(b)
        return a + (b * -1.0)

    def __mul__(a, b):
        b = b if isinstance(b, T) else T(b)
        def bk(g):
            return [_unb(g * b.v, a.v.shape), _unb(g * a.v, b.v.shape)]
        return T(a.v * b.v, (a, b), bk)
    __rmul__ = __mul__

    def __matmul__(a, b):
        def bk(g):
            ga = g @ np.swapaxes(b.v, -1, -2)
            gb = np.swapaxes(a.v, -1, -2) @ g
            return [_unb(ga, a.v.shape), _unb(gb, b.v.shape)]
        return T(a.v @ b.v, (a, b), bk)

    def relu(a):
        m = a.v > 0
        return T(a.v * m, (a,), lambda g: [g * m])

    def sigmoid(a):
        s = 1 / (1 + np.exp(-a.v))
        return T(s, (a,), lambda g: [g * s * (1 - s)])

    def softplus(a):
        s = 1 / (1 + np.exp(-a.v))
        return T(np.logaddexp(0, a.v), (a,), lambda g: [g * s])

    def square(a):
        return T(a.v ** 2, (a,), lambda g: [2 * g * a.v])

    def sum(a, axis=None, keepdims=False):
        def bk(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return [np.broadcast_to(g, a.v.shape).astype(DT)]
        return T(a.v.sum(axis=axis, keepdims=keepdims), (a,), bk)

    def mean(a, axis=None, keepdims=False):
        n = a.v.size if axis is None else a.v.shape[axis]
        return a.sum(axis, keepdims) * (1.0 / n)

    def reshape(a, *s):
        return T(a.v.reshape(*s), (a,), lambda g: [g.reshape(a.v.shape)])

    def transpose(a, *ax):
        inv = np.argsort(ax)
        return T(a.v.transpose(*ax), (a,), lambda g: [g.transpose(*inv)])

    def take(a, idx, axis):
        def bk(g):
            out = np.zeros_like(a.v)
            sl = [slice(None)] * a.v.ndim
            # scatter-add along axis
            np.add.at(np.moveaxis(out, axis, 0), idx, np.moveaxis(g, axis, 0))
            return [out]
        return T(np.take(a.v, idx, axis=axis), (a,), bk)

    def softmax(a, axis=-1):
        e = np.exp(a.v - a.v.max(axis, keepdims=True))
        s = e / e.sum(axis, keepdims=True)
        def bk(g):
            return [s * (g - (g * s).sum(axis, keepdims=True))]
        return T(s, (a,), bk)

    def log_softmax(a, axis=-1):
        m = a.v.max(axis, keepdims=True)
        lse = np.log(np.exp(a.v - m).sum(axis, keepdims=True)) + m
        out = a.v - lse
        s = np.exp(out)
        return T(out, (a,), lambda g: [g - s * g.sum(axis, keepdims=True)])

    def layernorm(a, eps=1e-5):
        mu = a.v.mean(-1, keepdims=True)
        xc = a.v - mu
        var = (xc ** 2).mean(-1, keepdims=True)
        inv = 1 / np.sqrt(var + eps)
        y = xc * inv
        n = a.v.shape[-1]
        def bk(g):
            return [inv / n * (n * g - g.sum(-1, keepdims=True) - y * (g * y).sum(-1, keepdims=True))]
        return T(y, (a,), bk)

    # ---- backprop ----
    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            t, done = stack.pop()
            if done:
                order.append(t); continue
            if id(t) in seen or not t.req:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for q in t.p:
                if id(q) not in seen and q.req:
                    stack.append((q, False))
        self.g = np.ones_like(self.v)
        for t in reversed(order):
            if t.back is None or t.g is None:
                continue
            gs = t.back(t.g)
            for q, gq in zip(t.p, gs):
                if q.req:
                    q.g = gq.astype(DT) if q.g is None else q.g + gq
            if not isinstance(t, Param):
                t.g = None; t.p = (); t.back = None   # free the graph eagerly


def concat(ts, axis):
    sizes = [t.v.shape[axis] for t in ts]
    cut = np.cumsum(sizes)[:-1]
    def bk(g):
        return np.split(g, cut, axis=axis)
    return T(np.concatenate([t.v for t in ts], axis), tuple(ts), bk)


class Param(T):
    def __init__(self, v):
        super().__init__(v, req=True)


def glorot(rng, i, o):
    return Param(rng.normal(0, np.sqrt(2.0 / (i + o)), (i, o)))


class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-5, b1=0.9, b2=0.999):
        self.ps = params; self.lr = lr; self.wd = wd; self.b1 = b1; self.b2 = b2; self.t = 0
        self.m = [np.zeros_like(p.v) for p in params]; self.s = [np.zeros_like(p.v) for p in params]

    def step(self, clip=5.0):
        self.t += 1
        gn = np.sqrt(sum(float((p.g ** 2).sum()) for p in self.ps if p.g is not None))
        sc = min(1.0, clip / (gn + 1e-8))
        for p, m, s in zip(self.ps, self.m, self.s):
            if p.g is None:
                continue
            g = p.g * sc + self.wd * p.v
            m *= self.b1; m += (1 - self.b1) * g
            s *= self.b2; s += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t); sh = s / (1 - self.b2 ** self.t)
            p.v -= (self.lr * mh / (np.sqrt(sh) + 1e-8)).astype(DT)
            p.g = None
