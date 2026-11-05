"""Phasor-domain fault simulator for the IEEE 33-bus feeder with current-limited
inverter-based resources (IBRs). Sequence-network formulation.

Conventions: per-unit on 12.66 kV / 10 MVA. Bus indices are 0-based internally
(bus 1 of the paper == index 0).
"""
import numpy as np

V_BASE_KV = 12.66
S_BASE_MVA = 10.0
Z_BASE = V_BASE_KV ** 2 / S_BASE_MVA          # ohm
OHM_PER_KM = 0.45                              # |z| used to convert impedance to length

# Baran & Wu (1989) line data: from, to, R(ohm), X(ohm)
LINES = np.array([
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941), (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351), (8, 9, 1.0300, 0.7400), (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302)])
LOADS_KW = np.array([0, 100, 90, 120, 60, 60, 200, 200, 60, 60, 45, 60, 60, 120, 60, 60, 60, 90,
                     90, 90, 90, 90, 90, 420, 420, 60, 60, 60, 120, 200, 150, 210, 60], float)
LOADS_KVAR = np.array([0, 60, 40, 80, 30, 20, 100, 100, 20, 20, 30, 35, 35, 80, 10, 20, 20, 40,
                       40, 40, 40, 40, 50, 200, 200, 25, 25, 20, 70, 600, 70, 100, 40], float)

N = 33
FROM = LINES[:, 0].astype(int) - 1
TO = LINES[:, 1].astype(int) - 1
NL = len(LINES)
Z1_LINE = (LINES[:, 2] + 1j * LINES[:, 3]) / Z_BASE
Z0_LINE = 3.0 * Z1_LINE
LEN_KM = np.abs(LINES[:, 2] + 1j * LINES[:, 3]) / OHM_PER_KM

IBR_BUSES = np.array([14, 18, 22, 25, 30, 33]) - 1
IBR_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.2, 0.2, 0.15])
PMU_BUSES = np.array([1, 3, 6, 9, 13, 18, 22, 25, 29, 33]) - 1

FAULT_TYPES = ['AG', 'BC', 'BCG', 'ABC']


def ybus(z_line, shunt):
    Y = np.zeros((N + 1, N + 1), complex)  # last row/col reserved for the fault node
    for k in range(NL):
        Y[FROM[k], FROM[k]] += 1 / z_line[k]
        Y[TO[k], TO[k]] += 1 / z_line[k]
        Y[FROM[k], TO[k]] -= 1 / z_line[k]
        Y[TO[k], FROM[k]] -= 1 / z_line[k]
    Y[np.arange(N), np.arange(N)] += shunt
    return Y


def ybus_with_fault(z_line, shunt, line, alpha):
    """Y-bus of size N+1 with the fault node (index N) inserted in `line` at fraction alpha."""
    zl = z_line.copy()
    Y = np.zeros((N + 1, N + 1), complex)
    for k in range(NL):
        i, j = FROM[k], TO[k]
        if k == line:
            za, zb = alpha * zl[k], (1 - alpha) * zl[k]
            for (a, b, z) in ((i, N, za), (N, j, zb)):
                Y[a, a] += 1 / z; Y[b, b] += 1 / z; Y[a, b] -= 1 / z; Y[b, a] -= 1 / z
        else:
            Y[i, i] += 1 / zl[k]; Y[j, j] += 1 / zl[k]; Y[i, j] -= 1 / zl[k]; Y[j, i] -= 1 / zl[k]
    Y[np.arange(N), np.arange(N)] += shunt
    return Y


def ibr_current(V, Ipre_d, rating, k_q, imax_pu):
    """Positive-sequence IBR injection with reactive priority and current limiting."""
    vm = np.abs(V)
    ang = np.exp(1j * np.angle(V))
    imax = imax_pu * rating
    iq = np.clip(k_q * np.maximum(0.9 - vm, 0) / 0.9 * rating * 1.0, 0, imax)
    id_ = np.minimum(Ipre_d, np.sqrt(np.maximum(imax ** 2 - iq ** 2, 0)))
    return (id_ - 1j * iq) * ang


def simulate(rng, line, alpha, ftype, rf_ohm, load_scale, pen, ssc_mva, k_q=2.0, imax_pu=1.2):
    """Return dict with prefault and fault sequence voltages (all buses) and substation currents."""
    rf = rf_ohm / Z_BASE
    S_load = load_scale * (LOADS_KW + 1j * LOADS_KVAR) / 1000 / S_BASE_MVA
    y_load = np.conj(S_load)  # constant impedance at 1 pu
    zs1 = (1.0 / (ssc_mva / S_BASE_MVA)) * (0.1 + 1j) / np.abs(0.1 + 1j)
    zs0 = zs1
    sh1 = y_load.copy().astype(complex); sh1[0] += 1 / zs1
    sh0 = np.zeros(N, complex); sh0[0] += 1 / zs0
    E = 1.0 + 0j
    p_ibr_total = pen * load_scale * LOADS_KW.sum() / 1000 / S_BASE_MVA
    rating = np.zeros(N); rating[IBR_BUSES] = p_ibr_total * IBR_WEIGHTS * 1.1  # 10 % headroom
    p_ibr = np.zeros(N); p_ibr[IBR_BUSES] = p_ibr_total * IBR_WEIGHTS

    # --- prefault (positive sequence only) ---
    Y1 = ybus(Z1_LINE, sh1)[:N, :N]
    Z1 = np.linalg.inv(Y1)
    Inorton = np.zeros(N, complex); Inorton[0] = E / zs1
    V = Z1 @ Inorton
    for _ in range(30):
        Iibr = np.where(rating > 0, np.conj(p_ibr / np.where(V == 0, 1, V)), 0)
        Vn = Z1 @ (Inorton + Iibr)
        if np.max(np.abs(Vn - V)) < 1e-9:
            V = Vn; break
        V = Vn
    Vpre = V
    Ipre_d = np.abs(np.where(rating > 0, p_ibr / np.abs(Vpre), 0))

    # --- faulted networks ---
    shf1 = np.append(sh1, 0); shf0 = np.append(sh0, 0)
    Yf1 = ybus_with_fault(Z1_LINE, sh1, line, alpha)
    Yf0 = ybus_with_fault(Z0_LINE, sh0, line, alpha)
    Zf1 = np.linalg.inv(Yf1); Zf2 = Zf1; Zf0 = np.linalg.inv(Yf0)
    z1, z2, z0 = Zf1[N, N], Zf2[N, N], Zf0[N, N]
    Inor = np.zeros(N + 1, complex); Inor[0] = E / zs1
    Iibr = np.zeros(N + 1, complex); Iibr[:N] = np.where(rating > 0, np.conj(p_ibr / Vpre), 0)
    for it in range(60):
        Voc = Zf1 @ (Inor + Iibr)
        vf = Voc[N]
        if ftype == 'ABC':
            I1 = vf / (z1 + rf); I2 = 0; I0 = 0
        elif ftype == 'AG':
            I1 = vf / (z1 + z2 + z0 + 3 * rf); I2 = I1; I0 = I1
        elif ftype == 'BC':
            I1 = vf / (z1 + z2 + rf); I2 = -I1; I0 = 0
        else:  # BCG
            zg = z0 + 3 * rf
            I1 = vf / (z1 + z2 * zg / (z2 + zg)); I2 = -I1 * zg / (z2 + zg); I0 = -I1 * z2 / (z2 + zg)
        V1 = Voc - Zf1[:, N] * I1
        V2 = -Zf2[:, N] * I2
        V0 = -Zf0[:, N] * I0
        Inew = np.zeros(N + 1, complex)
        Inew[:N] = np.where(rating > 0, ibr_current(V1[:N], Ipre_d, rating, k_q, imax_pu), 0)
        if np.max(np.abs(Inew - Iibr)) < 1e-8:
            Iibr = Inew; break
        Iibr = 0.5 * Iibr + 0.5 * Inew
    # substation feeder currents (into line 1-2), per sequence
    def isub(Vs, zl):
        k0 = 0
        if line == k0:
            return (Vs[0] - Vs[N]) / (alpha * zl[0])
        return (Vs[0] - Vs[1]) / zl[0]
    Isub = np.array([isub(V1, Z1_LINE), isub(V2, Z1_LINE), isub(V0, Z0_LINE)])
    Ipre_sub = (Vpre[0] - Vpre[1]) / Z1_LINE[0]
    return dict(Vpre=Vpre, V1=V1[:N], V2=V2[:N], V0=V0[:N], Isub=Isub, Ipre_sub=Ipre_sub,
                If=np.array([I1, I2, I0]), Iibr=Iibr[:N], rating=rating, vf_pre=Voc[N])


def sample_scenario(rng, pen=None, ftype=None):
    line = rng.integers(NL)
    alpha = rng.uniform(0.05, 0.95)
    if ftype is None:
        ftype = rng.choice(FAULT_TYPES, p=[0.5, 0.15, 0.2, 0.15])
    rf = rng.uniform(0.01, 50.0) if ftype in ('AG', 'BCG') else rng.uniform(0.01, 10.0)
    if pen is None:
        pen = rng.uniform(0.0, 1.0)
    return dict(line=int(line), alpha=float(alpha), ftype=str(ftype), rf_ohm=float(rf),
                load_scale=float(rng.uniform(0.5, 1.2)), pen=float(pen),
                ssc_mva=float(rng.uniform(80, 300)), k_q=float(rng.uniform(1.5, 3.0)),
                imax_pu=float(rng.uniform(1.1, 1.5)))


# ---------------- geometry helpers (km) ----------------
def node_distance_matrix():
    import networkx as nx
    G = nx.Graph()
    for k in range(NL):
        G.add_edge(int(FROM[k]), int(TO[k]), w=LEN_KM[k])
    D = np.zeros((N, N))
    for s, d in nx.all_pairs_dijkstra_path_length(G, weight='w'):
        for t, v in d.items():
            D[s, t] = v
    return D


D_NODE = node_distance_matrix()


def point_distance(l1, a1, l2, a2):
    if l1 == l2:
        return abs(a1 - a2) * LEN_KM[l1]
    e1 = [(FROM[l1], a1 * LEN_KM[l1]), (TO[l1], (1 - a1) * LEN_KM[l1])]
    e2 = [(FROM[l2], a2 * LEN_KM[l2]), (TO[l2], (1 - a2) * LEN_KM[l2])]
    return min(d1 + D_NODE[n1, n2] + d2 for n1, d1 in e1 for n2, d2 in e2)


# ---------------- topology reconfiguration ----------------
TIES = np.array([(8, 21, 2.0, 2.0), (9, 15, 2.0, 2.0), (12, 22, 2.0, 2.0), (18, 33, 0.5, 0.5), (25, 29, 0.5, 0.5)])
BASE_LINES = LINES.copy()


def set_topology(lines):
    """Replace the feeder's section list (array of from, to, R, X) in place for all module users."""
    global LINES, FROM, TO, NL, Z1_LINE, Z0_LINE, LEN_KM, D_NODE
    LINES = np.array(lines, float)
    FROM = LINES[:, 0].astype(int) - 1; TO = LINES[:, 1].astype(int) - 1; NL = len(LINES)
    Z1_LINE = (LINES[:, 2] + 1j * LINES[:, 3]) / Z_BASE; Z0_LINE = 3.0 * Z1_LINE
    LEN_KM = np.abs(LINES[:, 2] + 1j * LINES[:, 3]) / OHM_PER_KM
    D_NODE = node_distance_matrix()


def reconfigured_lines():
    """Baran-Wu loss-minimising configuration: branches 7, 9, 14, 32 opened, ties 8-21, 9-15, 12-22, 18-33 closed."""
    import networkx as nx
    open_pairs = {(7, 8), (9, 10), (14, 15), (32, 33)}
    keep = [l for l in BASE_LINES if (int(l[0]), int(l[1])) not in open_pairs]
    keep += [t for t in TIES[:4]]
    # orient every section away from the substation
    G = nx.Graph()
    for l in keep:
        G.add_edge(int(l[0]), int(l[1]), r=l[2], x=l[3])
    assert nx.is_tree(G) and G.number_of_nodes() == 33
    out = [(u, v, G[u][v]['r'], G[u][v]['x']) for u, v in nx.bfs_edges(G, 1)]
    return np.array(out)
