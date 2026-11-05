"""Topology library: random radial reconfigurations of the IEEE 33-bus feeder and a cache that
swaps every topology-dependent quantity (simulator + model statics) in one call."""
import numpy as np, networkx as nx
import sim, models

ALL_BRANCHES = np.vstack([sim.BASE_LINES, sim.TIES])          # 32 lines + 5 ties = 37 branches


def oriented(edges):
    G = nx.Graph()
    for l in edges:
        G.add_edge(int(l[0]), int(l[1]), r=float(l[2]), x=float(l[3]))
    assert nx.is_tree(G) and G.number_of_nodes() == 33
    return np.array([(u, v, G[u][v]['r'], G[u][v]['x']) for u, v in nx.bfs_edges(G, 1)])


def key(lines):
    return tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in lines[:, :2]))


def random_tree(rng):
    """Random radial configuration: random-weight spanning tree of the 37-branch meshed graph."""
    G = nx.Graph()
    for i, l in enumerate(ALL_BRANCHES):
        G.add_edge(int(l[0]), int(l[1]), w=rng.random(), i=i)
    T = nx.minimum_spanning_tree(G, weight='w')
    return oriented([ALL_BRANCHES[d['i']] for _, _, d in T.edges(data=True)])


def library(seed=2026):
    rng = np.random.default_rng(seed)
    base = oriented(sim.BASE_LINES)
    bw = sim.reconfigured_lines()
    seen = {key(base), key(bw)}
    rand = []
    while len(rand) < 14:
        t = random_tree(rng)
        if key(t) not in seen:
            seen.add(key(t)); rand.append(t)
    train = [base] + rand[:11]                 # 12 training topologies (incl. the base case)
    test = [bw] + rand[11:14]                  # 4 unseen topologies (Baran-Wu + 3 random)
    return train, test


_CACHE = {}
_SIM_ATTRS = ['LINES', 'FROM', 'TO', 'NL', 'Z1_LINE', 'Z0_LINE', 'LEN_KM', 'D_NODE']
_MOD_ATTRS = ['A', 'A_HAT', 'LAPPE', 'Y1N', 'Y0N', 'Z1N', 'ZE', 'ZE_N', 'YT', 'INC', 'LINE_FEATS', 'CUMZ', 'NL']


def use(lines):
    k = tuple(tuple(int(x) for x in l[:2]) for l in lines)   # order-sensitive: section indices matter
    if k not in _CACHE:
        sim.set_topology(lines); models.rebuild()
        _CACHE[k] = ({a: getattr(sim, a) for a in _SIM_ATTRS}, {a: getattr(models, a) for a in _MOD_ATTRS})
    s, m = _CACHE[k]
    for a, v in s.items():
        setattr(sim, a, v)
    for a, v in m.items():
        setattr(models, a, v)
