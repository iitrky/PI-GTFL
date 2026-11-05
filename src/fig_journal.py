"""Journal figures (Electrical Engineering, Springer): Fig1-Fig5, 174 mm wide, 600 dpi, sans-serif lettering,
no titles inside figures, patterns in addition to colour."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sim

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'hatch.linewidth': 0.6})
W = 174 / 25.4
INK = '#1f2933'; BLUE = '#1d4ed8'; TEAL = '#0f766e'; ORANGE = '#b45309'; GREY = '#4b5563'; RED = '#b91c1c'
DPI = 600


def save(fig, name):
    fig.savefig(f'{name}.png', dpi=DPI)
    fig.savefig(f'{name}.tiff', dpi=DPI, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def box(ax, x, y, w, h, title, lines, color, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.06', fc='white', ec=color, lw=1.1))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.36), w, 0.36, boxstyle='round,pad=0.01,rounding_size=0.06', fc=color, ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 0.18, title, ha='center', va='center', color='white', weight='bold', fontsize=7)
    for i, l in enumerate(lines):
        ax.text(x + 0.08, y + h - 0.58 - i * 0.25, l, ha='left', va='center', color=INK, fontsize=fs)


def arrow(ax, p, q, color=INK, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=1.0, ls=ls))


def fig1():
    fig, ax = plt.subplots(figsize=(W, W * 0.42))
    ax.set_xlim(0, 15.6); ax.set_ylim(0.6, 7.2); ax.axis('off')
    box(ax, 0.1, 4.9, 2.9, 2.2, '1  Field layer', ['Feeder with current-limited', 'grid-following IBRs', 'µPMUs on ~30 % of buses',
                                                 'Substation V and I', 'C37.118 streams to PDC', 'Switch status (SCADA)'], GREY)
    box(ax, 3.3, 4.9, 2.9, 2.2, '2  Feature builder', ['Pre-fault and fault frames', 'ΔV1, V2, V0 (Re, Im)', 'Scale-normalised copy',
                                                     'Feeder-head ΔI1, I2, I0', 'Availability mask', 'X  (N × 20)'], BLUE)
    box(ax, 6.5, 4.9, 3.0, 2.2, '3  Topology refresh', ['Current switching state', 'Adjacency A, Laplacian PE', 'Z-bus → distance Ze',
                                                      'Line R, X, length', 'Normalised Y0, Y1, Y2', 'Recomputed per state'], BLUE)
    box(ax, 9.8, 4.9, 5.7, 2.2, '4  PI-GTFL encoder (Fig. 2)', ['3 layers of impedance-biased attention', 'Edge head: section logits and distance α',
                                                               'Graph head: fault type', 'Node head: ΔV at all buses',
                                                               'One forward pass, about 1 ms', ''], TEAL)
    box(ax, 9.8, 0.7, 5.7, 2.6, '5  Training (offline, Algorithm 1)', ['12 randomised radial topologies', 'Noise and µPMU-drop augmentation',
                                                                     'Loss = CE + distance + type', '  + measurement fit', '  + fault-aware KCL residual',
                                                                     'Calibration set for Algorithm 2', ''], ORANGE)
    box(ax, 4.7, 0.7, 4.8, 2.6, '6  Calibrated decision (Algorithm 2)', ['Posterior over sections', 'Threshold q from calibration set',
                                                                         'Set S = {k : p_k ≥ 1 − q}', '|S| = 1 → single section + distance',
                                                                         '|S| > 1 → ranked patrol list', 'Target miss rate α (e.g. 2 %)', ''], RED)
    box(ax, 0.1, 0.7, 4.3, 2.6, '7  Protection and FLISR', ['Trip and isolation commands', '(IEC 61850 GOOSE or MMS)', 'Restoration switching',
                                                          'Crew dispatch with', 'distance from upstream bus', 'HIL validation path', ''], GREY)
    for a, b in [((3.0, 6.0), (3.3, 6.0)), ((6.2, 6.0), (6.5, 6.0)), ((9.5, 6.0), (9.8, 6.0))]:
        arrow(ax, a, b)
    arrow(ax, (12.65, 4.9), (12.65, 3.3), ORANGE, '<|-|>')
    ax.text(12.8, 4.1, 'back-propagation', fontsize=6.2, color=ORANGE, va='center')
    arrow(ax, (9.8, 5.2), (9.5, 3.0))
    arrow(ax, (4.7, 2.0), (4.4, 2.0))
    arrow(ax, (1.7, 3.3), (1.7, 4.9), GREY, '-|>', '--')
    ax.text(1.85, 4.1, 'switching events', fontsize=6.2, color=GREY, va='center')
    fig.tight_layout(pad=0.2); save(fig, 'Fig1')


def fig2():
    fig, ax = plt.subplots(figsize=(W, W * 0.42))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.6); ax.axis('off')

    def b(x, y, w, h, t, c, fs=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=c, lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK)
    b(0.1, 2.7, 1.5, 1.2, 'Node\nembeddings\nH (N × d)', GREY)
    b(2.1, 2.7, 1.2, 1.2, 'Layer\nnorm', BLUE)
    b(3.8, 4.4, 1.6, 0.8, 'Q = W_Q h', BLUE); b(3.8, 2.9, 1.6, 0.8, 'K = W_K h', BLUE); b(3.8, 1.55, 1.6, 0.75, 'V = W_V h', BLUE)
    b(6.0, 3.6, 1.9, 1.2, 'Scores\nQKᵀ / √dk', TEAL)
    b(6.0, 5.3, 1.9, 1.1, 'Topology prior\n−γh Ze + βh A', ORANGE)
    b(8.4, 3.6, 1.4, 1.2, 'Softmax\n(per head)', TEAL)
    b(10.3, 2.7, 1.6, 1.2, 'Weighted sum\nof V, concat\n4 heads, W_O', TEAL)
    b(12.4, 2.7, 1.4, 1.2, 'Residual\n+ LN + FFN\n+ residual', BLUE)
    b(14.3, 4.6, 1.6, 1.1, 'Edge head\nsection, α', RED, 6.4)
    b(14.3, 3.0, 1.6, 1.1, 'Graph head\nfault type', RED, 6.4)
    b(14.3, 1.4, 1.6, 1.1, 'Node head\nΔV at all buses', RED, 6.4)
    b(10.3, 0.1, 3.5, 1.0, 'KCL residual of ΔV  (training only)', ORANGE, 6.4)
    arrow(ax, (1.6, 3.3), (2.1, 3.3))
    for y in (4.8, 3.3, 1.8):
        arrow(ax, (3.3, 3.3), (3.8, y))
    arrow(ax, (5.4, 4.8), (6.0, 4.4)); arrow(ax, (5.4, 3.3), (6.0, 3.9))
    arrow(ax, (6.95, 5.3), (6.95, 4.8), ORANGE)
    ax.text(7.05, 5.05, 'add', fontsize=6, color=ORANGE, va='center')
    arrow(ax, (7.9, 4.2), (8.4, 4.2)); arrow(ax, (9.8, 4.2), (10.6, 3.9)); arrow(ax, (5.4, 1.8), (10.6, 2.7))
    arrow(ax, (11.9, 3.3), (12.4, 3.3))
    ax.annotate('', xy=(2.7, 1.25), xytext=(13.1, 1.25), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.9, ls='--',
                                                                          connectionstyle='arc3,rad=0'))
    ax.text(7.6, 1.0, 'repeat × 3 layers', fontsize=6.2, color=GREY, ha='center')
    for y in (5.15, 3.55, 1.95):
        arrow(ax, (13.8, 3.3), (14.3, y))
    arrow(ax, (15.1, 1.4), (13.8, 0.6), ORANGE)
    fig.tight_layout(pad=0.2); save(fig, 'Fig2')


def fig3():
    pos = {}
    for b_ in range(1, 19): pos[b_] = (b_ - 1, 0)
    for i, b_ in enumerate(range(19, 23)): pos[b_] = (1 + i, 1.4)
    for i, b_ in enumerate(range(23, 26)): pos[b_] = (2 + i, -1.4)
    for i, b_ in enumerate(range(26, 34)): pos[b_] = (5 + i, -2.8)
    fig, ax = plt.subplots(figsize=(W, W * 0.33))
    for k in range(32):
        a, c = int(sim.BASE_LINES[k, 0]), int(sim.BASE_LINES[k, 1])
        (x1, y1), (x2, y2) = pos[a], pos[c]
        ax.plot([x1, x1, x2] if y1 != y2 else [x1, x2], [y1, y2, y2] if y1 != y2 else [y1, y2], color=GREY, lw=1.0, zorder=1)
    for a, c, _, _ in sim.TIES:
        (x1, y1), (x2, y2) = pos[int(a)], pos[int(c)]
        ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.8, ls=':', zorder=0)
    pm = set(sim.PMU_BUSES + 1); ib = set(sim.IBR_BUSES + 1)
    for b_, (x, y) in pos.items():
        ax.scatter(x, y, s=42, c='#1d4ed8' if b_ in pm else 'white', edgecolors=INK, zorder=3, lw=0.8,
                   marker='o')
        ax.text(x + (0.25 if b_ in (23, 26) else 0), y + 0.32, str(b_), ha='center', fontsize=6)
        if b_ in ib:
            ax.scatter(x, y - 0.42, marker='s', s=26, c=ORANGE, zorder=3)
    ax.scatter([-0.6], [0], marker='>', s=90, c=INK); ax.text(-0.9, -0.55, 'Substation\n12.66 kV', fontsize=6, ha='center')
    ax.scatter([], [], s=42, c='#1d4ed8', edgecolors=INK, label='µPMU bus (10)')
    ax.scatter([], [], s=42, c='white', edgecolors=INK, label='Unmonitored bus')
    ax.scatter([], [], marker='s', s=26, c=ORANGE, label='Grid-following IBR (6)')
    ax.plot([], [], color=GREY, ls=':', label='Normally open tie (5)')
    ax.legend(loc='lower left', frameon=False, fontsize=6.5, ncol=4, bbox_to_anchor=(0, -0.02))
    ax.set_xlim(-1.5, 18); ax.set_ylim(-3.9, 2.0); ax.axis('off')
    fig.tight_layout(pad=0.2); save(fig, 'Fig3')


R = json.load(open('results.json'))
STY = {'pigtfl': ('PI-GTFL (proposed)', '#b45309', 'o', '-'), 'gt_vanilla': ('Vanilla graph transformer', '#4b5563', 'v', ':'),
       'cnn': ('CNN', '#0f766e', '^', '--'), 'gcn': ('GCN', '#6d28d9', 'D', '-.')}


def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.36))
    for ax, tags, xs, xl, lab in [(axs[0], ['tve0.0', 'main', 'tve0.01', 'tve0.02', 'tve0.03'], [0, 0.5, 1, 2, 3], 'µPMU noise, TVE (%)', '(a)'),
                                  (axs[1], ['main', 'drop1', 'drop2', 'drop3', 'drop4', 'drop5'], [0, 1, 2, 3, 4, 5], 'Missing µPMUs (of 9 non-substation units)', '(b)')]:
        for v, (n, c, mk, ls) in STY.items():
            y = np.array([R[t][v]['sec_acc'] for t in tags]) * 100
            ax.plot(xs, y, color=c, marker=mk, ls=ls, lw=1.3, ms=4, label=n)
            sd = np.array([R[t][v]['std_sec_acc'] for t in tags]) * 100
            ax.fill_between(xs, y - sd, y + sd, color=c, alpha=0.15, lw=0)
        ax.set_xlabel(xl); ax.set_ylabel('Section accuracy (%)'); ax.grid(alpha=0.25)
        ax.text(-0.14, 1.02, lab, transform=ax.transAxes, fontsize=8, weight='bold')
    axs[0].axvspan(1, 3.1, color='#f3f4f6', zorder=0); axs[1].axvspan(3, 5.1, color='#f3f4f6', zorder=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.3); save(fig, 'Fig4')


def fig5():
    T = json.load(open('results_topo_train.json'))
    tops = ['Baran-Wu (unseen)', 'Random A (unseen)', 'Random B (unseen)', 'Random C (unseen)']
    M = [('pigtfl (multi-topo)', 'PI-GTFL', '#b45309', ''), ('gt_nophys (multi-topo)', 'Without KCL loss', '#1d4ed8', '////'),
         ('gt_vanilla (multi-topo)', 'Vanilla graph transformer', '#4b5563', '....'), ('gt_nobias (multi-topo)', 'Without impedance bias', '#9ca3af', 'xxxx'),
         ('gcn (multi-topo)', 'GCN', '#6d28d9', '\\\\\\\\'), ('impedance (oracle)', 'Impedance, oracle lateral', '#111827', '----'),
         ('pigtfl (base only)', 'PI-GTFL, base topology only', '#fcd34d', 'oo')]
    fig, ax = plt.subplots(figsize=(W, W * 0.36)); w = 0.115
    for i, (k, lab, c, h) in enumerate(M):
        y = [T['per_topology'][t]['all'][k]['sec_acc'] * 100 for t in tops] + [T['unseen_mean']['all'][k]['sec_acc'] * 100]
        ax.bar(np.arange(5) + (i - 3) * w, y, w, color=c, hatch=h, edgecolor='white' if c not in ('#fcd34d', '#9ca3af') else INK,
               lw=0.4, label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels(['Baran–Wu', 'Random A', 'Random B', 'Random C', 'Mean of four'])
    ax.set_ylabel('Section accuracy (%)'); ax.set_xlabel('Unseen feeder topology (four tie switches closed in each)')
    ax.axvline(3.5, color='#d1d5db', lw=0.8); ax.grid(axis='y', alpha=0.25); ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=6.3, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(pad=0.3); save(fig, 'Fig5')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); print('done')
