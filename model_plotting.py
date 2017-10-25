import matplotlib.pyplot as plt
from nelpy.analysis.hmm_sparsity import HMMSurrogate
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import nelpy as nel
import nelpy.plotting as npl
import nelpy.plotting.graph as npx
import seaborn as sns
from nelpy import hmmutils

def score_bottleneck_ratio(transmat, n_samples=50000, verbose=False):
    from nelpy.analysis.ergodic import steady_state
    def Qij(i, j, P, pi):
        return pi[i] * P[i,j]

    def QAB(A, B, P, pi):
        sumQ = 0
        for i in A:
            for j in B:
                sumQ += Qij(i, j, P, pi)
        return sumQ

    def complement(S, Omega):
        return Omega - S

    def Pi(S, pi):
        sumS = 0
        for i in S:
            sumS += pi[i]
        return sumS

    def Phi(S, P, pi, Omega):
        Sc = complement(S, Omega)
        return QAB(S, Sc, P, pi) / Pi(S, pi)

    P = transmat
    num_states = transmat.shape[0]
    Omega = set(range(num_states))
    pi_ = steady_state(P).real

    min_Phi = 1
    for nn in range(n_samples):
        n_samp_in_subset = np.random.randint(1, num_states-1)
        S = set(np.random.choice(num_states, n_samp_in_subset, replace=False))
        while Pi(S, pi_) > 0.5:
            n_samp_in_subset -=1
            if n_samp_in_subset < 1:
                n_samp_in_subset = 1
            S = set(np.random.choice(num_states, n_samp_in_subset, replace=False))
        candidate_Phi = Phi(S, P, pi_, Omega)
        if candidate_Phi < min_Phi:
            min_Phi = candidate_Phi
            if verbose:
                print("{}: {} (|S| = {})".format(nn, min_Phi, len(S)))
    return min_Phi

import numpy.linalg as LA

def spectral_gap(transmat):
    evals = LA.eigvals(transmat)
    sorder = np.argsort(np.abs(evals))
    gap = np.real(evals[sorder[-1]] - np.abs(evals[sorder[-2]]))
    return gap


class ColorBarLocator(object):
    def __init__(self, pax, pad=5, width=10):
        self.pax = pax
        self.pad = pad
        self.width = width

    def __call__(self, ax, renderer):
        x, y, w, h = self.pax.get_position().bounds
        fig = self.pax.get_figure()
        inv_trans = fig.transFigure.inverted()
        pad, _ = inv_trans.transform([self.pad, 0])
        width, _ = inv_trans.transform([self.width, 0])
        return [x+w+pad, y, width, h]

def plot_transmat(ax, fig, hmm, edge_threshold=0.0, title='', cbar=True, ylabel=True, **fig_kws):
    cmap = fig_kws.get('cmap', plt.cm.viridis)
    
    num_states = hmm.hmm.n_components
    
    img = ax.matshow(np.where(hmm.hmm.transmat>edge_threshold, hmm.hmm.transmat, 0), cmap=cmap, vmin=0, vmax=1, interpolation='none', aspect='equal')
    ax.set_aspect('equal')
    
    if cbar:
        divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size=0.1, pad=0.1)
        cax = fig.add_axes([0,0,0,0], axes_locator=ColorBarLocator(ax))
        cb=plt.colorbar(img, cax=cax)
        cb.set_label('probability', labelpad=-8)
        cb.set_ticks([0,1])
        npl.utils.no_ticks(cax)
        
#     if not cbar:
#         cax.set_visible(False)
    if ylabel:
        ax.set_yticks([0.5, num_states-1.5])
        ax.set_yticklabels(['1', str(num_states)])    
        ax.set_ylabel('state $i$', labelpad=-16)
    else:
        ax.set_yticks([])
        ax.set_yticklabels('')
    
    ax.set_xticks([0.5, num_states-1.5])
    ax.set_xticklabels(['1', str(num_states)])
    ax.set_xbound(lower=0.0, upper=num_states-1)
    ax.set_ybound(lower=0.0, upper=num_states-1)
    
    ax.set_xlabel('state $j$', labelpad=-16)
    
    ax.set_title(title + ' A')
    sns.despine(ax=ax)    
    
def plot_lambda(ax, fig, hmm, cbar=True, ylabel=True, title='', lo=None, **fig_kws):
    import matplotlib.colors as colors

    cmap = fig_kws.get('cmap', plt.cm.viridis)
    norm = fig_kws.get('norm', colors.LogNorm())
    cb_ticks = fig_kws.get('cb_ticks')
    
    num_states = hmm.hmm.n_components
    num_units = hmm.hmm.n_features
    
    ax.set_aspect(num_states/num_units)

    
    if lo is not None:
        img = ax.matshow(hmm.hmm.means[:,lo].T, cmap=cmap, norm=norm, interpolation='none', aspect='auto')
    else:
        img = ax.matshow(hmm.hmm.means.T, cmap=cmap, norm=norm, interpolation='none', aspect='auto')
    
    if cbar:
        divider = make_axes_locatable(ax)
        cax = fig.add_axes([0,0,0,0], axes_locator=ColorBarLocator(ax))
        # cax = divider.append_axes("right", size=0.1, pad=0.1)
        cb=plt.colorbar(img, cax=cax)
        #cb.set_label('firing rate', labelpad=-8)
        cb.set_ticks(cb_ticks)
        #cb.set_ticklabels(['lo', 'hi'])
        npl.utils.no_ticks(cax)
    
    if ylabel:
        ax.set_yticks([0.5, num_units-1.5])
        ax.set_yticklabels(['1', str(num_units)])
        ax.set_ylabel('unit', labelpad=-16)
    else:
        ax.set_yticks([])
        ax.set_yticklabels('')
        
    ax.set_xticks([0.5, num_states-1.5])
    ax.set_xticklabels(['1', str(num_states)])    
    
    ax.set_ybound(lower=0.0, upper=num_units-1)
    ax.set_xbound(lower=0.0, upper=num_states-1)
    
    ax.set_xlabel('state', labelpad=-16)
    ax.set_title(title + ' $\Lambda$')
    sns.despine(ax=ax)   
    
def plot_sun_graph(ax, hmm, edge_threshold=0.0, lw=2, ec='k', nc='k', node_size=3, **fig_kws):
    plt.sca(ax)
    
    Gi = npx.inner_graph_from_transmat(hmm.hmm.transmat)
    Go = npx.outer_graph_from_transmat(hmm.hmm.transmat)
    
    npx.draw_transmat_graph_inner(Gi, edge_threshold=edge_threshold, lw=lw, ec=ec, node_size=node_size)
    npx.draw_transmat_graph_outer(Go, Gi, edge_threshold=edge_threshold, lw=lw, ec=ec, nc=nc, node_size=node_size*2)

    ax.set_xlim(-1.4,1.4)
    ax.set_ylim(-1.4,1.4)
#     ax0, img = npl.imagesc(hmm.transmat, ax=axes[0])
    npl.utils.clear_left_right(ax)
    npl.utils.clear_top_bottom(ax)
    
#     ax.set_title('1 - $|\lambda_2| =$ {0:.2f}'.format(float(spectral_gap(hmm.hmm.transmat))))
    ax.set_title('$\gamma^*=$ {0:.3f}'.format(float(spectral_gap(hmm.hmm.transmat))), y=1.02)
    
    ax.set_aspect('equal')
    
def plot_connectivity_graph(ax, hmm, edge_threshold=0.0, lw=2, ec='k', node_size=3, **fig_kws):
    plt.sca(ax)
    
    G = npx.graph_from_transmat(hmm.hmm.transmat)
    
    npx.draw_transmat_graph(G, edge_threshold=edge_threshold, lw=lw, ec=ec, node_size=node_size)
#     ax.set_xlim(-1.3,1.3)
#     ax.set_ylim(-1.3,1.3)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
#     ax0, img = npl.imagesc(hmm.transmat, ax=axes[0])
    npl.utils.clear_left_right(ax)
    npl.utils.clear_top_bottom(ax)
    ax1.set_aspect('equal')
    
def plot_transmat_gini_departure(ax, hmms, n_max=500, **fig_kws):
    
    hist_kws={"range": (0.5, 1)}
    bins=50

    with sns.color_palette("Blues_d", 8):
        for hmm in hmms:
            data = np.array(hmm.results['gini_tmat_departure'])
            data = data[:n_max,:]
            sns.distplot(data.sum(axis=0)/len(data), hist=False, hist_kws=hist_kws, bins=bins, label=hmm.label, ax=ax)

    ax.set_title('tmat gini departure, N=50')
    
    ax.set_xlim(0.6, 1)
    
    sns.despine(ax=ax)
    
def plot_transmat_gini_arrival(ax, hmms, n_max=500, **fig_kws):
    
    hist_kws={"range": (0.8, 1)}
    bins=50

    with sns.color_palette("Blues_d", 8):
        for hmm in hmms:
            data = np.array(hmm.results['gini_tmat_arrival'])
            data = data[:n_max,:]
            sns.distplot(data.sum(axis=0)/len(data), hist=False, hist_kws=hist_kws, bins=bins, label=hmm.label, ax=ax)

    ax.set_title('tmat gini arrival, N=250')
    ax.legend('')
    ax.set_xlim(0.7, 1)
    
    sns.despine(ax=ax)
    
def plot_bottleneck(ax, hmms, n_max=500, **fig_kws):
    
    hist_kws={"range": (0, 0.5)}
    bins=50

    for hmm in hmms:
        data = np.array(hmm.results['bottleneck'])
        data = data[:n_max]
        sns.distplot(data, hist=False, hist_kws=hist_kws, bins=bins, label=hmm.label, ax=ax)

    ax.set_title('bottleneck, N=250')
    
    ax.legend('')
    
    ax.set_xlim(0, 0.5)
    
    sns.despine(ax=ax)
    
def plot_gini_lambda(ax, hmms, n_max=500, **fig_kws):
    
    hist_kws={"range": (0.7, 0.9)}
    bins=50

    for hmm in hmms:
        data = np.array(hmm.results['gini_lambda'])
        data = data[:n_max]
        sns.distplot(data, hist=False, hist_kws=hist_kws, bins=bins, label=hmm.label, ax=ax)

    ax.set_title('lambda gini, N=250')
    ax.legend('')
    ax.set_xlim(0.7, 1)
    
    sns.despine(ax=ax)
    
def plot_lambda_gini_across_states(ax, hmms, n_max=5000, **fig_kws):
    
    hist_kws={"range": (0.0, 1)}
    bins=30

    for hmm in hmms:
        data = np.array(hmm.results['gini_lambda_across_states'])
        data = data[:n_max,:]
        sns.distplot(data.sum(axis=0)/len(data), hist_kws=hist_kws, bins=bins, hist=False, kde=True, label=hmm.label, ax=ax, kde_kws={'bw':0.05})
    
    ax.set_title('lambda gini across states, N=50')
    ax.legend('')
    ax.set_xlim(0., 1)
    
    sns.despine(ax=ax)


