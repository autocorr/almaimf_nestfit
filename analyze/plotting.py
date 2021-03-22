#!/usr/bin/env python3

import sys
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import scipy as sp
from skimage import morphology
import matplotlib as mpl
from matplotlib import (patheffects, colors)
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import aplpy
import radio_beam
from astropy.io import fits
from astropy import units as u
from astropy import (convolution, coordinates, wcs)

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
from nestfit import plotting as nfplot
from nestfit.main import (HdfStore, take_by_components)

from . import (PLOT_PATH, DATA_PATH, RUN_PATH, ALL_TARGETS)
from .core import (get_cubestack, get_store)


# Create a filter to stop that matplotlib deprecation warning
warnings.simplefilter('ignore')
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out')
plt.rc('ytick', direction='out')


CLR_CMAP = plt.cm.Spectral_r
CLR_CMAP.set_bad('0.5', 1.0)
HOT_CMAP = plt.cm.afmhot
HOT_CMAP.set_bad('0.5', 1.0)
RDB_CMAP = plt.cm.RdBu_r
RDB_CMAP.set_bad('0.5', 1.0)
VIR_CMAP = plt.cm.viridis
VIR_CMAP.set_bad('0.5', 1.0)

_cmap_list = [(0.5, 0.5, 0.5, 1.0)] + [plt.cm.plasma(i) for i in range(plt.cm.plasma.N)]
NBD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        'Discrete Plasma', _cmap_list, len(_cmap_list),
)
NBD_CMAP.set_bad('0.2')


##############################################################################
#                       NestFit plotting routines
##############################################################################

def store_to_plotter(store, label=None):
    pad = nfplot.PaddingConfig(
            edge_pads=(0.7, 0.8, 0.6, 0.3),
            sep_pads=(0.2, 0.2),
            cbar_width=0.15,
    )
    if label is None:
        plot_dir = str(PLOT_PATH)
    else:
        plot_dir = str(PLOT_PATH/label)
    spl = nfplot.StorePlotter(store, plot_dir=plot_dir, pad=pad)
    return spl


def plot_test_pix(store, stack, pix=(100, 100)):
    spl = store_to_plotter(store)
    # FIXME These functions are still ammonia specific
    nfplot.plot_amm_specfit(spl, stack, pix, n_model=1, zoom=True)
    nfplot.plot_amm_post_stack(spl, pix, n_model=1)
    nfplot.plot_amm_spec_grid(spl, stack, pix, (10, 10))


def make_all_plots(field, store_suffix):
    stack = get_cubestack(field)
    with get_store(field, store_suffix) as store:
        spl = store_to_plotter(store, label=f'{field}_{store_suffix}')
        nfplot.plot_nbest(spl)
        nfplot.plot_conv_nbest(spl)
        nfplot.plot_deblend_peak(spl)
        nfplot.plot_deblend_intintens(spl, vmax=None)
        nfplot.plot_map_props(spl)
        nfplot.plot_evdiff(spl, conv=True)
        nfplot.plot_quan_props(spl, conv=False)
        nfplot.plot_err_props(spl, conv=False)
        #plot_test_pix(store, stack)


def plot_param_hists(field, store_suffix, outname='param_marg_posteriors'):
    store = get_store(field, store_suffix)
    # Run metadata.
    pdf_bins = store.hdf['/products/pdf_bins'][...]
    n_params = store.hdf.attrs['n_params']
    all_labels = store.hdf.attrs['tex_labels_with_units']
    # Read in posteriors and MAP values from store file.
    # dimensions (b, l)
    nbest = store.hdf['/products/conv_nbest'][...]
    # dimensions (m, p, b, l)
    pmap = store.hdf['/products/nbest_MAP'][...]
    # dimensions (r, m, p, h, b, l)
    post = store.hdf['/products/post_pdfs'][...]
    post = take_by_components(post, nbest, incl_zero=False)  # -> (m, p, h, b, l)
    # Average/summed posterior distibution. Mask positions without
    # detections.
    apdf = np.nansum(post[...,nbest>=0], axis=(0, 3))
    # Begin plotting
    fig, axes = plt.subplots(ncols=1, nrows=n_params, figsize=(4, 1*n_params))
    indices = [1, 2, 3, 0]
    for ii, ax in zip(indices, axes):
        x_label = all_labels[ii]
        bins = pdf_bins[ii]
        # MAP histogram values
        map_vals = pmap[:,ii,:,:]
        hist, _, _ = ax.hist(map_vals.flatten(), bins=bins, density=True,
                color='firebrick', histtype='step', linewidth=1.0, zorder=22)
        # Prior distribution
        #ax.plot(x, dist*hist.max()/dist.max(), 'm-', zorder=21)
        # Average/summed posterior distribution
        ax.fill_between(bins, apdf[ii,:]*hist.max()/apdf.max(), color='0.7',
                step='mid', zorder=0)
        # Labels
        ax.set_ylim(0, hist.max()*1.1)
        ax.set_xlim(bins.min(), bins.max())
        ax.set_xlabel(x_label)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('PDF')
    plt.tight_layout(h_pad=0.5)
    save_figure(f'{field}_{store_suffix}_{outname}')
    store.close()


