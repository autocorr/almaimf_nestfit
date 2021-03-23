#!/usr/bin/env python3
"""
====
Core
====
Apply NestFit to test ALMA-IMF N2H+ J=1-0 datasets. This script may be called
from the command line for batch queue processing. Edit the ``__main__`` block.
Post-processing must be parallelized over different sources, so is not
currently effective for these datasets (2).

Authors
-------
Brian Svoboda

Copyright 2021 by Brian Svoboda under the MIT License.
"""

import sys
import shutil
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import scipy as sp

import spectral_cube
from astropy import convolution
from astropy import units as u
from astropy import constants as c
from astropy.io import fits

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
import nestfit as nf
from nestfit import (
        Distribution,
        Prior,
        ResolvedPlacementPrior,
        ConstantPrior,
        PriorTransformer,
        take_by_components,
)

from . import (PLOT_PATH, DATA_PATH, RUN_PATH, ALL_TARGETS)


FWHM = 2 * np.sqrt(2 * np.log(2))


def fix_cube(filen):
    """
    Fix a few things about the cubes so that they don't have to be modified on
    every run. The N2H+ model operates in terms brightness temperature units
    Kelvin, so the existing units of Jy/beam need to be converted. The data
    must also be in ascending frequency order.
    """
    cube = spectral_cube.SpectralCube.read(filen, memmap=False)
    # Convert cube to K from Jy/beam
    cube = cube.to('K')
    cube = cube.with_spectral_unit('Hz')
    # and into ascending frequency ordering from descending
    xarr = cube.spectral_axis
    if xarr[1] < xarr[0]:
        cube = cube[::-1]
    outfilen = f'{Path(filen).stem}_fixed.fits'
    cube.write(outfilen, overwrite=True)


def get_cube(field):
    target = ALL_TARGETS[field]
    cube = spectral_cube.SpectralCube.read(str(target.path), memmap=False)
    return cube


def get_cubestack(field):
    cube = get_cube(field)
    # Cubes are big-endian byteorder, cast to machine little-endian through by
    # copying.
    cube._data = cube._data.astype(float, copy=False)
    rms = cube[:10].std().value
    datacube = nf.DataCube(cube, noise_map=rms, trans_id=1)
    return nf.CubeStack([datacube])


def get_uniform_priors(field, size=500):
    u = np.linspace(0, 1, size)
    target = ALL_TARGETS[field]
    vsys = target.vsys
    # Prior distribution x axes, must be uniform grid spacing.
    if field == 'W43-MM1':
        x_voff = np.linspace(vsys-7.5, vsys+7.5, size)
        x_tex  = np.linspace(     2.8,     30.0, size)
        x_ltau = np.linspace(    -2.0,      2.0, size)
        x_sigm = np.linspace(     0.2,      2.0, size)
    elif field == 'W51-IRS2':
        x_voff = np.linspace(    45.0,     73.0, size)
        x_tex  = np.linspace(     2.8,     30.0, size)
        x_ltau = np.linspace(    -2.0,      2.0, size)
        x_sigm = np.linspace(     0.2,      2.0, size)
    # Prior PDFs values, all uniform
    uniform = np.ones_like(u) / size
    f_voff = uniform.copy()
    f_tex  = uniform.copy()
    f_ltau = uniform.copy()
    f_sigm = uniform.copy()
    # Create distribution instances from "x" and "y" values
    d_voff = Distribution(x_voff, f_voff)
    d_tex  = Distribution(x_tex,  f_tex)
    d_ltau = Distribution(x_ltau, f_ltau)
    d_sigm = Distribution(x_sigm, f_sigm)
    # Create instances of interpolation objects called in the Nested Sampling.
    # The integer indicates the index position of the parameter in the model:
    #   0 -> voff, velocity centroid
    #   1 -> tex,  excitation temperature
    #   2 -> ltau, logarithmic tau
    #   3 -> sigm, velocity dispersion
    # The ordering of the Prior objects in the array is not actually important.
    # The `ResolvedPlacementPrior` ties the velocity centroid and dispersion
    # together such that components are resolved according to the geometric
    # mean separation criteria:
    #   sep > scale * FWHM * sqrt(sig1 * sig2)
    priors = np.array([
            ResolvedPlacementPrior(
                Prior(d_voff, 0),
                Prior(d_sigm, 3),
                scale=1.2,
            ),
            Prior(d_tex,  1),
            Prior(d_ltau, 2),
    ])
    return PriorTransformer(priors)


def get_empirical_priors(field):
    with h5py.File(DATA_PATH/f'{field}.hdf', 'r') as hdf:
        bins = hdf['bins'][...]
        vals = hdf['vals'][...]
    # Make sure that the priors are normalized
    vals /= vals.sum(axis=1, keepdims=True)
    # Prior distribution x axes, must be uniform grid spacing.
    x_voff, x_tex, x_ltau, x_sigm = bins
    f_voff, f_tex, f_ltau, f_sigm = vals
    # Create distribution instances from "x" and "y" values
    d_voff = Distribution(x_voff, f_voff)
    d_tex  = Distribution(x_tex,  f_tex)
    d_ltau = Distribution(x_ltau, f_ltau)
    d_sigm = Distribution(x_sigm, f_sigm)
    # See comment in `get_uniform_priors` on the instantiating the priors.
    priors = np.array([
            ResolvedPlacementPrior(
                Prior(d_voff, 0),
                Prior(d_sigm, 3),
                scale=1.2,
            ),
            Prior(d_tex,  1),
            Prior(d_ltau, 2),
    ])
    return PriorTransformer(priors)


def verify_prior_transformer(utrans, ncomp=1):
    """
    The `PriorTransformer` transforms [0, 1] values from the unit cube into the
    actual parameter values. Thus it's a good idea to check that 0.0 (minimum),
    0.5 (median), and 1.0 (maximum) all give the expected values.

    Parameters
    ----------
    utrans : nestfit.PriorTransformer
    ncomp : int
        Number of velocity components
    """
    assert ncomp > 0
    utheta = np.zeros(4*ncomp)
    for v in (0.0, 0.5, 1.0):
        utheta[:] = v
        utrans.transform(utheta, ncomp)
        print(v, utheta)


def if_store_exists_delete(name):
    filen = f'{name}.store'
    if Path(filen).exists():
        print(f'-- Deleting {filen}')
        shutil.rmtree(filen)


def run_nested(field, store_suffix, utrans, nproc=8, ncomp_max=2):
    """
    Run the N2H+ line fitting using Nested Sampling. The results will be
    written to a ".store" file written to the ``RUN_PATH``.

    Parameters
    ----------
    field : str
        Target field name
    store_suffix : str
        Name suffix to append to the field name when creating the HdfStore
        file, of the form ``"<RUN_PATH>/{field}_{store_suffix}.store"``.
    utrans : nestfit.PriorTransformer
        Prior transformer instance, create with e.g., `get_uniform_priors`
    nproc : int
        Number of processes to use. A value of 1 will use serial execution
        without the Python ``multiprocessing`` library.
    ncomp_max : int
        Maximum number of velocity components to fit in a spectrum. Models will
        be fit with an increasing number of components until they are no longer
        significant according to the Bayes factors (model i and i-1) or the
        maximum number of components is reached.
    """
    store_name = RUN_PATH / f'{field}_{store_suffix}'
    if_store_exists_delete(store_name)
    runner_cls = nf.DiazenyliumRunner
    stack = get_cubestack(field)
    fitter = nf.CubeFitter(stack, utrans, runner_cls, ncomp_max=ncomp_max,
            mn_kwargs={'nlive': 500}, nlive_snr_fact=20)
    fitter.fit_cube(store_name=store_name, nproc=nproc)


def get_store(field, store_suffix):
    store_path = RUN_PATH / f'{field}_{store_suffix}.store'
    assert store_path.exists()
    store = nf.HdfStore(str(store_path))
    return store


def get_info_kernel(nrad):
    hpbw = 0.5  # arcsec
    pix_size = 0.11  # arcsec
    beam_sigma_pix = hpbw / FWHM / pix_size
    k_arr = nf.get_indep_info_kernel(beam_sigma_pix, nrad=nrad)
    k_arr = nf.apply_circular_mask(k_arr, radius=nrad//2)
    post_kernel = convolution.CustomKernel(k_arr)
    return post_kernel


def get_runner(stack, utrans, ncomp=1):
    nlon, nlat = stack.spatial_shape
    spec_data, has_nans = stack.get_spec_data(nlon//2, nlat//2)
    assert not has_nans
    runner = nf.DiazenyliumRunner.from_data(spec_data, utrans, ncomp=ncomp)
    return runner


def postprocess_run(field, store_suffix, utrans):
    print(f':: Post-processing {field}_{store_suffix}')
    # Standard deviation in pixels: 1.1 -> FWHM 0.28 as (cf. HPBW / 2 = 0.25 as)
    evid_kernel = convolution.Gaussian2DKernel(1.1)
    post_kernel = get_info_kernel(6)  # 3.5 pixel radius circular window
    stack = get_cubestack(field)
    runner = get_runner(stack, utrans, ncomp=1)
    # begin post-processing steps
    with get_store(field, store_suffix) as store:
        #nf.aggregate_run_attributes(store)
        #nf.convolve_evidence(store, evid_kernel)
        #nf.aggregate_run_products(store)
        #nf.aggregate_run_pdfs(store)
        nf.deblend_hf_intensity(store, stack, runner)
        #nf.convolve_post_pdfs(store, post_kernel, evid_weight=False)
        #nf.quantize_conv_marginals(store)


def create_priors_from_posteriors(field, store_suffix):
    """
    The summation of the per-pixel posteriors gives a better global prior than
    the uniform distributions used initially to test. This also decreases the
    run-time because less volume of parameter space is needed to be explored in
    order to calculate the evidence.
    """
    with get_store(field, store_suffix) as store:
        # Run metadata.
        pdf_bins = store.hdf['/products/pdf_bins'][...]
        # Read in posteriors and MAP values from store file.
        # dimensions (b, l)
        nbest = store.hdf['/products/conv_nbest'][...]
        # dimensions (r, m, p, h, b, l)
        post = store.hdf['/products/post_pdfs'][...]
        post = take_by_components(post, nbest, incl_zero=False)  # -> (m, p, h, b, l)
        # Average/summed posterior distibution. Mask positions without
        # detections.
        apdf = np.nansum(post[...,nbest>=0], axis=(0, 3))
        # Normalize the PDFs
        apdf /= apdf.sum(axis=1, keepdims=True)
        # Write out to HDF5 file
        assert pdf_bins.shape == apdf.shape
        with h5py.File(DATA_PATH/f'{field}.hdf', 'w') as hdf:
            hdf.create_dataset('bins', data=pdf_bins)
            hdf.create_dataset('vals', data=apdf)


def export_deblended_to_fits(field, store_suffix):
    with get_store(field, store_suffix) as store:
        # dimensions (t, m, S, b, l)
        hfdb = store.hdf['/products/hf_deblended'][...]
        bins = store.hdf['/products/pdf_bins'][...]
        header = store.read_header()
    restfreq = header['RESTFRQ'] * u.Hz
    varr = bins[0]  # -> index for vcen
    farr = (restfreq - restfreq * (varr * u.km/u.s / c.c)).to('Hz').value
    header['CDELT3'] = farr[1] - farr[0]
    header['CRPIX3'] = 1
    header['CRVAL3'] = farr[0]
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['NAXIS3'] = farr.shape[0]
    # Select 1-0 transition and sum profiles from multiple components
    np.nan_to_num(hfdb, copy=False)
    cube = hfdb[0,...].sum(axis=0)  # -> (S, b, l)
    # Create FITS file
    hdu = fits.PrimaryHDU(data=cube, header=header)
    hdu.writeto(DATA_PATH/f'{field}_{store_suffix}_hfdb.fits', overwrite=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) > 0
    flag = args[0]
    assert flag in ('--run-nested', '--post-proc')
    # Parameters
    field = 'W51-IRS2'
    store_suffix = 'test_emp_2comp'
    utrans = get_empirical_priors(field)
    ncomp_max = 2
    if flag == '--run-nested':
        run_nested(field, store_suffix, utrans, nproc=16, ncomp_max=ncomp_max)
    elif flag == '--post-proc':
        postprocess_run(field, store_suffix)


