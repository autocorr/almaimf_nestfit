ALMA-IMF NestFit Example
========================
The ``analyze`` package applies `NestFit
<https://github.com/autocorr/nestfit>`_ to N2H+ 1-0 image cubes from the
ALMA-IMF program. To use this package on new images, priors will need
be constructed similar to as found in ``analyze.core.get_uniform_priors`` or
``analyze.core.get_empirical_priors``. Then run NestFit using
``analyze.core.run_nested`` and perform post-processing on the store file with
``analyze.core.postprocess_run``. Plots can then be generated from the run using
``analyze.plotting.make_all_plots``.

License
-------
Copyright 2021 by Brian Svoboda and released under the MIT License.
