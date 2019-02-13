#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function


def process_tic(ticid, sector=None, subsample=20, datadir="."):
    import os
    cache_dir = "./theano_cache/{0}".format(os.getpid())
    os.environ["THEANO_FLAGS"] = \
        "compiledir={0}".format(cache_dir)

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    import lightkurve as lk

    import pymc3 as pm
    import theano.tensor as tt

    import exoplanet as xo

    # Output directory
    outdir = os.path.join(datadir, "results", "{0}".format(ticid))
    os.makedirs(outdir, exist_ok=True)

    # Download the light curve
    download_dir = os.path.join(datadir, "lightkurve")
    lc = lk.search_lightcurvefile("TIC {0}".format(ticid), sector=sector)
    lc = lc[len(lc) - 1].download(download_dir=download_dir)

    # Remove bad data
    pdc = lc.PDCSAP_FLUX.remove_nans()
    m = pdc.quality == 0
    x = np.ascontiguousarray(pdc.time[m])
    y = np.ascontiguousarray(pdc.flux[m])
    mu = np.median(y)
    y = (y / mu - 1)*1e3

    # Clip outliers using sigma clipping
    m = np.ones(len(y), dtype=bool)
    for i in range(10):
        y_prime = np.interp(x, x[m], y[m])
        smooth = savgol_filter(y_prime, 2001, polyorder=3)
        resid = y - smooth
        sigma = np.sqrt(np.median(resid**2))
        m0 = np.abs(resid) < 5*sigma
        if m.sum() == m0.sum():
            m = m0
            break
        m = m0

    # Subsample the data
    x = np.array(x[m][::subsample], dtype=np.float64)
    y = np.array(y[m][::subsample], dtype=np.float64)

    freq = np.exp(np.linspace(-np.log(1/400.0), -np.log(10), 500))
    with pm.Model() as model:

        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # A jitter term describing excess white noise
        logs2 = pm.Normal("logs2", mu=2*np.log(np.std(y)), sd=5.0)

        # A SHO term to capture long term trends
        logw1_guess = np.log(2*np.pi/3)
        logw1 = pm.Normal("logw1", mu=logw1_guess, sd=10)
        logpower1 = pm.Normal("logpower1",
                              mu=np.log(np.var(y))+4*logw1_guess,
                              sd=10.0)
        logS1 = pm.Deterministic("logS1", logpower1 - 4 * logw1)

        kernel = xo.gp.terms.SHOTerm(log_S0=logS1, log_w0=logw1,
                                     Q=1/np.sqrt(2))
        gp = xo.gp.GP(kernel, x, tt.exp(logs2) + np.zeros_like(y), J=2)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        pm.Potential("loglike", gp.log_likelihood(y - mean))

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict())
        pm.Deterministic("psd", kernel.psd(2*np.pi*freq))

        # Optimize to find the maximum a posteriori parameters
        map_soln = xo.optimize(start=model.test_point, vars=[mean, logs2])
        map_soln = xo.optimize(start=map_soln, vars=[mean, logs2, logpower1,
                                                     logw1])
        map_soln = xo.optimize(start=map_soln)

        # Run autodiff VI
        approx = pm.fit(n=10000, method="fullrank_advi", start=map_soln)
        trace = approx.sample(5000)

    # Save the trace and summary
    df = pm.trace_to_dataframe(trace)
    df.to_hdf(os.path.join(outdir, "trace.h5"), "trace")
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write("name,mean,error\n")
        for k in ["mean", "logs2", "logw1", "logpower1"]:
            f.write("{0},{1},{2}\n".format(k, np.mean(trace[k]),
                                           np.std(trace[k])))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    plt.tight_layout()

    ax = axes[0]
    ax.plot(x, y, "k")
    q = np.percentile(trace["pred"], [16, 50, 84], axis=0)
    ax.fill_between(x, q[0], q[2], alpha=0.5, color="C0")
    ax.plot(x, q[1], color="C0")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("flux [ppt]")
    ax.set_xlabel("time")

    ax = axes[1]
    q = np.percentile(trace["psd"], [16, 50, 84], axis=0)
    ax.fill_between(freq, q[0], q[2], alpha=0.5, color="C0")
    ax.loglog(freq, q[1], color="C0")
    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylabel("power")
    ax.set_xlabel("frequency [1/day]")

    fig.savefig(os.path.join(outdir, "plot.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        for ticid in sys.argv[1:]:
            process_tic(ticid)
    else:
        import random
        with open("output3.txt", "r") as f:
            ticids = [int(l.split("/")[-1].split("-")[2]) for l in f
                      if ".fits" in l]
            ticids = list(sorted(set(ticids)))
        ticid = random.choice(ticids)
        print(ticid)
        process_tic(ticid)
