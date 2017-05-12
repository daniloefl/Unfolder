#!/usr/bin/env python

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
import matplotlib.cm as cm
import scipy

from Unfolder.ComparisonHelpers import *
from Unfolder.Unfolder import Unfolder
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D
from readHistograms import *

sns.set(context = "paper", style = "whitegrid", font_scale=2.0)

varname = "observable"
extension = "eps"

# get histograms from file
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("histograms.pkl", "A")

recoWithoutFakes = mig.project("y")

eff_noerr = H1D(eff)
for k in range(0, len(eff_noerr.err)):
  eff_noerr.err[k] = 0

bkg_noerr = H1D(bkg)
for k in range(0, len(bkg_noerr.err)):
  bkg_noerr.err[k] = 0

# generate fake data
data = recoWithFakes

# Create unfolding class
m = Unfolder(bkg, mig, eff, truth)
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
#m.setFirstDerivativePrior()


m.run(data)
m.setAlpha(1.0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

# plot correlations
#m.plotPairs("pairPlot.%s" % extension) # takes forever
m.plotCov("covPlot.%s" % extension)
m.plotCorr("corrPlot.%s" % extension)
m.plotCorrWithNP("corrPlotWithNP.%s" % extension)
m.plotSkewness("skewPlot.%s" % extension)
m.plotKurtosis("kurtosisPlot.%s" % extension)

m.plotNP("plotNP.%s" % extension)

# plot unfolded spectrum
m.plotUnfolded("plotUnfolded.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

