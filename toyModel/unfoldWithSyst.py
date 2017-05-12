#!/usr/bin/env python

# Unfolds using uncertainties in FBU, but with no regularisation


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

sns.set(context = "paper", style = "whitegrid", font_scale=2)

varname = "observable"
extension = "eps"

# uncertainties
uncUnfList = []

# bias in regularisation?
fb = 0.0

data_input = "A"

import sys
if len(sys.argv) > 1:
  if "withSyst" in sys.argv:
    uncUnfList = ["B", "C"]
  if "bias" in sys.argv:
    fb = 1.0

  if "inputA" in sys.argv:
    data_input = "A"
  if "inputB" in sys.argv:
    data_input = "B"
  if "inputC" in sys.argv:
    data_input = "C"


# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms("histograms.pkl", "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms("histograms.pkl", "B")
truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

# generate perfect fake data
import sys
data = recoWithFakes[data_input]
#data = recoWithFakes["B"]
#data = recoWithFakes["C"]

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
if fb > 0:
  m.setFirstDerivativePrior(fb)

m.run(data)
m.sample(100000)

unf_orig = m.hunf
  
for k in uncUnfList:
  m.addUnfoldingUncertainty(k, bkg[k], mig[k], eff[k])

m.run(data)
m.setAlpha(0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncUnfList:
  m.plotNPUMarginal(i, "plotNPUMarginal_%s.%s" % (i, extension))

# plot correlations
#m.plotPairs("pairPlot.%s" % extension) # takes forever
m.plotCov("covPlot.%s" % extension)
m.plotCorr("corrPlot.%s" % extension)
m.plotCorrWithNP("corrPlotWithNP.%s" % extension)
m.plotSkewness("skewPlot.%s" % extension)
m.plotKurtosis("kurtosisPlot.%s" % extension)

m.plotNP("plotNP.%s" % extension)
m.plotNPU("plotNPU.%s" % extension)

# plot unfolded spectrum
m.plotUnfolded("plotUnfolded.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

comparePlot([truth[data_input], m.hunf, unf_orig], ["Truth %s" % data_input, "FBU w/ syst.", "FBU"], 1.0, False, "", "compareMethods.%s" % extension)

