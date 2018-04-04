#!/usr/bin/env python3

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
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D, getNormResponse
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
    uncUnfList = ["B"]
  if "bias" in sys.argv:
    fb = 1.0

  if "inputA" in sys.argv:
    data_input = "A"
  if "inputB" in sys.argv:
    data_input = "B"


# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}
response = {}

truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms(direc = "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms(direc = "B")
#truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  response[i] = getNormResponse(mig[i], eff[i])
  recoWithoutFakes[i] = mig[i].project("y")

# generate perfect fake data
import sys
data = recoWithFakes[data_input]

Nr = len(bkg["A"].val)

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
if fb > 0:
  m.setFirstDerivativePrior(fb)

m.run(data)
#m.graph("model.png")
m.sample(10000)

unf_orig = m.hunf

del m
n = None
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
if fb > 0:
  m.setFirstDerivativePrior(fb)
  
for k in uncUnfList:
  # use uncertainty at reconstruction level, by using the nominal truth folded with the alternative response matrix
  m.addUncertainty(k, bkg["A"], np.dot(truth["A"].val, response[k].val))
  # one can also use a non-linear term in the unfolding
  #m.addUnfoldingUncertainty(k, mig[k], eff[k])

m.run(data)
#m.graph("modelWithSysts.png")
m.setAlpha(0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncUnfList:
  m.plotNPMarginal(i, "plotNPMarginal_%s.%s" % (i, extension))

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
#m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

comparePlot([truth[data_input], m.hunf, unf_orig], ["Truth %s" % data_input, "FBU w/ syst.", "FBU"], 1.0, False, "", logy = True, fname = "compareMethods.%s" % extension)

