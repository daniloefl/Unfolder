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

sns.set(context = "paper", style = "whitegrid", font_scale=1.1)

varname = "observable"
extension = "eps"

data_input = "A"

import sys
if len(sys.argv) > 1:
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

for i in recoWithFakes:
  response[i] = getNormResponse(mig[i], eff[i])
  recoWithoutFakes[i] = mig[i].project("y")

# generate perfect fake data
import sys
data = recoWithFakes[data_input] + bkg[data_input]

Nr = len(bkg["A"].val)

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()

m.run(data)
m.sample(50000)

unf_orig = m.hunf
unf_orig_mode = m.hunf_mode

del m
n = None
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
  
# use uncertainty at reconstruction level, by using the nominal truth folded with the alternative response matrix
m.addUncertainty("B", bkg["B"], recoWithoutFakes["B"])

print("Response")
print(m.response.val)
print("Bkg and reco (nominal)")
print(m.bkg.val)
print(m.recoWithoutFakes.val)
print("Uncertainties in bkg and reco")
print(m.bkg_syst['B'].val)
print(m.reco_syst['B'].val)
print("Data")
print(data.val)

m.run(data)
m.sample(50000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in ["B"]:
  m.plotNPMarginal(i, "plotNPMarginal_%s.%s" % (i, extension))

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
#m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

comparePlot([truth[data_input], m.hunf_mode, unf_orig_mode], ["Truth %s" % data_input, "FBU w/ syst. (mode)", "FBU (mode)"], 1.0, False, "", logy = False, fname = "compareMethods.%s" % extension)
comparePlot([truth[data_input], m.hunf, unf_orig], ["Truth %s" % data_input, "FBU w/ syst. (mean)", "FBU (mean)"], 1.0, False, "", logy = False, fname = "compareMethods_mean.%s" % extension)

