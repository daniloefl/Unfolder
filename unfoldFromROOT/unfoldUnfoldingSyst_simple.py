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

sns.set(context = "paper", style = "whitegrid", font_scale=2)

varname = "observable"
extension = "eps"

# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

truth[""], recoWithFakes[""], bkg[""], mig[""], eff[""], nrt[""] = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttCoarse")
truth["b"], recoWithFakes["b"], bkg["b"], mig["b"], eff["b"], nrt["b"] = getHistograms("out_ttallhad_psrw_Syst.root", "aMcAtNloHerwigppEvtGen", "mttCoarse")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i].T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig_%s.%s" % (i,extension))

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", "bkg_%s.%s" % (i, extension))
  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff_%s.%s" % (i,extension))

# generate perfect fake data
data = recoWithFakes[""]
#data = recoWithFakes["b"]

# Create unfolding class
m = Unfolder(bkg[""], mig[""], eff[""], truth[""])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
#m.setFirstDerivativePrior()

m.run(data)
m.setAlpha(0.0)
m.sample(100000)

unf_orig = m.hunf
  
uncUnfList = ["b"]
for k in uncUnfList:
  m.addUnfoldingUncertainty(k, bkg[k], mig[k], eff[k])

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response.T(), "Particle-level bin", "Reconstructed-level bin", "Response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension)
# and also the migration probabilities matrix
plotH2D(m.response_noeff.T(), "Particle-level bin", "Reconstructed-level bin", "Migration probabilities P(r|t)", "migrationMatrix.%s" % extension)

m.run(data)
m.setAlpha(0.0)
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

comparePlot([truth[""], truth["b"], m.hunf, unf_orig], ["Truth", "Truth B", "FBU with systematic uncertainties", "FBU no systematic uncertainties"], 1.0, False, "", "compareMethods.%s" % extension)

