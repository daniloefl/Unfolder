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
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D, getNormResponse
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
response = {}

truth[""], recoWithFakes[""], bkg[""], mig[""], eff[""], nrt[""] = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
truth["me"], recoWithFakes["me"], bkg["me"], mig["me"], eff["me"], nrt["me"] = getHistograms("out_ttallhad_psrw_Syst.root", "aMcAtNloHerwigppEvtGen", "mttAsymm")
truth["ps"], recoWithFakes["ps"], bkg["ps"], mig["ps"], eff["ps"], nrt["ps"] = getHistograms("out_ttallhad_psrw_Syst.root", "PowhegHerwigppEvtGen", "mttAsymm")

for i in recoWithFakes:
  response[i] = getNormResponse(mig[i], eff[i])
  recoWithoutFakes[i] = mig[i].project("y")

inp = ""
import sys
if len(sys.argv) > 1:
  inp = sys.argv[1]

# generate perfect fake data
data = recoWithFakes[inp]
#data = recoWithFakes["me"]
#data = recoWithFakes["ps"]

# Create unfolding class
m = Unfolder(bkg[""], mig[""], eff[""], truth[""])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
#m.setFirstDerivativePrior()


# add uncertainties
#uncList = ['sjcalib1030', 'eup', 'ecup'] + ['lup'+str(x) for x in range(0, 10+1)] + ['cup'+str(x) for x in range(0, 3+1)] + ['bup'+str(x) for x in range(0, 3+1)] + ['ewkup']
uncList = []
for k in uncList:
  print "Getting histograms for syst. ", k
  struth, srecoWithFakes, sbkg, smig, seff, snrt = getHistograms("out_ttallhad_psrw_Syst.root", k, "mttAsymm")
  m.addUncertainty(k, sbkg, smig.project('y'))

m.run(data)
m.setAlpha(0.0)
m.sample(100000)

unf_orig = m.hunf
  
uncUnfList = ["me", "ps"]
for k in uncUnfList:
  #m.addUnfoldingUncertainty(k, mig[k], eff[k])
  m.addUncertainty(k, bkg[k], np.dot(truth[k].val, response[k].val))

m.run(data)
m.setAlpha(0.0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncList:
  m.plotNPMarginal(i, "plotNPMarginal_%s.%s" % (i, extension))

for i in uncUnfList:
  #m.plotNPUMarginal(i, "plotNPUMarginal_%s.%s" % (i, extension))
  m.plotNPMarginal(i, "plotNPUMarginal_%s.%s" % (i, extension))

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
m.plotOnlyUnfolded(luminosity*1e-3, True, "fb/GeV", "plotOnlyUnfolded.%s" % extension)

comparePlot([truth[""], truth["me"], truth["ps"], m.hunf, unf_orig], ["Particle-level", "Particle-level ME", "Particle-level PS", "FBU", "FBU no syst."], luminosity*1e-3, True, "fb/GeV", "compareMethods.%s" % extension)

