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
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")

recoWithoutFakes = mig.project("y")

eff_noerr = H1D(eff)
for k in range(0, len(eff_noerr.err)):
  eff_noerr.err[k] = 0

bkg_noerr = H1D(bkg)
for k in range(0, len(bkg_noerr.err)):
  bkg_noerr.err[k] = 0

# generate fake data
data = recoWithFakes

# Try alternative
# Create alternative method for unfolding
#tunfolder = getTUnfolder(bkg, mig, eff, data, regMode = ROOT.TUnfold.kRegModeDerivative)
tunfolder = getTUnfolder(bkg, mig, eff, data, regMode = ROOT.TUnfold.kRegModeNone, normMode = 0)
# no regularization
#printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(0)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff_noerr

comparePlot([truth, tunfold_result], ["Particle-level", "TUnfold"], luminosity*1e-3, True, "fb/GeV", "plotTUnfold.%s" % extension)

# now use D'Agostini
useDAgostini = False
try:
  dagostini_mig = getDAgostini(bkg, mig, eff_noerr, data, nIter = 1)
  dagostini_result = dagostini_mig #/eff_noerr

  comparePlot([truth, dagostini_result], ["Particle-level", "D'Agostini"], luminosity*1e-3, True, "fb/GeV", "plotDAgostini.%s" % extension)
  useDAgostini = True
except:
  print "Could not use D'Agostini method. Continuing."

# Create unfolding class
m = Unfolder(bkg, mig, eff, truth)
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
m.setAlpha(1.0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncList:
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
m.plotOnlyUnfolded(luminosity*1e-3, True, "fb/GeV", "plotOnlyUnfolded.%s" % extension)

if useDAgostini:
  comparePlot([truth, tunfold_result, dagostini_result, m.hunf], ["Particle-level", "TUnfold", "D'Agostini", "FBU"], luminosity*1e-3, True, "fb/GeV", "compareMethods.%s" % extension)
else:
  comparePlot([truth, tunfold_result, m.hunf], ["Particle-level", "TUnfold", "FBU"], luminosity*1e-3, True, "fb/GeV", "compareMethods.%s" % extension)

