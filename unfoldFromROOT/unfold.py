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

sns.set(color_codes=True)
sns.set(font_scale=1.3)

varname = "m_{tt}"
extension = "png"

# get histograms from file
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")

recoWithoutFakes = mig.project("y")

# plot migration matrix as it will be used next for unfolding
plotH2D(mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig.%s" % extension)

# plot 1D histograms for cross checks
plotH1D(bkg, "Reconstructed "+varname, "Events", "Background", "bkg.%s" % extension)
plotH1D(truth, "Particle-level "+varname, "Events", "Particle-level distribution", "truth.%s" % extension)
plotH1D(nrt, "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt.%s" % extension)
plotH1D(recoWithFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes.%s" % extension)
plotH1D(recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes.%s" % extension)
plotH1D(eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff.%s" % extension)

# generate fake data
data = recoWithFakes

# Try alternative
# Create alternative method for unfolding
f_truth, f_recoWithFakes, f_bkg, f_mig, f_eff, f_nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
f_data = f_recoWithFakes
#tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeDerivative)
tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
#printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(0)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff

comparePlot([f_data, f_data - f_bkg, truth, tunfold_result], ["Data", "Data - bkg", "Particle-level", "TUnfold"], luminosity*1e-3, True, "fb/GeV", "plotTUnfold.png")

# now use D'Agostini
useDAgostini = False
try:
  dagostini_mig = getDAgostini(bkg, mig, data, nIter = 1)
  dagostini_result = dagostini_mig/eff

  comparePlot([data, data - bkg, truth, dagostini_result], ["Data", "Data - bkg", "Particle-level", "D'Agostini"], luminosity*1e-3, True, "fb/GeV", "plotDAgostini.png")
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
  plotH1D(m.bkg_syst[k], "Reconstructed "+varname, "Events", "Background Uncertainty "+k, "bkg_unc_%s.%s" % (k, extension))
  plotH1D(m.reco_syst[k], "Reconstructed "+varname, "Events", "Impact in reconstructed distribution due to uncertainty "+k, "recoWithoutFakes_unc_%s.%s" % (k, extension))
  

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response.T(), "Particle-level bin", "Reconstructed-level bin", "Response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension)
# and also the migration probabilities matrix
plotH2D(m.response_noeff.T(), "Particle-level bin", "Reconstructed-level bin", "Migration probabilities P(r|t)", "migrationMatrix.%s" % extension)

plotH1D(m.truth, "Particle-level "+varname, "Events", "Particle-level distribution estimated from migrations", "truth_crossCheck.%s" % extension)
plotH1D(m.eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection in unfolder", "eff_crossCheck.%s" % extension)
plotH2D(m.mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig_crossCheck.%s" % extension)

plotH1D(m.bkg, "Reconstructed "+varname, "Events", "Background (including fakes)", "bkg_crossCheck.%s" % extension)
plotH1D(m.recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution", "recoWithoutFakes_crossCheck.%s" % extension)

m.run(data)
m.setAlpha(1.0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncList:
  m.plotNPMarginal(i, "plotNPMarginal_%s.%s" % (i, extension))

# plot correlations
m.plotPairs("pairPlot.%s" % extension) # takes forever
m.plotCov("covPlot.%s" % extension)
m.plotCorr("corrPlot.%s" % extension)
m.plotCorrWithNP("corrPlotWithNP.%s" % extension)
m.plotSkewness("skewPlot.%s" % extension)
m.plotKurtosis("kurtosisPlot.%s" % extension)

m.plotNP("plotNP.%s" % extension)


#print "Mean of unfolded data:"
#print np.mean(m.trace.Truth, axis = 0)
#print "Sqrt of variance of unfolded data:"
#print np.std(m.trace.Truth, axis = 0)
#print "Skewness of unfolded data:"
#print scipy.stats.skew(m.trace.Truth, bias = False)
#print "Kurtosis of unfolded data:"
#print scipy.stats.kurtosis(m.trace.Truth, bias = False)
#print "Print out of the covariance matrix follows:"
#print np.cov(m.trace.Truth, rowvar = False)


# plot unfolded spectrum
m.plotUnfolded("plotUnfolded.png")
m.plotOnlyUnfolded(luminosity*1e-3, True, "fb/GeV", "plotOnlyUnfolded.png")

if useDAgostini:
  comparePlot([data, data - bkg, truth, tunfold_result, dagostini_result, m.hunf], ["Data", "Data - background", "Particle-level", "TUnfold", "D'Agostini", "FBU"], luminosity*1e-3, True, "fb/GeV", "compareMethods.png")
else:
  comparePlot([data, data - bkg, truth, tunfold_result, m.hunf], ["Data", "Data - background", "Particle-level", "TUnfold", "FBU"], luminosity*1e-3, True, "fb/GeV", "compareMethods.png")

