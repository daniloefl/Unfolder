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
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal") #, "mttAsymm")

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

# generate perfect fake data
data = recoWithFakes

# generate fake data from model
pseudo_data = getDataFromModel(bkg, mig, eff, truth)

comparePlot([data, pseudo_data, data - bkg, pseudo_data - bkg],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments",
             "Reco. projected from unfolding factors - bkg", "Reco. simulated with toy experiments - bkg"],
            luminosity*1e-3, True, "fb/GeV", "pseudoData.png")

# Try alternative
# Create alternative method for unfolding
f_truth, f_recoWithFakes, f_bkg, f_mig, f_eff, f_nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
f_data = f_recoWithFakes
pseudo_f_data = getDataFromModel(f_bkg, f_mig, f_eff, f_truth)
tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeDerivative)
#tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
tau = printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(tau)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff

pseudo_tunfolder = getTUnfolder(f_bkg, f_mig, pseudo_f_data, regMode = ROOT.TUnfold.kRegModeDerivative)
#pseudo_tunfolder = getTUnfolder(f_bkg, f_mig, pseudo_f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
tau_pseudo = printLcurve(tunfolder, "tunfold_lcurve_pseudo.png")
pseudo_tunfolder.DoUnfold(tau_pseudo)
pseudo_tunfold_mig = H1D(pseudo_tunfolder.GetOutput("tunfold_pseudo_result"))
pseudo_tunfold_result = pseudo_tunfold_mig/eff

comparePlot([f_data, pseudo_f_data, f_truth,
             tunfold_result,
             pseudo_tunfold_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (TUnfold) from projected reco.",
             "Unfolded (TUnfold) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_TUnfold.png")

# Create unfolding class
m = Unfolder(bkg, mig, eff, truth)
#m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
m.setFirstDerivativePrior()

m_pseudo = Unfolder(bkg, mig, eff, truth)
#m_pseudo.setUniformPrior()
#m_pseudo.setGaussianPrior()
#m_pseudo.setCurvaturePrior()
m_pseudo.setFirstDerivativePrior()

# add uncertainties
uncList = [] #'sjcalib1030', 'eup', 'ecup'] + ['lup'+str(x) for x in range(0, 10+1)] + ['cup'+str(x) for x in range(0, 3+1)] + ['bup'+str(x) for x in range(0, 3+1)] + ['ewkup']
for k in uncList:
  print "Getting histograms for syst. ", k
  struth, srecoWithFakes, sbkg, smig, seff, snrt = getHistograms("out_ttallhad_psrw_Syst.root", k)
  m.addUncertainty(k, sbkg, smig.project('y'))
  plotH1D(m.bkg_syst[k], "Reconstructed "+varname, "Events", "Background Uncertainty "+k, "bkg_unc_%s.%s" % (k, extension))
  plotH1D(m.reco_syst[k], "Reconstructed "+varname, "Events", "Impact in reconstructed distribution due to uncertainty "+k, "recoWithoutFakes_unc_%s.%s" % (k, extension))
  m_pseudo.addUncertainty(k, sbkg, smig.project('y'))
  

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response.T(), "Particle-level bin", "Reconstructed-level bin", "Response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension)
# and also the migration probabilities matrix
plotH2D(m.response_noeff.T(), "Particle-level bin", "Reconstructed-level bin", "Migration probabilities P(r|t)", "migrationMatrix.%s" % extension)

# the following does the same as before, but
# it is used for double checking that the unfolding class reads the info. properly
#plotH1D(m.truth, "Particle-level "+varname, "Events", "Particle-level distribution estimated from migrations", "truth_crossCheck.%s" % extension)
#plotH1D(m.eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection in unfolder", "eff_crossCheck.%s" % extension)
#plotH2D(m.mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig_crossCheck.%s" % extension)
#plotH1D(m.bkg, "Reconstructed "+varname, "Events", "Background (including fakes)", "bkg_crossCheck.%s" % extension)
#plotH1D(m.recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution", "recoWithoutFakes_crossCheck.%s" % extension)

# does the same for the pseudo-data
m_pseudo.run(pseudo_data)
alpha, minBias = m_pseudo.scanAlpha(10000, np.arange(0.0, 2.0, 0.01), "scanAlpha_pseudo.%s" % extension)
print "Found alpha = ", alpha, " with bias = ", minBias
m_pseudo.setAlpha(alpha)
m_pseudo.sample(100000)

m.run(data)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)
m_pseudo.plotMarginal("plotMarginal_pseudo.%s" % extension)
for i in uncList:
  m.plotNPMarginal(i, "plotNPMarginal_%s.%s" % (i, extension))
  m_pseudo.plotNPMarginal(i, "plotNPMarginal_pseudo_%s.%s" % (i, extension))

# plot correlations graphically
# it takes forever and it is just pretty
# do it only if you really want to see it
# I just get annoyed waiting for it ...
#m.plotPairs("pairPlot.%s" % extension) # takes forever!

for k in [ [m, ""], [m_pseudo, "_pseudo"]]:
  k[0].plotCov("covPlot%s.%s" % (k[1], extension))
  k[0].plotCorr("corrPlot%s.%s" % (k[1], extension))
  k[0].plotCorrWithNP("corrPlotWithNP%s.%s" % (k[1], extension))
  k[0].plotSkewness("skewPlot%s.%s" % (k[1], extension))
  k[0].plotKurtosis("kurtosisPlot%s.%s" % (k[1], extension))
  k[0].plotNP("plotNP%s.%s" % (k[1], extension))

# for debugging
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

m_pseudo.plotUnfolded("plotUnfolded_pseudo.png")
m_pseudo.plotOnlyUnfolded(luminosity*1e-3, True, "fb/GeV", "plotOnlyUnfolded_pseudo.png")

comparePlot([data, pseudo_data, truth,
             m.hunf, m_pseudo.hunf,
             tunfold_result, pseudo_tunfold_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (FBU) from projected reco.", "Unfolded (FBU) from independently simulated reco.",
             "Unfolded (TUnfold) from projected reco.", "Unfolded (TUnfold) from independently simulated reco.",
            ],
            luminosity*1e-3, True, "fb/GeV", "biasTest.png")


