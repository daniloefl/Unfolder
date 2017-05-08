#!/usr/bin/env python

# This only dumps the model histograms stored in histograms.pkl

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

# plot migration matrix as it will be used next for unfolding
plotH2D(mig, "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig.%s" % extension)

# plot 1D histograms for cross checks
plotH1D(bkg, "Reconstructed "+varname, "Events", "Background", "bkg.%s" % extension)
plotH1D(truth, "Particle-level "+varname, "Events", "Particle-level distribution", "truth.%s" % extension)
plotH1D({truth: "Original truth", mig.project("x")/eff: "Projected"}, "Particle-level "+varname, "Events", "Particle-level distribution", "truth_project.%s" % extension)
plotH1D({truth - mig.project("x")/eff: "Original truth - projected"}, "Particle-level "+varname, "Events", "Particle-level distribution", "truth_project_diff.%s" % extension)
plotH1D(nrt, "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt.%s" % extension)
plotH1D(recoWithFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes.%s" % extension)
plotH1D(recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes.%s" % extension)
plotH1D({recoWithoutFakes+bkg: "Reco with fakes from projection", recoWithFakes: "From getHistograms"}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "reco_project.%s" % extension)
plotH1D({recoWithoutFakes+bkg - recoWithFakes: "Reco with fakes from projection - getHistograms"}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "reco_project_diff.%s" % extension)
plotH1D(eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff.%s" % extension)

eff_noerr = H1D(eff)
for k in range(0, len(eff_noerr.err)):
  eff_noerr.err[k] = 0

bkg_noerr = H1D(bkg)
for k in range(0, len(bkg_noerr.err)):
  bkg_noerr.err[k] = 0

# Create unfolding class
m = Unfolder(bkg, mig, eff, truth)

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response, "Reconstructed-level bin", "Particle-level bin", "Transpose of response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension)
# and also the migration probabilities matrix
plotH2D(m.response_noeff, "Reconstructed-level bin", "Particle-level bin", "Transpose of migration probabilities P(r|t)", "migrationMatrix.%s" % extension)

plotH1D(m.truth, "Particle-level "+varname, "Events", "Particle-level distribution estimated from migrations", "truth_crossCheck.%s" % extension)
plotH1D(m.eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection in unfolder", "eff_crossCheck.%s" % extension)
plotH2D(m.mig, "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig_crossCheck.%s" % extension)

plotH1D(m.bkg, "Reconstructed "+varname, "Events", "Background (including fakes)", "bkg_crossCheck.%s" % extension)
plotH1D(m.recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution", "recoWithoutFakes_crossCheck.%s" % extension)
