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

truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}
# get histograms from file
truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms("histograms.pkl", "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms("histograms.pkl", "B")
truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i], "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig_%s.%s" % (i,extension), fmt ="")

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", "bkg_%s.%s" % (i, extension))

  plotH1D({"Original truth": truth[i], "Projected": mig[i].project("x")/eff[i]}, "Particle-level "+varname, "Events", "Particle-level distribution", "truth_project.%s" % extension)
  plotH1D({"Original truth - projected": truth[i] - mig[i].project("x")/eff[i]}, "Particle-level "+varname, "Events", "Particle-level distribution", "truth_project_diff.%s" % extension)

  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff_%s.%s" % (i,extension))

  plotH1D({"Reco with fakes from projection": recoWithoutFakes[i]+bkg[i], "From getHistograms": recoWithFakes[i]}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "reco_project.%s" % extension)
  plotH1D({"Reco with fakes from projection - getHistograms": recoWithoutFakes[i]+bkg[i] - recoWithFakes[i]}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "reco_project_diff.%s" % extension)

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response, "Reconstructed-level bin", "Particle-level bin", "Transpose of response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension, 0, np.amax(eff.val)*1.2)
# and also the migration probabilities matrix
plotH2D(m.response_noeff, "Reconstructed-level bin", "Particle-level bin", "Transpose of migration probabilities P(r|t)", "migrationMatrix.%s" % extension, 0, 1)

plotH1D(m.truth, "Particle-level "+varname, "Events", "Particle-level distribution estimated from migrations", "truth_crossCheck.%s" % extension)
plotH1D(m.eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection in unfolder", "eff_crossCheck.%s" % extension)
plotH2D(m.mig, "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig_crossCheck.%s" % extension, fmt = "")

plotH1D(m.bkg, "Reconstructed "+varname, "Events", "Background (including fakes)", "bkg_crossCheck.%s" % extension)
plotH1D(m.recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution", "recoWithoutFakes_crossCheck.%s" % extension)

