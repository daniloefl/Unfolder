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
truth["D"], recoWithFakes["D"], bkg["D"], mig["D"], eff["D"], nrt["D"] = getHistograms("histograms.pkl", "D")

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
  m = Unfolder(bkg[i], mig[i], eff[i], truth[i])

  # plot response matrix P(r|t)*eff(r)
  plotH2D(m.response, "Reconstructed-level bin", "Particle-level bin", "Transpose of response matrix P(r|t)*eff(t)", "responseMatrix_%s.%s" % (i, extension), 0, np.amax(eff[i].val)*1.2)
# and also the migration probabilities matrix
  plotH2D(m.response_noeff, "Reconstructed-level bin", "Particle-level bin", "Transpose of migration probabilities P(r|t)", "migrationMatrix_%s.%s" % (i, extension), 0, 1)

comparePlot([bkg["A"], bkg["B"], bkg["C"], bkg["D"]],
            ["Background A", "Background B", "Background C", "Background D"],
            1.0, False, "", "compare_bkg.%s" % extension)

comparePlot([eff["A"], eff["B"], eff["C"], eff["D"]],
            ["Efficiency A", "Efficiency B", "Efficiency C", "Efficiency D"],
            1.0, False, "", "compare_eff.%s" % extension)

comparePlot([recoWithoutFakes["A"], recoWithoutFakes["B"], recoWithoutFakes["C"], recoWithoutFakes["D"]],
            ["Reco A", "Reco B", "Reco C", "Reco D"],
            1.0, False, "", "compare_reco.%s" % extension)

comparePlot([recoWithFakes["A"], recoWithFakes["B"], recoWithFakes["C"], recoWithFakes["D"]],
            ["Reco+bkg A", "Reco+bkg B", "Reco+bkg C", "Reco+bkg D"],
            1.0, False, "", "compare_recobkg.%s" % extension)

comparePlot([truth["A"], truth["B"], truth["C"], truth["D"]],
            ["Truth A", "Truth B", "Truth C", "Truth D"],
            1.0, False, "", "compare_truth.%s" % extension)

