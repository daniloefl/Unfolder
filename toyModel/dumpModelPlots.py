#!/usr/bin/env python3

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
truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms(direc = "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms(direc = "B")
#truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i], "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", logz = False, fname = "mig_%s.%s" % (i,extension), fmt ="")

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", logy = False, fname = "bkg_%s.%s" % (i, extension))

  plotH1D({"Original truth": truth[i], "Projected": mig[i].project("x")/eff[i]}, "Particle-level "+varname, "Events", "Particle-level distribution", logy = True, fname = "truth_project.%s" % extension)
  plotH1D({"Original truth - projected": truth[i] - mig[i].project("x")/eff[i]}, "Particle-level "+varname, "Events", "Particle-level distribution", logy = False, fname = "truth_project_diff.%s" % extension)

  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", logy = True, fname = "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", logy = False, fname = "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", logy = False, fname = "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", logy = True, fname = "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", logy = False, fname = "eff_%s.%s" % (i,extension))

  plotH1D({"Reco with fakes from projection": recoWithoutFakes[i]+bkg[i], "From getHistograms": recoWithFakes[i]}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", logy = True, fname = "reco_project.%s" % extension)
  plotH1D({"Reco with fakes from projection - getHistograms": recoWithoutFakes[i]+bkg[i] - recoWithFakes[i]}, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", logy = False, fname = "reco_project_diff.%s" % extension)

  # Create unfolding class
  m = Unfolder(bkg[i], mig[i], eff[i], truth[i])

  # plot response matrix P(r|t)*eff(r)
  plotH2D(m.response, "Reconstructed-level bin", "Particle-level bin", "Transpose of response matrix P(r|t)*eff(t)", logz = True, fname = "responseMatrix_%s.%s" % (i, extension))
# and also the migration probabilities matrix
  plotH2D(m.response_noeff, "Reconstructed-level bin", "Particle-level bin", "Transpose of migration probabilities P(r|t)", logz = False, fname = "migrationMatrix_%s.%s" % (i, extension), vmin = 0, vmax = 1)

comparePlot(listHist = [bkg["A"], bkg["B"]],
            listLegend = ["Background A", "Background B"],
            f = 1.0, normaliseByBinWidth = False, units = "", logy = True, fname = "compare_bkg.%s" % extension)

comparePlot(listHist = [eff["A"], eff["B"]],
            listLegend = ["Efficiency A", "Efficiency B"],
            f = 1.0, normaliseByBinWidth = False, units = "", fname = "compare_eff.%s" % extension)

comparePlot(listHist = [recoWithoutFakes["A"], recoWithoutFakes["B"]],
            listLegend = ["Reco A", "Reco B"],
            f = 1.0, normaliseByBinWidth = False, units = "", logy = True, fname = "compare_reco.%s" % extension)

comparePlot(listHist = [recoWithFakes["A"], recoWithFakes["B"]],
            listLegend = ["Reco+bkg A", "Reco+bkg B"],
            f = 1.0, normaliseByBinWidth = False, units = "", logy = True, fname = "compare_recobkg.%s" % extension)

comparePlot(listHist = [truth["A"], truth["B"]],
            listLegend = ["Truth A", "Truth B"],
            f = 1.0, normaliseByBinWidth = False, units = "", logy = True, fname = "compare_truth.%s" % extension)

