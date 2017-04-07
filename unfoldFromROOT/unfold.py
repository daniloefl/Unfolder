#!/usr/bin/env python

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
import matplotlib.cm as cm

from Unfolder.Unfolder import Unfolder
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D
from readHistograms import *

sns.set(color_codes=True)
sns.set(font_scale=1.3)

varname = "m_{tt}"
extension = "png"

# get histograms from file
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal")

recoWithoutFakes = mig.project("y")

# plot migration matrix as it will be used next for unfolding
plotH2D(mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig", extension)

# plot 1D histograms for cross checks
plotH1D(bkg, "Reconstructed "+varname, "Events", "Background", "bkg", extension)
plotH1D(truth, "Particle-level "+varname, "Events", "Particle-level distribution", "truth", extension)
plotH1D(nrt, "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt", extension)
plotH1D(recoWithFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes", extension)
plotH1D(recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes", extension)
plotH1D(eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff", extension)

# generate fake data
data = recoWithFakes

# Create unfolding class
m = Unfolder(bkg, mig, eff, truth)
m.setUniformPrior()
#m.setGaussianPrior()

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response.T(), "Particle-level bin", "Reconstructed-level bin", "Response matrix P(r|t)*eff(t)", "responseMatrix", extension)
# and also the migration probabilities matrix
plotH2D(m.response_noeff.T(), "Particle-level bin", "Reconstructed-level bin", "Migration probabilities P(r|t)", "migrationMatrix", extension)

plotH1D(m.truth, "Particle-level "+varname, "Events", "Particle-level distribution estimated from migrations", "truth_crossCheck", extension)
plotH1D(m.eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection in unfolder", "eff_crossCheck", extension)
plotH2D(m.mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig_crossCheck", extension)

plotH1D(m.bkg, "Reconstructed "+varname, "Events", "Background (including fakes)", "bkg_crossCheck", extension)
plotH1D(m.recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution", "recoWithoutFakes_crossCheck", extension)

m.run(data)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

# plot correlations
sns.pairplot(pm.trace_to_dataframe(m.trace))
plt.savefig("pairPlot.%s" % extension)
plt.close()

# plot unfolded spectrum
m.plotUnfolded("plotUnfolded.png")

