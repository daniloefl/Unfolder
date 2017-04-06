#!/usr/bin/env python

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
import matplotlib.cm as cm

from Unfolder import Unfolder

from utils import *

sns.set(color_codes=True)
sns.set(font_scale=1.3)

makeDebugPlots = True
extension = "png"


# set observable axis
x = np.linspace(0, 2.0, 10)
# set this to half the axis width for plotting later
x_err = [(2.0-0.0)/10.0*0.5]*len(x)

# generate truth and reco samples
tr, bkg = generateSample()
# plot some histograms
if makeDebugPlots:
  plotSample(tr, bkg, x, extension)

# generate histograms of truth, reco, background, migration matrix, reco and not truth, and efficiency
htruth, hreco, hbkg, hmig, hnrt, heff = makeHistograms(tr, bkg, x)

# plot migration matrix as it will be used next for unfolding
if makeDebugPlots:
  plotMigrations(hmig.T, x, "Number of events for each (reco, truth) configuration", "prt", extension)

# plot the truth, reco and bkg histograms as they will be received by the unfolding
# code
if makeDebugPlots:
  plotUnfoldingTargets(hbkg, htruth, hreco, x, x_err, extension)

# plot the efficiency factor
if makeDebugPlots:
  plotEfficiency(heff, x, x_err, extension)

# generate fake data
hdata = np.zeros(len(hbkg))
for i in range(0, len(hbkg)): hdata[i] += hbkg[i]
for i in range(0, len(hreco)): hdata[i] += hreco[i]


# Create unfolding class
m = Unfolder.Unfolder(hbkg, hmig, heff)
m.setUniformPrior()
#m.setGaussianPrior()

# double check that the truth calculated in the unfolding class
# using hmig is the same as the one we estimated directly before
fig = plt.figure(figsize=(10, 10))
plt.errorbar(x[:-1]+x_err[:-1], m.truth, np.zeros(len(m.truth)), x_err[:-1], fmt = 'ro', markersize=10, label="Truth from response matrix projection")
plt.errorbar(x[:-1]+x_err[:-1], htruth, np.zeros(len(htruth)), x_err[:-1], fmt = 'bv', markersize=5, label = "Truth from histogram of data")
plt.legend()
plt.xlabel("Truth observable")
plt.ylabel("Events")
plt.savefig("crossCheckTruth.png")
plt.close()

# plot response matrix P(r|t)*eff(r)
plotMigrations(m.response.T, x, "Response matrix P(r|t)*eff(t)", "responseMatrix", extension)
# and also the migration probabilities matrix
plotMigrations(m.response_noeff.T, x, "Migration probabilities P(r|t)", "migrationMatrix", extension)

#print "TEST"
#m.test()
#print "END OF TEST"
m.run(hdata)
m.sample(20000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

# plot correlations
sns.pairplot(pm.trace_to_dataframe(m.trace))
plt.savefig("pairPlot.%s" % extension)
plt.close()

#pm.plot_posterior(m.trace, varnames = ['Truth'], color='#87ceeb')
#pm.forestplot(m.trace, varnames=['Truth'], main="Truth")

# plot unfolded spectrum
m.plotUnfolded(x, x_err, "plotUnfolded.png")

