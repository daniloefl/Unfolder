
import itertools
from dfu import dfu
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
sns.set(color_codes=True)
sns.set(font_scale=1.3)
import matplotlib.cm as cm

makeDebugPlots = True
showPlots = False
extension = "png"

def produceIt(N, dist, tl, rl):
  for i in range(0, N):
    t = dist()
    if t < 0 or t > 2: continue
    r = t*np.random.normal(1.0, 0.2)
    if r > 2:
      r = 2
    if np.random.uniform() < 0.8:
      r = -1
    tl.append(t)
    rl.append(r)

def generateSample():
  bkg = []
  for i in range(0, 500):
    bkg.append(np.random.exponential(2.0))

  N1 = 100000
  N2 = 10000
  tl = []
  rl = []
  produceIt(N1, lambda : np.random.normal(0.4, 0.4), tl, rl)
  produceIt(N2, lambda : np.random.exponential(1.0), tl, rl)
  tr = pd.DataFrame({'Truth': tl, 'Reco': rl})

  return [tr, bkg]

def plotSample(tr, bkg, x):
  fig = plt.figure()
  ax = fig.add_subplot(131, title='Truth')
  sns.distplot(tr['Truth'], x, kde = False, hist = True, label = "Truth")
  ax = fig.add_subplot(132, title='Reco')
  sns.distplot(tr['Reco'], x, kde = False, hist = True, label = "Reco")
  ax = fig.add_subplot(133, title='Background')
  sns.distplot(bkg, x, kde = False, hist = True, label = "Bkg.")
  plt.legend()
  if showPlots:
    plt.show()
  plt.savefig('plotSample.%s' % extension)
  plt.close()

  fig = plt.figure()
  sns.jointplot(x = "Truth", y = "Reco", data = tr[tr['Reco'] >= 0], kind="reg", scatter_kws={"s": 20})
  if showPlots:
    plt.show()
  plt.savefig('jointDist.%s' % extension)
  plt.close()

def makeHistograms(tr, bkg, x):
  htruth, useless1 = np.histogram(tr['Truth'], bins = x)
  hreco, useless2 = np.histogram(tr['Reco'], bins = x)
  hbkg, useless3 = np.histogram(bkg, bins = x)
  hbkg = np.array(hbkg, dtype = np.float64)
  hmig, useless1, useless2 = np.histogram2d(tr['Truth'], tr['Reco'], bins=(x, x))

  hnrt, useless = np.histogram(tr[tr['Reco'] < 0], bins = x)
  # eff = truth and reco / truth  = (truth - (not reco and truth)) / truth = 1 - (not reco and truth)/truth
  heff = [1 - a*1.0/(b*1.0) for (a,b) in itertools.izip(hnrt, htruth)]

  return [htruth, hreco, hbkg, hmig, hnrt, heff]

def plotMigrations(hmig, x, title = "Migration matrix M(t, r)", fname = "prt"):
  fig = plt.figure(figsize=(15, 15))
  sns.heatmap(hmig, cmap=cm.RdYlGn, annot = True, linewidths=.5, square = True, annot_kws={"size": 20})
  #, extent=[x[0], x[-1], x[0], x[-1]])
  plt.title(title)
  plt.ylabel("Reco")
  plt.xlabel("Truth")
  #plt.colorbar()
  plt.tight_layout()
  if showPlots:
    plt.show()
  plt.savefig('%s.%s' % (fname, extension))
  plt.close()

def plotUnfoldingTargets(hbkg, htruth, hreco, x, x_err):
  fig = plt.figure()
  ax = fig.add_subplot(131, title='Bkg')
  plt.errorbar(x[:-1]+x_err[:-1], hbkg, np.sqrt(hbkg), x_err[:-1], fmt = 'gv', markersize=10)
  plt.ylabel("Background events")
  plt.xlabel("Reco observable")

  ax = fig.add_subplot(132, title='Truth')
  plt.errorbar(x[:-1]+x_err[:-1], htruth, np.sqrt(htruth), x_err[:-1], fmt='bo', markersize=10)
  plt.ylabel("Truth events")
  plt.xlabel("Truth observable")

  ax = fig.add_subplot(133, title='Reco')
  plt.errorbar(x[:-1]+x_err[:-1], hreco, np.sqrt(hreco), x_err[:-1], fmt='ro', markersize=10)
  plt.ylabel("Reco events")
  plt.xlabel("Reco observable")

  if showPlots:
    plt.show()
  plt.savefig('%s.%s' % ("plotUnfoldingTargets", extension))
  plt.close()

def plotEfficiency(heff, x, x_err):
  fig = plt.figure()
  plt.title("Efficiency")
  plt.errorbar(x[:-1]+x_err[:-1], heff, np.zeros(len(heff)), x_err[:-1], fmt = 'ro', markersize=10)
  plt.ylabel("Efficiency")
  plt.xlabel("Truth observable")
  if showPlots:
    plt.show()
  plt.savefig('%s.%s' % ("plotEfficiency", extension))
  plt.close()



# set observable axis
x = np.linspace(0, 2.0, 10)
# set this to half the axis width for plotting later
x_err = [(2.0-0.0)/10.0*0.5]*len(x)

# generate truth and reco samples
tr, bkg = generateSample()
# plot some histograms
if makeDebugPlots:
  plotSample(tr, bkg, x)

# generate histograms of truth, reco, background, migration matrix, reco and not truth, and efficiency
htruth, hreco, hbkg, hmig, hnrt, heff = makeHistograms(tr, bkg, x)

# plot migration matrix as it will be used next for unfolding
if makeDebugPlots:
  plotMigrations(hmig.T, x, "Number of events for each (reco, truth) configuration")

# plot the truth, reco and bkg histograms as they will be received by the unfolding
# code
if makeDebugPlots:
  plotUnfoldingTargets(hbkg, htruth, hreco, x, x_err)

# plot the efficiency factor
if makeDebugPlots:
  plotEfficiency(heff, x, x_err)

# generate fake data
hdata = np.zeros(len(hbkg))
for i in range(0, len(hbkg)): hdata[i] += hbkg[i]
for i in range(0, len(hreco)): hdata[i] += hreco[i]


# Create unfolding class
m = dfu(hbkg, hmig, heff)

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
plotMigrations(m.response.T, x, "Response matrix P(r|t)*eff(t)", "responseMatrix")
# and also the migration probabilities matrix
plotMigrations(m.response_noeff.T, x, "Migration probabilities P(r|t)", "migrationMatrix")

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

