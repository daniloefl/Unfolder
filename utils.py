
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
import matplotlib.cm as cm

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

def plotSample(tr, bkg, x, extension = "png"):
  fig = plt.figure()
  ax = fig.add_subplot(131, title='Truth')
  sns.distplot(tr['Truth'], x, kde = False, hist = True, label = "Truth")
  ax = fig.add_subplot(132, title='Reco')
  sns.distplot(tr['Reco'], x, kde = False, hist = True, label = "Reco")
  ax = fig.add_subplot(133, title='Background')
  sns.distplot(bkg, x, kde = False, hist = True, label = "Bkg.")
  plt.legend()
  plt.savefig('plotSample.%s' % extension)
  plt.close()

  fig = plt.figure()
  sns.jointplot(x = "Truth", y = "Reco", data = tr[tr['Reco'] >= 0], kind="reg", scatter_kws={"s": 20})
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

def plotMigrations(hmig, x, title = "Migration matrix M(t, r)", fname = "prt", extension = "png"):
  fig = plt.figure(figsize=(15, 15))
  sns.heatmap(hmig, cmap=cm.RdYlGn, annot = True, linewidths=.5, square = True, annot_kws={"size": 20})
  #, extent=[x[0], x[-1], x[0], x[-1]])
  plt.title(title)
  plt.ylabel("Reco")
  plt.xlabel("Truth")
  #plt.colorbar()
  plt.tight_layout()
  plt.savefig('%s.%s' % (fname, extension))
  plt.close()

def plotUnfoldingTargets(hbkg, htruth, hreco, x, x_err, extension = "png"):
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

  plt.savefig('%s.%s' % ("plotUnfoldingTargets", extension))
  plt.close()

def plotEfficiency(heff, x, x_err, extension = "png"):
  fig = plt.figure()
  plt.title("Efficiency")
  plt.errorbar(x[:-1]+x_err[:-1], heff, np.zeros(len(heff)), x_err[:-1], fmt = 'ro', markersize=10)
  plt.ylabel("Efficiency")
  plt.xlabel("Truth observable")
  plt.savefig('%s.%s' % ("plotEfficiency", extension))
  plt.close()


