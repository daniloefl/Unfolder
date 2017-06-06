#!/usr/bin/env python

import copy
from Unfolder.Histogram import H1D, H2D
import numpy as np
import pickle

'''
Generate unfolding factors.
'''
def generateHistograms(fname = "histograms.pkl"):
  Nev = 400000
  wL = 10.0 # generate wL times more events than we expect in data

  # number of truth bins
  xt = np.concatenate(( np.arange(1e3, 1.5e3, 0.25e3), np.arange(1.5e3, 3e3, 0.5e3)))
  #xt = np.arange(1e3, 3.1e3, 2e3)
  xt_err = np.diff(xt)*0.5
  xt = xt[:-1]
  xt += xt_err
  Nt = len(xt)

  # number of reco bins
  xf = np.concatenate(( np.arange(1e3, 1.5e3, 0.5*0.25e3), np.arange(1.5e3, 3e3, 0.5*0.5e3) ) )
  #xf = np.arange(1e3, 3.1e3, 2e3)
  xf_err = np.diff(xf)*0.5
  xf = xf[:-1]
  xf += xf_err
  Nr = len(xf)

  e = {}
  b = {}
  a = {}
  b = {}
  e["A"] = [0.20 for x in range(0, Nt)]
  e["B"] = [0.22 for x in range(0, Nt)]
  e["C"] = [0.18 for x in range(0, Nt)]
  e["D"] = [0.18 + 0.02*x for x in range(0, Nt)]

  a["A"] = 0.20
  a["B"] = 0.20
  a["C"] = 0.25
  a["D"] = 0.20
  b["A"] = 0.01
  b["B"] = 0.01
  b["C"] = 0.02
  b["D"] = 0.01

  #a["A"] = 0
  #a["B"] = 0
  #a["C"] = 0
  #a["D"] = 0
  #b["A"] = 0
  #b["B"] = 0
  #b["C"] = 0
  #b["D"] = 0

  p = {}
  p["l"] = {}
  p["l"]["A"] = 1/3e-3
  p["l"]["B"] = 1/3e-3
  p["l"]["C"] = 1/3e-3
  p["l"]["D"] = 1/3e-3

  truth = {}
  reco = {}
  mig = {}
  bkg = {}

  truth2 = {}
  reco2 = {}
  mig2 = {}
  bkg2 = {}
  for direc in ["A", "B", "C", "D"]:
    truth[direc] = H1D(np.zeros(Nt))
    truth[direc].x = xt
    truth[direc].x_err = xt_err
    mig[direc] = H2D(np.zeros((Nt, Nr)))
    mig[direc].x = xt
    mig[direc].x_err = xt_err
    mig[direc].y = xf
    mig[direc].y_err = xf_err
    reco[direc] = H1D(np.zeros(Nr))
    reco[direc].x = xf
    reco[direc].x_err = xf_err
    bkg[direc] = H1D(10*np.ones(Nr))
    bkg[direc].x = xf
    bkg[direc].x_err = xf_err
    for i in range(0, Nr): bkg[direc].err[i] = 0

    for k in range(0, Nev):
      # xp = x - 1e3 (force distribution to start at 1e3)
      # x = xp + 1e3
      # exp(-K x) = exp(-K xp)*exp(-1e3)
      # so, generate exp(-K xp) and weight it down by exp(-1e3)
      Op = 0
      Op = np.random.exponential(p["l"][direc])
      O = Op + 1e3
      #w = (1.0/wL)*np.exp(-1e3) # exp(-1e3) is too small, so just assume that wL = lumi*cross section are enough to factor it out
      w = 1.0/wL

      # migration model
      S = O*(a[direc]/np.sqrt(O) + b[direc])
      if S > 0:
        Or = O + np.random.normal(0, S)
      else:
        Or = O

      bt = truth[direc].fill(O, w)
      if np.random.uniform() > e[direc][bt]:
        continue
      mig[direc].fill(O, Or, w)
      br = reco[direc].fill(Or, w)
    reco[direc] = reco[direc] + bkg[direc]

  with open(fname, 'wb') as output:
    for direc in ["A", "B", "C", "D"]:
      model = {}
      model["name"] = direc
      model["mig"] = mig[direc].T()
      model["truth"] = truth[direc]
      model["reco"] = reco[direc]
      model["bkg"] = bkg[direc]
      pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  generateHistograms()

