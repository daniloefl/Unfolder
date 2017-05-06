#!/usr/bin/env python

import copy
from Unfolder.Histogram import H1D, H2D
import numpy as np
import pickle

'''
Generate unfolding factors.
'''
def generateHistograms(fname = "histograms.pkl"):
  Nev = 500000
  w = 1.0/10.0

  # number of truth bins
  xt = np.arange(0+0.5, 5, 0.5)
  Nt = len(xt)
  xt_err = np.ones(Nt)*0.5*0.5

  # number of reco bins
  xf = np.arange(0+0.125, 5, 0.125)
  Nr = len(xf)
  xf_err = np.ones(Nr)*0.125*0.5

  e = {}
  b = {}
  a = {}
  b = {}
  e["A"] = [0.20 for x in range(0, Nt)]
  e["B"] = [0.21 for x in range(0, Nt)]
  e["C"] = [0.22 for x in range(0, Nt)]

  a["A"] = 0.20
  a["B"] = 0.25
  a["C"] = 0.22
  b["A"] = 0.05
  b["B"] = 0.041
  b["C"] = 0.059

  p = {}
  p["l"] = {}
  p["m"] = {}
  p["s"] = {}
  p["l"]["A"] = 1/0.5
  p["l"]["B"] = 1/0.8
  p["l"]["C"] = 1/0.3
  p["m"]["A"] = 4.0
  p["m"]["B"] = 4.4
  p["m"]["C"] = 3.8
  p["s"]["A"] = 1.0
  p["s"]["B"] = 1.5
  p["s"]["C"] = 1.2

  truth = {}
  reco = {}
  mig = {}
  bkg = {}

  truth2 = {}
  reco2 = {}
  mig2 = {}
  bkg2 = {}
  for direc in ["A", "B", "C"]:
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
      O = 0
      if np.random.uniform() > 0.20:
        O = np.random.exponential(p["l"][direc])
      else:
        O = np.random.normal(p["m"][direc], p["s"][direc])
      # migration model
      Or = O + np.random.normal(0, O*(a[direc]/np.sqrt(O) + b[direc]))

      bt = truth[direc].fill(O, w)
      if np.random.uniform() > e[direc][bt]:
        continue
      mig[direc].fill(O, Or, w)
      br = reco[direc].fill(Or, w)
    reco[direc] = reco[direc] + bkg[direc]

  with open(fname, 'wb') as output:
    for direc in ["A", "B", "C"]:
      model = {}
      model["name"] = direc
      model["mig"] = mig[direc].T()
      model["truth"] = truth[direc]
      model["reco"] = reco[direc]
      model["bkg"] = bkg[direc]
      pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  generateHistograms()

