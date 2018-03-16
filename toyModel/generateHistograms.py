#!/usr/bin/env python

import copy
from Unfolder.Histogram import H1D, H2D
import numpy as np
import pickle

'''
Sample the distribution P(m) = (1- m/sqrts)^6 / (m/sqrts)^4.8 for m > minMass, or 0 otherwise.
x is defined as m/sqrts.
dP/dx = (- 6 (1-x)^5 x^4.8 - 4.8 (1 - x)^6 x^3.8)/x^9.6
If dP/dx = 0, x \in {-4, 0, 1}.
As x = -4 is unphysical, P(x) has extrema at m = 0 and m = sqrts.
'''
class GenerateSample:
  def __init__(self, minMass = 0.5, sqrts = 7.0):
    self.x = 2.0/sqrts
    self.sigma = 1.0/sqrts
    self.sqrts = sqrts
    self.minMass = minMass

  '''
  Proposition function is a Gaussian centered at the previous point.
  '''
  def g(self, x_old):
    return np.random.normal(x_old, self.sigma)

  '''
  PDF shape to produce.
  '''
  def f(self, x):
    if x < self.minMass/self.sqrts: return 0
    return (1 - x)**6 / (x**4.8)

  '''
  Get a new sample of the pdf given by the normalised f.
  '''
  def sample(self):
    accepted = False
    x_old = self.x
    x_new = self.g(x_old)
    alpha = self.f(x_new)/self.f(x_old)
    u = np.random.uniform(0, 1)
    if u <= alpha:
      self.x = x_new
      accepted = True
    else:
      self.x = x_old
    return self.x*self.sqrts
  

'''
Generate unfolding factors.
'''
def generateHistograms(fname = "histograms.pkl"):
  Nev = 400000
  wL = 100.0 # generate wL times more events than we expect in data

  # number of truth bins
  xt = 0.5*np.exp(np.arange(0, 12, 1)*0.15)
  xt_err = np.diff(xt)*0.5
  xt = xt[:-1]
  xt += xt_err
  Nt = len(xt)

  # number of reco bins
  xf = 0.5*np.exp(np.arange(0, 24, 1)*0.15*0.5)
  xf_err = np.diff(xf)*0.5
  xf = xf[:-1]
  xf += xf_err
  Nr = len(xf)

  e = {}
  b = {}
  a = {}
  b = {}
  e["A"] = [0.40 for x in range(0, Nt)]
  e["B"] = [0.42 for x in range(0, Nt)]
  e["C"] = [0.38 for x in range(0, Nt)]

  a["A"] = 0.20
  a["B"] = 0.25
  a["C"] = 0.15
  b["A"] = 0.02
  b["B"] = 0.03
  b["C"] = 0.01

  #a["A"] = 0
  #a["B"] = 0
  #a["C"] = 0
  #b["A"] = 0
  #b["B"] = 0
  #b["C"] = 0

  gs = GenerateSample(minMass = 0.5, sqrts = 7)

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
    bkg[direc] = H1D(np.zeros(Nr))
    bkg[direc].x = xf
    bkg[direc].x_err = xf_err
    for i in range(0, Nr): bkg[direc].err[i] = 0

  for k in range(0, int(Nev*wL)):
    O = gs.sample()
    w = 1.0/wL

    # implement overflow bin
    O_over = O
    if O_over > xt[-1]+xt_err[-1]:
      O_over = xt[-1]
    # guarantee same truth histogram for all
    # do not histogram events below lowest bin (ie: do not plot underflow)
    # we assume this is the boundary of the fiducial region
    # but use those low O events in the migration model, as they can be smeared in
    if O_over > xt[0]-xt_err[0]:
      for direc in ["A", "B", "C"]:
        bt = truth[direc].fill(O_over, w)

    # migration model
    for direc in ["A", "B", "C"]:
      dm = O*(a[direc]/np.sqrt(O) + b[direc])
      Or = O + np.random.normal(0, dm)

      # if reco-level bin is below lowest bin, reject it
      # this effect is considered in the efficiency later
      if Or < xf[0]-xf_err[0]:
        continue
      # implement overflow bin
      if Or > xf[-1]+xf_err[-1]:
        Or = xf[-1]

      # implement efficiency
      if np.random.uniform(0, 1) > e[direc][bt]:
        continue
      mig[direc].fill(O_over, Or, w)
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

