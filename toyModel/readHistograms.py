
import copy
from Unfolder.Histogram import H1D, H2D
import numpy as np
import pickle

'''
Read unfolding factors from a ROOT file
'''
def getHistograms(fname = "histograms.pkl", direc = "A"):
  m = None
  with open(fname, 'rb') as inp:
    model = pickle.load(inp)
    while model != None:
      if model["name"] != direc:
        model = pickle.load(inp)
        continue
      m = model
      break
  
  truth = m["truth"]
  recoWithFakes = m["reco"]
  # input assumed to have reco in X axis and truth in Y, so transpose it to the truth in X axis convention
  mig = m["mig"].T()

  # fakes
  bkg = m["bkg"]

  tr_1dtruth = mig.project('x')
  nrt = truth - tr_1dtruth

  ones = H1D(np.ones(len(nrt.val)))
  ones.err = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.x = copy.deepcopy(nrt.x)
  ones.x_err = copy.deepcopy(nrt.x_err)
  eff = ones + nrt.divideBinomial(truth)*(-1.0)
  #eff = mig.project('x').divideBinomial(truth)

  return [truth, recoWithFakes, bkg, mig, eff, nrt]

