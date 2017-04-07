
import copy
import ROOT
from Unfolder.Histogram import H1D, H2D
import numpy as np

'''
Read unfolding factors from a ROOT file
'''
def getHistograms(fname = "out_ttallhad_psrw_Syst.root", direc = "nominal"):
  L = 36.1
  f = ROOT.TFile.Open(fname)
  truth = L*H1D(f.Get("%s/%s" % (direc, "unfoldPart_mttCoarse_cat2b2HTTmasscut")))
  recoWithFakes = L*H1D(f.Get("%s/%s" % (direc, "unfoldReco_mttCoarse_cat2b2HTTmasscut")))
  # input assumed to have reco in X axis and truth in Y, so transpose it to the truth in X axis convention
  mig = L*H2D(f.Get("%s/%s" % (direc, "unfoldMigRecoPart_mttCoarse_cat2b2HTTmasscut"))).T()

  # fakes
  bkg = L*H1D(f.Get("%s/%s" % (direc, "unfoldRecoNotPart_mttCoarse_cat2b2HTTmasscut")))
  #bkg = bkg + other bkgs!!! FIXME

  tr_1dtruth = mig.project('x')
  nrt = truth - tr_1dtruth

  ones = H1D(np.ones(len(nrt.val)))
  ones.err = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.x = copy.deepcopy(nrt.x)
  ones.x_err = copy.deepcopy(nrt.x_err)
  eff = ones + nrt.divideBinomial(truth)*(-1.0)
  #eff = mig.project('x').divideBinomial(truth)

  return [truth, recoWithFakes, bkg, mig, eff, nrt]

