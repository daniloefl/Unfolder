
import ROOT
from Unfolder.Histogram import H1D, H2D
import numpy as np

'''
Read unfolding factors from a ROOT file
'''
def getHistograms(fname = "out_ttallhad_psrw_Syst.root", direc = "nominal"):
  f = ROOT.TFile.Open(fname)
  truth = H1D(f.Get("%s/%s" % (direc, "unfoldPart_mttCoarse_cat2b2HTTmasscut")))
  reco = H1D(f.Get("%s/%s" % (direc, "unfoldReco_mttCoarse_cat2b2HTTmasscut")))
  # input assumed to have reco in X axis and truth in Y, so transpose it to the truth in X axis convention
  mig = H2D(f.Get("%s/%s" % (direc, "unfoldMigRecoPart_mttCoarse_cat2b2HTTmasscut"))).T()

  bkg = H1D(f.Get("%s/%s" % (direc, "unfoldRecoNotPart_mttCoarse_cat2b2HTTmasscut")))

  tr_1dtruth = mig.project('x')
  nrt = truth - tr_1dtruth
  eff = nrt/truth*(-1.0) + H1D(np.ones(len(nrt.val)))

  return [truth, reco, bkg, mig, eff]

