#!/usr/bin/env python

import copy
import ROOT
from Unfolder.Histogram import H1D, H2D
import numpy as np

'''
Generate unfolding factors.
'''
def generateHistograms(fname = "out_ttallhad_psrw_Syst.root", variable = "mttCoarse"):
  Nev = 100000
  L = 1.0
  f = ROOT.TFile.Open(fname, "recreate")
  f.mkdir("nominal")
  f.mkdir("aMcAtNloHerwigppEvtGen")
  f.mkdir("PowhegHerwigppEvtGen")

  # number of truth bins
  Nt = 5
  # number of reco bins
  Nr = 10

  e = {}
  b = {}
  a = {}
  b = {}
  e["nominal"] = [0.05 + x*0.01 for x in range(0, Nt)]
  e["aMcAtNloHerwigppEvtGen"] = [0.07 - 0.01*x for x in range(0, Nt)]
  e["PowhegHerwigppEvtGen"] = [0.06 + 0.005*x for x in range(0, Nt)]

  b["nominal"] = [0.10 - 0.02*x**2 for x in range(0, Nr)]
  b["aMcAtNloHerwigppEvtGen"] = [0.12 - 0.02*x**2 for x in range(0, Nr)]
  b["PowhegHerwigppEvtGen"] = [0.15 - 0.01*x**2 for x in range(0, Nr)]

  a["nominal"] = 0.5
  a["aMcAtNloHerwigppEvtGen"] = 0.3
  a["PowhegHerwigppEvtGen"] = 0.7
  b["nominal"] = 0.1
  b["aMcAtNloHerwigppEvtGen"] = 0.05
  b["PowhegHerwigppEvtGen"] = 0.08

  p = {}
  p["l"] = {}
  p["m"] = {}
  p["s"] = {}
  p["l"]["nominal"] = 1/0.8
  p["l"]["aMcAtNloHerwigppEvtGen"] = 1/0.5
  p["l"]["PowhegHerwigppEvtGen"] = 1/0.6
  p["m"]["nominal"] = 1.0
  p["m"]["aMcAtNloHerwigppEvtGen"] = 1.1
  p["m"]["PowhegHerwigppEvtGen"] = 1.05
  p["s"]["nominal"] = 0.2
  p["s"]["aMcAtNloHerwigppEvtGen"] = 0.3
  p["s"]["PowhegHerwigppEvtGen"] = 0.2

  truth = {}
  reco = {}
  mig = {}
  bkg = {}
  for direc in ["nominal", "aMcAtNloHerwigppEvtGen", "PowhegHerwigppEvtGen"]:
    truth[direc] = H1D(np.zeros(Nt))
    mig[direc] = H2D(np.zeros((Nr, Nt)))
    reco[direc] = H1D(np.zeros(Nr))
    bkg[direc] = H1D(np.zeros(Nr))
    for k in range(0, Nev):
      O = 0
      if np.random.uniform() > 0.2:
        O = np.random.exponential(p["l"][direc])
      else:
        O = np.random.normal(p["m"][direc], p["s"][direc])
      # migration model
      Or = O + np.random.normal(0, O*(a[direc]/np.sqrt(O) + b[direc]))

      bt = truth[direc].fill(O, 1.0/L)
      if np.random.uniform() > e[direc][bt]:
        continue
      br = reco[direc].fill(Or, 1.0/L)
      mig[direc].fill(Or, O, 1.0/L)
    f.cd(direc)
    mig[direc].T().toROOT("%s" % ("unfoldMigRecoPart_%s_cat2b2HTTmasscut" % variable)).Write()
    truth[direc].toROOT("%s" % ("unfoldPart_%s_cat2b2HTTmasscut" % variable)).Write()
    reco[direc].toROOT("%s" % ("unfoldReco_%s_cat2b2HTTmasscut" % variable)).Write()
    bkg[direc].toROOT("%s" % ("unfoldRecoNotPart_%s_cat2b2HTTmasscut" % variable)).Write()
    f.Write()

if __name__ == "__main__":
  generateHistograms()

