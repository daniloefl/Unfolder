#!/usr/bin/env python

import copy
import ROOT
from Unfolder.Histogram import H1D, H2D
import numpy as np

'''
Generate unfolding factors.
'''
def generateHistograms(fname = "out_ttallhad_psrw_Syst.root"):
  v1 = "mttAsymm"
  v2 = "mttCoarse"

  Nev = 500000
  L = 36.1*10.0
  f = ROOT.TFile.Open(fname, "recreate")
  f.mkdir("nominal")
  f.mkdir("aMcAtNloHerwigppEvtGen")
  f.mkdir("PowhegHerwigppEvtGen")

  # number of truth bins
  Nt = 10
  # number of reco bins
  Nr = 20
  Nr2 = 10

  e = {}
  b = {}
  a = {}
  b = {}
  e["nominal"] = [0.20 for x in range(0, Nt)]
  e["aMcAtNloHerwigppEvtGen"] = [0.21 for x in range(0, Nt)]
  e["PowhegHerwigppEvtGen"] = [0.22 for x in range(0, Nt)]

  a["nominal"] = 0.20
  a["aMcAtNloHerwigppEvtGen"] = 0.25
  a["PowhegHerwigppEvtGen"] = 0.22
  b["nominal"] = 0.01
  b["aMcAtNloHerwigppEvtGen"] = 0.011
  b["PowhegHerwigppEvtGen"] = 0.009

  p = {}
  p["l"] = {}
  p["m"] = {}
  p["s"] = {}
  p["l"]["nominal"] = 1/0.5
  p["l"]["aMcAtNloHerwigppEvtGen"] = 1/0.51
  p["l"]["PowhegHerwigppEvtGen"] = 1/0.52
  p["m"]["nominal"] = 4.0
  p["m"]["aMcAtNloHerwigppEvtGen"] = 4.02
  p["m"]["PowhegHerwigppEvtGen"] = 4.05
  p["s"]["nominal"] = 1.0
  p["s"]["aMcAtNloHerwigppEvtGen"] = 1.05
  p["s"]["PowhegHerwigppEvtGen"] = 1.02

  truth = {}
  reco = {}
  mig = {}
  bkg = {}

  truth2 = {}
  reco2 = {}
  mig2 = {}
  bkg2 = {}
  for direc in ["nominal", "aMcAtNloHerwigppEvtGen", "PowhegHerwigppEvtGen"]:
    truth[direc] = H1D(np.zeros(Nt))
    truth[direc].x += 0.5
    mig[direc] = H2D(np.zeros((Nt, Nr)))
    mig[direc].x += 0.5
    mig[direc].y += 0.5
    reco[direc] = H1D(np.zeros(Nr))
    reco[direc].x += 0.5
    bkg[direc] = H1D(10*10.0/L*np.ones(Nr))
    bkg[direc].x += 0.5
    for i in range(0, Nr): bkg[direc].err[i] = 0

    truth2[direc] = H1D(np.zeros(Nt))
    truth2[direc].x += 0.5
    mig2[direc] = H2D(np.zeros((Nt, Nr2)))
    mig2[direc].x += 0.5
    mig2[direc].y += 0.5
    reco2[direc] = H1D(np.zeros(Nr2))
    reco2[direc].x += 0.5
    bkg2[direc] = H1D(10*10.0/L*np.ones(Nr2))
    bkg2[direc].x += 0.5
    for i in range(0, Nr2): bkg2[direc].err[i] = 0
    for k in range(0, Nev):
      O = 0
      if np.random.uniform() > 0.20:
        O = np.random.exponential(p["l"][direc])
      else:
        O = np.random.normal(p["m"][direc], p["s"][direc])
      #if O > Nt: continue
      # migration model
      Or = O + np.random.normal(0, O*(a[direc]/np.sqrt(O) + b[direc]))

      bt = truth[direc].fill(O, 1.0/L)
      bt2 = truth2[direc].fill(O, 1.0/L)
      if np.random.uniform() > e[direc][bt]:
        continue
      mig[direc].fill(O, Or, 1.0/L)
      mig2[direc].fill(O, Or, 1.0/L)
      br = reco[direc].fill(Or, 1.0/L)
      br2 = reco2[direc].fill(Or, 1.0/L)
    reco[direc] = reco[direc] + bkg[direc]
    reco2[direc] = reco2[direc] + bkg2[direc]
    f.cd(direc)
    mig[direc].T().toROOT("%s" % ("unfoldMigRecoPart_%s_cat2b2HTTmasscut" % v1)).Write()
    truth[direc].toROOT("%s" % ("unfoldPart_%s" % v1)).Write()
    reco[direc].toROOT("%s" % ("unfoldReco_%s_cat2b2HTTmasscut" % v1)).Write()
    bkg[direc].toROOT("%s" % ("unfoldRecoNotPart_%s_cat2b2HTTmasscut" % v1)).Write()
    mig2[direc].T().toROOT("%s" % ("unfoldMigRecoPart_%s_cat2b2HTTmasscut" % v2)).Write()
    truth2[direc].toROOT("%s" % ("unfoldPart_%s" % v2)).Write()
    reco2[direc].toROOT("%s" % ("unfoldReco_%s_cat2b2HTTmasscut" % v2)).Write()
    bkg2[direc].toROOT("%s" % ("unfoldRecoNotPart_%s_cat2b2HTTmasscut" % v2)).Write()
    f.Write()

if __name__ == "__main__":
  generateHistograms()

