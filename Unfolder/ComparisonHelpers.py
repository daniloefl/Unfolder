
import ROOT
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from Histogram import H1D, H2D, plotH1D, plotH2D, plotH1DWithText, plotH2DWithText

'''
To be used to scan TUnfold's object to find the optimal tau value for
the regularisation.
'''
def printLcurve(tunfolder, fname):
  nScan   = 30
  tauMin  = 0.
  tauMax  = 0.
  iBest   = -1
  logTauX = ROOT.TSpline3()
  logTauY = ROOT.TSpline3()
  lCurve  = ROOT.TGraph()
  iBest = tunfolder.ScanLcurve(nScan,tauMin,tauMax,lCurve, logTauX, logTauY)
  tau   = tunfolder.GetTau()
  print "With TUnfold: Lcurve scan chose tau =", tau

  c = ROOT.TCanvas()
  t = ROOT.Double()
  x = ROOT.Double()
  y = ROOT.Double()
  logTauX.GetKnot(iBest,t,x)
  logTauY.GetKnot(iBest,t,y)
  bestLcurve = ROOT.TGraph(1, array.array('d',[x]), array.array('d', [y]))
  lCurve.Draw("AC")
  bestLcurve.SetMarkerColor(ROOT.kRed+2)
  bestLcurve.Draw("*")
  c.Print(fname)

'''
Use this to run the TUnfold unfolding method.
For example:
tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
#printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(0)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff

comparePlot([f_data, f_data - f_bkg, truth, tunfold_result], ["Data", "Data - bkg", "Particle-level", "TUnfold"], luminosity*1e-3, True, "fb/GeV", "plotTUnfold.png")
'''
def getTUnfolder(bkg, mig, data, regMode = ROOT.TUnfold.kRegModeDerivative):
  regularisationMode = 0
  tunfolder = ROOT.TUnfoldDensity(mig.T().toROOT("tunfold_mig"), ROOT.TUnfold.kHistMapOutputVert, regMode, ROOT.TUnfold.kEConstraintArea, ROOT.TUnfoldDensity.kDensityModeeNone)
  dataBkgSub = data - bkg
  tunfolder.SetInput(dataBkgSub.toROOT("data_minus_bkg"))
  return tunfolder

'''
Use this to plot the histograms in listHist with the legends in listLegend.
'''
def comparePlot(listHist, listLegend, f = 1.0, normaliseByBinWidth = True, units = "fb", fname = "comparePlot.png"):
  fig = plt.figure(figsize=(10, 10))
  newListHist = []
  for item in listHist:
    newListHist.append(f*item)
    if normaliseByBinWidth:
      newListHist[-1].overBinWidth()
  sty = ['rv', 'ro', 'b^', 'ro', 'gv', 'ms']
  siz = [12, 10, 8, 6, 4, 2]
  c = 0
  for item in newListHist:
    plt.errorbar(item.x, item.val, item.err**0.5, item.x_err, fmt = sty[c], linewidth=2, label = listLegend[c], markersize=siz[c])
    c += 1
  plt.legend()
  plt.ylabel("Differential cross section ["+units+"]")
  plt.xlabel("Observable")
  plt.tight_layout()
  plt.savefig(fname)
  plt.close()

'''
This returns a migration-corrected object from RooUnfold using the D'Agostini unfolding procedure.
Here is an example of how to use it:
dagostini_mig = getDAgostini(bkg, mig, data)
dagostini_result = dagostini_mig/eff
comparePlot([data, data - bkg, truth, dagostini_result], ["Data", "Data - bkg", "Particle-level", "D'Agostini"], luminosity*1e-3, True, "fb/GeV", "plotDAgostini.png")
'''
def getDAgostini(bkg, mig, data, nIter = 5):
  reco = mig.project('y').toROOT("reco_rp")
  reco.SetDirectory(0)
  #truth = mig.project('x').toROOT("truth_rp")
  #truth.SetDirectory(0)
  m = mig.T().toROOT("m")
  m.SetDirectory(0)
  unf_response = ROOT.RooUnfoldResponse(reco, 0, m)
  dataBkgSub = data - bkg
  dd = dataBkgSub.toROOT("dataBkgSub_dagostini")
  dd.SetDirectory(0)
  dagostini = ROOT.RooUnfoldBayes(unf_response, dd, nIter)
  dagostini_hreco = dagostini.Hreco()
  return H1D(dagostini_hreco)

