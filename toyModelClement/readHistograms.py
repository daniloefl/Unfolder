
import copy
from Unfolder.Histogram import H1D, H2D
import numpy as np
import pickle
import json

'''
Read unfolding factors, deciding whether to use the JSON format or the PKL format based on the file name.
'''
def getHistograms(fname = "toyModelClement/ModelClement.json", direc = "A"):
  if '.json' in fname:
    return getHistogramsFromJson(fname, direc)
  return getHistogramsFromPkl

'''
Read unfolding factors from JSON format
'''
def getHistogramsFromJson(fname, direc):
  parsed_json = json.load(open(fname))

  # initial (real) data --> this actually depends on the model used
  # it will be changed below
  recoWithFakes = H1D(np.asarray(parsed_json["Data"]))

  # this is the response matrix, which is response = eff(truth = i) * P(reco = j|truth = i) = eff(truth = i) * P(t = i, r = j) / P(t = i)
  # the migration matrix is eff(truth = i) * P(t=i,r=j) = response(t = i, r = j) * P(t = i)
  if direc == "A":
    resp = H2D(np.asarray(parsed_json["Nominal"]["Mig"], dtype = np.float64))
    resp.err = np.zeros(shape = resp.val.shape)
    resp.err_up = np.zeros(shape = resp.val.shape)
    resp.err_dw = np.zeros(shape = resp.val.shape)
  else:
    resp = H2D(np.asarray(parsed_json["ModelVars"]["resolution"]["Variation"]["Mig"], dtype = np.float64))
    resp.err = np.zeros(shape = resp.val.shape)
    resp.err_up = np.zeros(shape = resp.val.shape)
    resp.err_dw = np.zeros(shape = resp.val.shape)

  # input assumed to have reco in Y axis and truth in X
  # from here on
  Ntruth = resp.val.shape[0]
  Nreco = resp.val.shape[1]

  # no background for now
  if direc == "A":
    bkg = H1D(np.asarray(parsed_json["Nominal"]["Bkgs"]["Bkg"], dtype = np.float64))
  else:
    bkg = H1D(np.asarray(parsed_json["ModelVars"]["resolution"]["Variation"]["Bkgs"]["Bkg"], dtype = np.float64))


  #print("Response matrix")
  #print(resp.val)

  # sum of response matrix in the reco rows gives the efficiency
  eff = H1D(np.zeros(Ntruth, dtype = np.float64))
  for itruth in range(Ntruth):
    for ireco in range(Nreco):
      eff.val[itruth] += resp.val[itruth, ireco]

  #print("Efficiency:")
  #print(eff.val)

  # get response matrix assuming efficiency = 1
  resp_noeff = copy.deepcopy(resp)
  for itruth in range(Ntruth):
    for ireco in range(Nreco):
      resp_noeff.val[itruth, ireco] /= eff.val[itruth]

  #print("Response matrix/eff: ")
  #print(resp_noeff.val)

  truth_a = []
  for i in range(Ntruth):
    truth_a.append(parsed_json["ModelVars"]["truthbin%d" % i]["InitialValue"])
  truth = H1D(np.asarray(truth_a, dtype = np.float64))
  truth.err = np.zeros(shape = truth.val.shape)
  truth.err_up = np.zeros(shape = truth.val.shape)
  truth.err_dw = np.zeros(shape = truth.val.shape)
  #print("Truth: ", truth.val)


  # get migration matrix, by multiplying by P(truth=i)
  mig = copy.deepcopy(resp)
  for itruth in range(Ntruth):
    for ireco in range(Nreco):
      mig.val[itruth, ireco] *= truth.val[itruth]

  #print("Migration matrix: ")
  #print(mig.val)

  recoWithFakes_a = []
  for ireco in range(Nreco):
    recoWithFakes_a.append(0)
    for itruth in range(Ntruth):
      recoWithFakes_a[-1] += resp.val[itruth, ireco]*truth.val[itruth]
  recoWithFakes = H1D(np.asarray(recoWithFakes_a, dtype = np.float64))
  for ireco in range(Nreco):
    recoWithFakes.err[ireco] = 0

  tr_1dtruth = mig.project('x')
  tr_1dtruth.err = np.zeros(shape = tr_1dtruth.val.shape)
  tr_1dtruth.err_up = np.zeros(shape = tr_1dtruth.val.shape)
  tr_1dtruth.err_dw = np.zeros(shape = tr_1dtruth.val.shape)
  nrt = truth - tr_1dtruth

  ones = H1D(np.ones(len(nrt.val)))
  ones.err = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.err_up = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.err_dw = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.x = copy.deepcopy(nrt.x)
  ones.x_err = copy.deepcopy(nrt.x_err)
  eff = ones + nrt.divideBinomial(truth)*(-1.0)
  #eff = mig.project('x').divideBinomial(truth)
  return [truth, recoWithFakes, bkg, mig, eff, nrt]
  
'''
Read unfolding factors from pkl format as generated by toyModel/generateHistograms.py
'''
def getHistogramsFromPkl(fname = "histograms.pkl", direc = "A"):
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
  ones.err_up = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.err_dw = copy.deepcopy(np.zeros(len(nrt.val)))
  ones.x = copy.deepcopy(nrt.x)
  ones.x_err = copy.deepcopy(nrt.x_err)
  eff = ones + nrt.divideBinomial(truth)*(-1.0)
  #eff = mig.project('x').divideBinomial(truth)

  return [truth, recoWithFakes, bkg, mig, eff, nrt]

