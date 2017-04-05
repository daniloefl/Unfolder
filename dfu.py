
import pymc3 as pm
import numpy as np
import seaborn as sns 
import theano
import theano.tensor

import matplotlib.pyplot as plt

theano.config.compute_test_value = 'warn'

class dfu:
  def __init__(self, bkg, data, mig, eff):
    self.bkg = bkg
    self.data = data
    self.mig = mig # migration matrix
    self.eff = eff
    self.response = np.zeros((len(data), len(data)))
    self.response_noeff = np.zeros((len(data), len(data)))
    for i in range(0, len(data)):
      rsum = 0.0
      for j in range(0, len(data)):
        rsum += self.mig[i, j]
      for j in range(0, len(data)):
        self.response[i, j] = self.mig[i, j]/rsum*eff[i]  # P(r|t) = P(t, r)/P(t) = Mtr*eff(t)/sum_k=1^Nr Mtk
        self.response_noeff[i, j] = self.mig[i, j]/rsum

  def asMat(self, x):
    return np.asarray(x,dtype=theano.config.floatX)

  def test(self):
    var_bkg = theano.shared(value = self.asMat(self.bkg))
    var_bkg.reshape((len(self.bkg), 1))
    var_response = theano.shared(value = self.asMat(self.response))
    T = theano.tensor.dvector()
    R = theano.tensor.dot(T, var_response)
    # + var_bkg
    print "Bkg.: ", self.bkg
    print theano.pp(var_bkg)
    print "RT: ", self.response
    print theano.pp(var_response)
    print "T: ", T
    print theano.pp(T)
    print "R: ", R
    print theano.pp(R)
    f = theano.function([T], R)
    test = [1.]
    test.extend([0.]*(len(self.bkg)-1))
    print "f:", f(test)

  def run(self):
    self.model = pm.Model()
    with self.model:
      self.T = pm.Uniform('Truth', 0.0, 5.0e4, shape = (len(self.data))) # prior -- pi(T)
      self.var_bkg = theano.shared(value = self.asMat(self.bkg))
      self.var_bkg.reshape((len(self.bkg), 1))
      self.var_response = theano.shared(value = self.asMat(self.response))
      self.R = theano.tensor.dot(self.T, self.var_response) + self.var_bkg
      self.U = pm.Poisson('U', mu = self.R, observed = self.data, shape = (len(self.data), 1))

  def sample(self, N):
    with self.model:
      start = pm.find_MAP()
      step = pm.NUTS(state = start)
      self.trace = pm.sample(N, step, start = start)
      pm.summary(self.trace)

      from scipy import stats
      self.hunf = np.zeros(len(self.data))
      self.hunf_mode = np.zeros(len(self.data))
      self.hunf_err = np.zeros(len(self.data))
      for i in range(0, len(self.data)):
        self.hunf[i] = np.mean(self.trace.Truth[:, i])
        self.hunf_mode[i] = stats.mode(self.trace.Truth[:, i])[0][0]
        self.hunf_err[i] = np.std(self.trace.Truth[:, i])

  def plotMarginal(self, fname):
    fig = plt.figure(figsize=(10, 20))
    for i in range(0, len(self.data)):
      ax = fig.add_subplot(len(self.data), 1, i+1, title='Truth')
      sns.distplot(self.trace.Truth[:, i], kde = True, hist = True, label = "Truth bin %d" % i, ax = ax)
      ax.set_title("Bin %d value" % i)
      ax.set_ylabel("Probability")
    plt.xlabel("Truth bin value")
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

