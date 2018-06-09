
# coding: utf-8

# In[ ]:


from __future__ import division
import numpy as np
import glob
import cPickle as pickle
import scipy.linalg as sl
import scipy.special as ss

import enterprise
from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

import libstempo as t2

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import os
import json


# ## Utility functions

# In[ ]:


class JumpProposal(object):
    
    def __init__(self, pta):
        """Set up some custom jump proposals
        
        :param params: A list of `enterprise` parameters
        
        """
        self.params = pta.params
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)
        
        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[p] = slice(ct, ct+size)
            ct += size
            
        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct
            
        self.snames = {}
        for sc in pta._signalcollections:
            for signal in sc._signals:
                self.snames[signal.signal_name] = signal.params
        
    def draw_from_prior(self, x, iter, beta):
        """Prior draw.
        
        The function signature is specific to PTMCMCSampler.
        """
        
        q = x.copy()
        lqxy = 0
        
        # randomly choose parameter
        idx = np.random.randint(0, self.npar)
        
        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[idx] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])
                
        return q, float(lqxy)
    
# utility function for finding global parameters
def get_global_parameters(pta):
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)
    
    gpars = np.unique(filter(lambda x: pars.count(x)>1, pars))
    ipars = np.array([p for p in pars if p not in gpars])
        
    return gpars, ipars

# utility function to get parameter groupings for sampling
def get_parameter_groups(pta):
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names
    
    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if any(gpars):
        groups.extend([[params.index(gp) for gp in gpars]])

    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if p not in gpars]
            if ind:
                groups.extend([ind])
    
    return groups


# ## Read in pulsar data

# In[ ]:


psr_name = 'J1744-1134'
psr = Pulsar('../EPTA_v2.2_git/{0}/{0}.par'.format(psr_name), '../EPTA_v2.2_git/{0}/{0}_all.tim'.format(psr_name))


# In[ ]:





# ## Setup model
# 
# We will add some addition model components that are not part of the base enterprise

# ### 1. Exponential decay function to model "void" in J1713+0747

# In[ ]:


@signal_base.function
def exp_decay(toas, freqs, log10_Amp=-7, t0=54000, log10_tau=1.7):
    t0 *= const.day
    tau = 10**log10_tau * const.day
    wf = - 10**log10_Amp * np.heaviside(toas-t0, 1) * np.exp(-(toas-t0)/tau)
    return wf * (1400/freqs)**2


# ### 2. Yearly DM sinusoid

# In[ ]:


@signal_base.function
def yearly_sinusoid(toas, freqs, log10_Amp=-7, phase=0):

    wf = 10**log10_Amp * np.sin(2*np.pi*const.fyr*toas+phase)
    return wf * (1400/freqs)**2

@signal_base.function
def yearly_sinusoid_basis(toas, freqs):
    
    F = np.zeros((len(toas), 2))
    F[:,0] = np.sin(2*np.pi*toas*const.fyr)
    F[:,1] = np.cos(2*np.pi*toas*const.fyr)
    
    Dm = (1400/freqs)**2

    return F * Dm[:, None], np.repeat(const.fyr, 2)

@signal_base.function
def yearly_sinusoid_prior(f):
    return np.ones(len(f)) * 1e20


# ### 3. DM EQUAD (EQUAD) term that scales like $\nu^{-4}$ (variance remember...)

# In[ ]:


# define DM EQUAD variance function
@signal_base.function
def dmequad_ndiag(freqs, log10_dmequad=-8):
    return np.ones_like(freqs) * (1400/freqs)**4 * 10**(2*log10_dmequad)


# ### 4. SVD timing model basis
# This allows for more stability over standard scaling methods

# In[ ]:


# SVD timing model basis
@signal_base.function
def svd_tm_basis(Mmat):
    u, s, v = np.linalg.svd(Mmat, full_matrices=False)
    return u, np.ones_like(s)

@signal_base.function
def tm_prior(weights):
    return weights * 10**40


# In[ ]:


# define selection by observing backend
selection1 = selections.Selection(selections.by_backend)

# special selection for ECORR only use wideband NANOGrav data
selection2 = selections.Selection(selections.nanograv_backends)

# white noise parameters
#efac = parameter.Uniform(0.5, 10.0)
efac = parameter.Normal(1.0, 0.1)
equad = parameter.Uniform(-10, -4)
ecorr = parameter.Uniform(-10, -4)

# red noise and DM parameters
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# DM turnover parameters
kappa = parameter.Uniform(0,7)
lf0 = parameter.Uniform(-9, -6.5)

# DM exponential parameters
t0 = parameter.Uniform(psr.toas.min()/86400, psr.toas.max()/86400)
log10_Amp = parameter.Uniform(-10, -2)
log10_tau = parameter.Uniform(np.log10(5), np.log10(500))

# DM EQUAD
dmvariance = dmequad_ndiag(log10_dmequad=equad)
dmeq = white_signals.WhiteNoise(dmvariance)

# white noise signals
#ef = white_signals.MeasurementNoise(efac=efac, selection=selection1)
#eq = white_signals.EquadNoise(log10_equad=equad, selection=selection1)
#ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection2)
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad=equad)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr)

# red noise signal
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(pl, components=30)

# DM GP signals (use turnover model for more flexibility)
dm_basis = utils.createfourierdesignmatrix_dm(nmodes=30)
dm_prior = utils.turnover(log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa)
dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm')

# DM exponential model
wf = exp_decay(log10_Amp=log10_Amp, t0=t0, log10_tau=log10_tau)
dmexp = deterministic_signals.Deterministic(wf, name='exp')

# DM sinusoid model
ys_prior = yearly_sinusoid_prior()
ys_basis = yearly_sinusoid_basis()
dmys = gp_signals.BasisGP(ys_prior, ys_basis, name='s1yr')

# timing model
basis = svd_tm_basis()
prior = tm_prior()
#tm = gp_signals.BasisGP(prior, basis)
tm = gp_signals.TimingModel()

# full model
s = ef + eq + rn + dmgp + tm + dmys
#if 'NANOGrav' in psr.flags['pta']:
#    s += ec
if psr.name == 'J1713+0747':
    s += dmexp

# set up PTA of one
pta = signal_base.PTA([s(psr)])


# In[ ]:





# In[ ]:


# dimension of parameter space
x0 = np.hstack(p.sample() for p in pta.params)
ndim = len(x0)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.1**2)

## change initial jump size for tau
#idx = pta.param_names.index('J1713+0747_exp_t0')
#cov[idx, idx] = 100

# parameter groupings
groups = get_parameter_groups(pta)

outdir = 'chains/{0}_noise/'.format(psr_name)
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, 
                 outDir=outdir, resume=True)
np.savetxt(outdir+'/pars.txt', map(str, pta.param_names), fmt='%s')

# add prior draw to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 15)


# In[ ]:


N = 1000000
sampler.sample(x0, N, SCAMweight=35, AMweight=10, DEweight=50)


# In[ ]:


#chain = np.loadtxt('chains/J1713+0747_standard/chain_1.txt')
#burn = int(0.25*chain.shape[0])

chain = np.loadtxt('chains/{0}_noise/chain_1.txt'.format(psr_name))
burn = int(0.25*chain.shape[0])
pars = pta.param_names

#chain2 = np.loadtxt('chains/J0437-4715_tm/chain_1.txt')
#burn2 = int(0.25*chain2.shape[0])


# In[ ]:


def make_noise_files(psrname, chain, pars, outdir='noisefiles/'):
    x = {}
    for ct, par in enumerate(pars):
        x[par] = np.median(chain[:, ct])

    os.system('mkdir -p {}'.format(outdir))
    with open(outdir + '/{}_noise.json'.format(psrname), 'w') as fout:
        json.dump(x, fout, sort_keys=True, indent=4, separators=(',', ': '))


# In[ ]:


make_noise_files(psrname=psr_name,chain=chain,pars=pars)


# In[ ]:





# In[ ]:




