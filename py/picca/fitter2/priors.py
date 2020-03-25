
prior_dic = {}

def gaussian(pars, prior_pars=None, name=None):
    ''' Gaussian prior that returns a chi2 '''
    mu = prior_pars[0]
    sigma = prior_pars[1]
    par = pars[name]
    return (par-mu)**2/sigma**2

def gaussian_norm(pars, prior_pars=None, name=None):
    ''' Gaussian prior that returns a normalized likelihood '''
    from numpy import log, pi
    mu = prior_pars[0]
    sigma = prior_pars[1]
    par = pars[name]
    chi2 = (par-mu)**2/sigma**2
    log_lik = -0.5 * log(2 * pi) - log(sigma)
    log_lik -= 0.5 * chi2
    return log_lik