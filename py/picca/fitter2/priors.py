
prior_dic = {}

def gaussian(pars, prior_pars=None, name=None):
    mu = prior_pars[0]
    sigma = prior_pars[1]
    par = pars[name]
    return (par-mu)**2/sigma**2
