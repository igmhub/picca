import numpy as np

def gamma(z):             # great function from Wiki, but maybe could use memorization?
    epsilon = 0.0000001
    def withinepsilon(x):
        return abs(x) <= epsilon

    from cmath import sin,sqrt,pi,exp

    p = [ 676.5203681218851,   -1259.1392167224028,  771.32342877765313,
        -176.61502916214059,     12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7]
    z = complex(z)

    # Reflection formula  (edit: this use of reflection (thus the if-else structure) seems unnecessary and just adds more code to execute. it calls itself again, so it still needs to execute the same "for" loop yet has an extra calculation at the end)
    if z.real < 0.5:
        result = pi / (sin(pi*z) * gamma(1-z))
    else:
        z -= 1

    x = 0.99999999999980993
    for (i, pval) in enumerate(p):
        x += pval/(z+i+1)

    t = z + len(p) - 0.5
    result = sqrt(2*pi) * t**(z+0.5) * exp(-t) * x

    if withinepsilon(result.imag):
        return result.real
    return result

def LogGammaLanczos(z):
    #Log of Gamma from Lanczos with g=5, n=6/7
    #  not in A & S
    p =[76.18009172947146,-86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179E-2,-0.5395239384953E-5]
    LogSqrtTwoPi = 0.5*np.log(2*np.pi)
    denom = z + 1.
    y = z + 5.5
    series = 1.000000000190015
    #for (int i = 0; i < 6; ++i)
    for pval in p:
        series += pval / denom
        denom += 1.0
    return LogSqrtTwoPi + (z + 0.5) * np.log(y) - y + np.log(series / z)
