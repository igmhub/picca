

from picca.fitter import broadband
import scipy as sp
from scipy import random
from picca.fitter.utils import L

index=sp.arange(2500)

rt = index%50
rp = (index-rt)//50

rt=4*(rt+.5)
rp=4*(rp+.5)

r=sp.sqrt(rt**2+rp**2)

mu = rp/r

r0=100.
ir = sp.arange(4)
ell = 2*sp.arange(4)
data = r*0

for i in ir:
    for l in ell:
        data += (r0/r)**i*L(mu,l)

noise = random.normal(size=len(data))*0.01
data += noise

class d:
    def __init__(self):
        return

da = d()
cuts = (r>10) & (r<180)
da.rt=rt
da.rp=rp

da.cuts = cuts
da.ico = sp.diag(r[cuts]*0+1.)
data=data[cuts]
bb = broadband.model(da,0,3,1,0,6,2)
