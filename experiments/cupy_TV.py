#%%
from cil.utilities import dataexample
from cil.optimisation.functions import TotalVariation
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.utilities.display import show2D

import time

N = 10
num_iter = 1000
isotropic = True

#%%
# get an example image

data = dataexample.CAMERA.get(size=(128, 128))
import numpy as np
geom = data.geometry.copy()
x = geom.allocate(0, backend='cupy')
print (x.backend)
x.fill(data)

#%% create a TotalVariation object
gTV = TotalVariation(max_iteration=num_iter, isotropic=isotropic, backend='cupy')
cTV = TotalVariation(max_iteration=num_iter, isotropic=True, backend='c')
nTV = TotalVariation(max_iteration=num_iter, isotropic=True, backend='numpy')

#%%
fgp = FGP_TV(max_iteration=num_iter, isotropic=isotropic, device='gpu')

#%%
d1 = fgp.proximal(data, tau=1)
t0 = time.time()
for _ in range(N):
    d1 = fgp.proximal(data, tau=1, out=d1)
t1 = time.time()
dt_fgp = t1-t0
print (f"Elapsed time: {dt_fgp:.6f} seconds")

#%%
# d3 = cTV.proximal(x, tau=1)

#%%
tmp = x.geometry.allocate(0, backend='cupy')
tmp.fill(x)
cud2 = gTV.proximal(tmp, tau=1)

t0 = time.time()
for _ in range(N):
    gTV.proximal(tmp, tau=1, out=cud2)
t1 = time.time()
dt_cutv = t1-t0
print (f"Elapsed time: {dt_cutv:.6f} seconds")

#%%
d3 = cTV.proximal(data, tau=1)
t0 = time.time()
for _ in range(N):
    cTV.proximal(data, tau=1, out=d3)
    print ("Iteration ", _)
t1 = time.time()
dt_Ctv = t1-t0
print (f"Elapsed time: {dt_Ctv:.6f} seconds")

#%%
d4 = nTV.proximal(data, tau=1)
t0 = time.time()
for _ in range(N):
    nTV.proximal(data, tau=1, out=d4)
    print ("Iteration ", _)
t1 = time.time()
dt_ntv = t1-t0
print (f"Elapsed time: {dt_ntv:.6f} seconds")

#%%
d2 = cud2.as_array().get()
show2D([d1, d2, d3, d4], title=[f'FGP_TV {dt_fgp/N:.3f}s', f'cu TotalVariation {dt_cutv/N:.3f}s',
                                 f'C TotalVariation {dt_Ctv/N:.3f}s',
                                 f'np TotalVariation {dt_ntv/N:.3f}s',
                                 ], origin='upper', cmap='inferno')
# %%
