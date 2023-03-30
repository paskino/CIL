#%%
import dask
from cil.io import ZEISSDataReader
from cil.processors import TransmissionAbsorptionConverter
import os
import time

#%%
dirname = os.path.abspath('C:/Users/ofn77899/Data/walnut/valnut/valnut_2014-03-21_643_28/tomo-A')

reader = ZEISSDataReader(file_name=os.path.join(dirname, 'valnut_tomo-A.txrm'))
#%%
t0 = time.time()
data = reader.read()
t1 = time.time()
print("Time to read data: ", t1-t0)

# from cil.utilities import dataexample
# data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

#%%
neglog = TransmissionAbsorptionConverter()
neglog.set_input(data)

t0 = time.time()
data_trans = neglog.get_output()
t1 = time.time()

print("Time to convert data processor: ", t1-t0)

#%%
from dask import array as da
import numpy as np

ddata = da.from_array(data)
#%% beer-lambert
t0 = time.time()
tmp = np.log(ddata)
np.negative(tmp, out=tmp)
t1 = time.time()
print("Time to define data pipeline: ", t1-t0)

#%%

# slice1 = data[0]
# print (slice1.geometry)
# data2 = data * 0
# data2[0] = slice1

# from cil.utilities.display import show2D
# show2D([data[1], data2[1]], title=['data', 'data2'], cmap='inferno', num_cols=2)

#%%
t0 = time.time()
arr_out = tmp.compute()
t1 = time.time()
print("Time to convert data dask: ", t1-t0)


# %%
labels = data.dimension_labels
print (labels)
print ("idx", labels.index('angle'))
# %%
