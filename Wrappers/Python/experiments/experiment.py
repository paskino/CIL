#%%

from cil.utilities import dataexample
from cil.optimisation.algorithms import PDHG
from cil.optimisation.functions import L2NormSquared, MixedL21Norm, IndicatorBox,\
    ZeroFunction, BlockFunction
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.plugins.tigre import ProjectionOperator
from cil.plugins.astra import ProjectionOperator as POA
from cil.plugins.astra import FBP
from cil.utilities.display import show2D
from cil.io import NEXUSDataWriter
def extract_single_line(data, **kwargs):
    try:
        geometry = data.geometry
        possible_dimensions = geometry.dimension_labels
    except AttributeError:
        possible_dimensions = ['x','y','z']
    i = 0
    for k,v in kwargs.items():
        if k not in possible_dimensions:
            raise ValueError(f'Unexpected key {k}, not in {possible_dimensions}')
        sliceme = {k:v}
        if i > 0:
            data_plot = data_plot.get_slice(**sliceme)
        else:
            data_plot = data.get_slice(**sliceme)
        i += 1
    return data_plot.as_array()

def line_plot(data, line_coords=None, label=None, title=None, color=None, size=(15,15)):
    '''Creates a 1D plot of data given some coordinates

    Parameters
    ----------
    data : ImageData, AcquisitionData, generic DataContainer or a list of such
           data from which to extract the line plot. 
    line_coords : tuple, list of tuples
        Specifies the line plot to show. 
        For 3D datacontainers two slices: [(direction0, index0),(direction1, index1)]. 
        For 4D datacontainers three slices: [(direction0, index0),(direction1, index1),(direction2, index2)].
    label : string or list of strings, optional
        Label for the line plot. If passed a list of data, label must be a list of matching length.
    title : string, optiona
        Title for the whole plot
    color : string or list of strings, optional
        Color for each line in the plot. If passed a list of data, color must be a list of matching length.
    size : tuple or list of ints, optional
        Specifies the size of the plot


    Example Usage:
    --------------

    line_plot( [gt, fbp_recon, algo1.solution * A1.norm()], 
                label=['Ground Truth', 'FBP', 'PDHG + TV + nn'],
                line_coords=(('horizontal_x',64), ('vertical',64)), 
                title=f'Comparison alpha {alpha}',
                color=('cyan', 'purple', 'orange'), 
                size=(15,9)
           )

    '''
    kwargs = {}
    for i, el in enumerate(line_coords):
        kwargs[el[0]] = el[1]
    data_plot = []

    if issubclass(data.__class__, (list, tuple)):
        for el in data:
            data_plot.append( extract_single_line(el, **kwargs))
            axes_labels = list(el.dimension_labels)
    else:
        data_plot.append( extract_single_line(data, **kwargs))
        axes_labels = list(el.dimension_labels)

    fig, ax = plt.subplots(1, 1, figsize=size)
    for i,el in enumerate(data_plot):
        try:
            if color is None:
                color = [None for _ in data]
            ax.plot(el, color=color[i], label=label[i])
            
        except:
            ax.plot(el, label=label, color=color)
    ax.set_title(title)

    xaxis = []
    for i, el in enumerate(axes_labels):
        if el not in kwargs.keys():
            xaxis.append(el)
    ax.set_xlabel(xaxis[0])
    ax.set_ylabel('Pixel value')

    plt.legend()
    plt.show()


#%%
data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
gt3d = dataexample.SIMULATED_SPHERE_VOLUME.get()

print (data.geometry)
print (gt3d.geometry)
# data = data.get_slice(vertical='centre')
data.log(out=data)
data *= -1
data.reorder('astra')

# gt = gt3d.get_slice(vertical='centre')
gt = gt3d
#%%
fbp_recon = FBP(data.geometry.get_ImageGeometry(), data.geometry)(data)
show2D([fbp_recon, gt], cmap='plasma')

#%%
alpha = 0.12
# %%

# problem 1 original pixel size
ig1 = data.geometry.get_ImageGeometry()
# A1 = ProjectionOperator(ig1, data.geometry)
A1 = POA(ig1, data.geometry)
g1 = GradientOperator(ig1)
print ("norm A1", A1.norm())
print ("norm g1", g1.norm())


'''
Should be looking at F(beta * x), i.e. F or a rescaled dataset
'''
from cil.optimisation.functions import Function
class ScaledArgFunction(Function):
    def __init__(self, function, scalar):
        self.function = function
        self.scalar = scalar

    def __call__(self, x):
        x *= self.scalar
        ret = self.function(x)
        x /= self.scalar
        return ret

    def gradient(self, x, out=None):
        x *= self.scalar
        should_return = False
        if out is None:
            out = self.function.gradient(x)
            should_return = True
        else:
            self.function.gradient(x, out=out)
        out *= self.scalar
        x /= self.scalar
        if should_return:
            return out

    def proximal(self, x, tau, out=None):
        # eq 6.6 of https://archive.siam.org/books/mo25/mo25_ch6.pdf
        should_return = False
        if out is None:
            out = x * 0
            should_return = True
        x *= self.scalar
        self.function.proximal( x, tau * self.scalar**2, out=out)
        x /= self.scalar
        out /= self.scalar
        if should_return:
            return out
    def convex_conjugate(self, x):
        # https://en.wikipedia.org/wiki/Convex_conjugate#Table_of_selected_convex_conjugates
        x /= self.scalar
        ret = self.function.convex_conjugate(x)
        x *= self.scalar
        return ret

#%%        

F = BlockFunction(
    # A1.norm() * L2NormSquared(b=data),
    # g1.norm() * MixedL21Norm()
    ScaledArgFunction( L2NormSquared(b=data/A1.norm()), A1.norm()),
    ScaledArgFunction( MixedL21Norm(), g1.norm())
)


K = BlockOperator(
    (1/A1.norm()) * A1, 
    (alpha/g1.norm()) * g1
    )


# high alpha + Indicator box -> zero solution
# small alpha —> Ground truth
G = IndicatorBox(lower=0)

# high alpha + ZeroFunction -> high alpha should converge to the mean value of your data 
# G = ZeroFunction()

algo1 = PDHG(f=F, g=G, operator=K, max_iteration=1000, update_objective_interval=100,log_file="algo1.nxs")

#%%
# algo1.max_iteration += 1000
algo1.run(1000)
# NEXUSDataWriter(algo1.solution, file_name='algo1.nxs').write()
#%%
show2D([algo1.solution * A1.norm(), gt], title=[f'PDHG rescaled {alpha}', 'Ground Truth'],
       cmap='plasma', fix_range=True)

#%%
import matplotlib.pyplot as plt

plt.plot(gt.as_array()[64,64,:], label='GT')
plt.plot(algo1.solution.as_array()[64,64,:]*(A1.norm()), label=f'PDHG {alpha}', color='purple')
plt.plot(fbp_recon.as_array()[64,64,:], label='FBP', color='r')
plt.legend()
plt.show()

#%%

#%%
line_plot([gt, fbp_recon, algo1.solution * A1.norm()], 
           label=['Ground Truth', 'FBP', 'PDHG + TV + nn'],
           line_coords=(('horizontal_x',64), ('vertical',64)), 
           title=f'Comparison alpha {alpha}',
           color=('cyan', 'purple', 'orange'), 
           size=(15,9))
# %%

# problem 2 rescale into alpha

alpha_tilde = alpha * ((A1.norm()**2) / g1.norm())

F2 = BlockFunction(
    L2NormSquared(b=data),
    MixedL21Norm()
)


K2 = BlockOperator(
    A1, 
    alpha_tilde * g1
    )

G2 = IndicatorBox(lower=0)
# G2 = ZeroFunction()

algo2 = PDHG(f=F2, g=G2, operator=K2, max_iteration=1000, update_objective_interval=100,log_file="algo1.nxs")

#%%
# algo1.max_iteration += 1000
algo2.run(1000)
#%%
show2D([algo1.solution*A1.norm(), algo2.solution, gt], 
    title=[f'Operator rescale Matthias {alpha}', f'Operator Rescale Vaggelis {alpha}', 'GT'] ,
     cmap='plasma', fix_range=False)
#%% 
line_plot([gt, fbp_recon, algo1.solution * A1.norm(), algo2.solution], 
           label=['Ground Truth', 'FBP', 'Matthias PDHG + TV + nn', 'Vaggelis PDHG + TV + nn'],
           line_coords=(('horizontal_x',64), ('vertical',64)), 
           title=f'Comparison alpha {alpha}'
           )
#%%

# Save the solutions
NEXUSDataWriter(file_name=f"Matthias_PDHG_rescale_alpha_{alpha}.nxs", 
                data=(algo1.solution * A1.norm())).write()
NEXUSDataWriter(file_name=f"Vaggelis_PDHG_rescale_L2NormSquared_alpha_{alpha}.nxs",
                data=algo2.solution).write()

# problem 2 same data pixel size = 1
# ag2 = data.geometry.copy()
# print (ag2)
# #%%
# ag2.pixel_size_h = 1.
# ag2.pixel_size_v = 1.
# print (ag2)

# # data2 = ag2.allocate(None)
# # data2.fill(data)

# from cil.framework import AcquisitionData
# # use the same data array as before only the geometry is different
# data2 = AcquisitionData(data.array, deep_copy=False, geometry=ag2, suppress_warning=True)

# ig2 = data2.geometry.get_ImageGeometry()
# print (ig2)
# #%%
# # A2 = ProjectionOperator(ig2, data2.geometry)
# A2 = POA(ig2, data2.geometry)
# print ("norm A2", A2.norm())

# # the Gradient is scaled by the voxel size, so in the case we set the pixel size
# # to 1 we need to rescale the alpha by the pixel size of ig
# # assuming cubic voxels
# alpha2 = alpha / (ig1.voxel_size_x)#* ig1.voxel_size_y * ig1.voxel_size_z)

# F2 = BlockFunction(
#     L2NormSquared(b=data2),
#     MixedL21Norm()
# )

# g2 = GradientOperator(ig2)
# print ("norm g2", g2.norm())

# K2 = BlockOperator(
#     A2, 
#     alpha2 * g2
#     )

# # G2 = ZeroFunction()
# G2 = IndicatorBox(lower=0)
# #%%
# algo2 = PDHG(f=F2, g=G2, operator=K2, max_iteration=1000, update_objective_interval=100, log_file="algo2.nxs")
# #%%
# # algo2.max_iteration += 1000
# algo2.run(1000)
# # NEXUSDataWriter(algo2.solution, file_name='algo2.nxs').write()
# # %%
# # show2D(algo2.solution, cmap='plasma', fix_range=False)

# # # %%
# # on a smaller voxel the same attenuation is given if the effective density is 
# # higher, scaling as the inverse of the voxel volume.
# diff = algo1.solution - algo2.solution/(ig1.voxel_size_x*ig1.voxel_size_y*ig1.voxel_size_z )

# show2D([algo1.solution, algo2.solution, gt, diff], 
#        title=['pix {} {} alpha {}'.format(ig1.voxel_size_x,ig1.voxel_size_y, alpha), 
#               'pix {} {} alpha {}'.format(ig2.voxel_size_x,ig2.voxel_size_y, alpha2),
#               'Ground Truth', 'diff'], cmap='plasma',
#        fix_range=False)

# #%%
# show2D([algo1.solution, algo2.solution, gt], 
#        title=['pix {} {} alpha {}'.format(ig1.voxel_size_x,ig1.voxel_size_y, alpha), 
#               'pix {} {} alpha {}'.format(ig2.voxel_size_x,ig2.voxel_size_y, alpha2),
#               'Ground Truth'], cmap='plasma', fix_range=False)
# # %%
# import matplotlib.pyplot as plt

# plt.semilogy([el[0] for el in algo1.loss], label='pix 64')
# plt.semilogy([el[0] for el in algo2.loss], label='pix 1')
# plt.legend()
# plt.show()
# # # %%

# # %%

# def prova (**kwargs):
#     if len(kwargs) == 0:
#         return 0
#     else:
#         return 1

# print(prova(a=1,b=2))

# %%
