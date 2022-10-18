# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

try:
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
except ImportError as ie:
    raise ImportError(ie , "\n\n", 
                      "islicer requires the additional package ipywidgets\n" +
                      "Please install it via conda as ipywidgets from the conda-forge channel\n")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy

from IPython.display import HTML
import random
from cil.utilities.display import set_origin


def display_slice(container, direction, title, cmap, size, axis_labels, origin):
    
        
    def get_slice_3D(x, minmax, roi_hdir, roi_vdir):
        
        if direction == 0:
            img = container[x]
            x_lim = container.shape[2]
            y_lim = container.shape[1]
            x_label = axis_labels[2]
            y_label = axis_labels[1] 
            
        elif direction == 1:
            img = container[:,x,:]
            x_lim = container.shape[2]
            y_lim = container.shape[0] 
            x_label = axis_labels[2]
            y_label = axis_labels[0]             
            
        elif direction == 2:
            img = container[:,:,x]
            x_lim = container.shape[1]
            y_lim = container.shape[0]    
            x_label = axis_labels[1]
            y_label = axis_labels[0]             
        
        if size is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=size)
        
        if isinstance(title, (list, tuple)):
            dtitle = title[x]
        else:
            dtitle = title
        
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=(1,.05), height_ratios=(1,))
        # image
        ax = fig.add_subplot(gs[0, 0])
      
        ax.set_xlabel(x_label)     
        ax.set_ylabel(y_label)

        img, data_origin, _ = set_origin(img, origin)
        aximg = ax.imshow(img, cmap=cmap, origin=data_origin, aspect='auto')#, extent=(*roi_hdir, *roi_vdir))
        aximg.set_clim(minmax)
        ax.set_xlim(*roi_hdir)
        ax.set_ylim(*roi_vdir)
        ax.set_title(dtitle + " {}".format(x))
        # colorbar
        ax = fig.add_subplot(gs[0, 1])
        plt.colorbar(aximg, cax=ax)
        plt.tight_layout()
        plt.show(fig)
        
    return get_slice_3D

    
def islicer(data, direction=0, title="", slice_number=None, cmap='gray', minmax=None, size=None, axis_labels=None, origin='lower-left'):
    """
    Creates an interactive slider that slices a 3D volume along an axis.

    Parameters
    ----------
    data : DataContainer or numpy.ndarray
        A 3-dimensional dataset from which 2-dimensional slices will be
        shown
    direction : int
        Axis to slice on. Can be 0,1,2 or the axis label, default 0
    title : str, list of str or tuple of str, default=''
        Title for the display
    slice_number : int, optional
        Start slice number (default is None, which results in the center
        slice being shown initially)
    cmap : str or matplotlib.colors.Colormap, default='gray'
        Set the colour map
    minmax : tuple
        Colorbar (min, max) values, default None (uses the min, max of
        values in `data`)
    size : int or tuple, optional
        Specify the figure size in inches. If `int` this specifies the
        width, and scales the height in order to keep the standard
        `matplotlib` aspect ratio, default None (use the default matplotlib
        figure size)
    axis_labels : list of str, optional
        The axis labels to use for each of the 3 dimensions in the data
        (default is None, resulting in labels extracted from the data, or
        ['X','Y','Z'] if no labels are present)
    origin : {'lower-left', 'upper-left', 'lower-right', 'upper-right'}
        Sets the display origin

    Returns
    -------
    slider : ipywidgets.IntSlider
        The slider whose value determines the slice on display.
    """
    
    if axis_labels is None:
        if hasattr(data, "dimension_labels"):
            axis_labels = [data.dimension_labels[0],data.dimension_labels[1],data.dimension_labels[2]]
        else:
            axis_labels = ['X', 'Y', 'Z']

    if isinstance (data, numpy.ndarray):
        container = data
    elif hasattr(data, "__getitem__"):
        container = data
    elif hasattr(data, "as_array"):
        container = data.as_array()
        
    if not isinstance (direction, int):
        if direction in data.dimension_labels:
            direction = data.get_dimension_axis(direction)                             

    if slice_number is None:
        slice_number = int(data.shape[direction]/2)
        
    slider = widgets.IntSlider(min=0, max=data.shape[direction]-1, step=1, 
                             value=slice_number, continuous_update=False, description=axis_labels[direction])
    amax = container.max()
    amin = container.min()
    if minmax is None:    
        cmax = amax
        cmin = amin
    else:
        cmin = min(minmax)
        cmax = max(minmax)
    
    if isinstance (size, (int, float)):
        default_ratio = 6./8.
        size = ( size , size * default_ratio )
    
    min_max = widgets.FloatRangeSlider(
                                value=[cmin, cmax],
                                min=amin,
                                max=amax,
                                step=(amax-amin)/100.,
                                description='display window',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='.1e',
                            )

    dirs_remaining = [i for i in range(3) if i != direction]
    h_dir, v_dir = dirs_remaining[1], dirs_remaining[0]
    h_dir_size = container.shape[h_dir]
    v_dir_size = container.shape[v_dir]

    roi_select_hdir = widgets.IntRangeSlider(
        value=[0, h_dir_size-1],
        min=0,
        max=h_dir_size-1,
        step=1,
        description=f'roi_{axis_labels[h_dir]}',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    roi_select_vdir = widgets.IntRangeSlider(
        value=[0, v_dir_size-1],
        min=0,
        max=v_dir_size-1,
        step=1,
        description=f'roi_{axis_labels[v_dir]}',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    interact(display_slice(container, 
                           direction, 
                           title=title, 
                           cmap=cmap, 
                           # minmax=(amin, amax),
                           size=size, axis_labels=axis_labels,
                           origin=origin),
                           x=slider, minmax=min_max,
                           roi_hdir=roi_select_hdir,
                           roi_vdir=roi_select_vdir)
    
    return slider
    

def link_islicer(*args):
    '''links islicers IntSlider widgets

    Parameters
    ----------
    args: islicer objects to link
    '''
    linked = [(widg, 'value') for widg in args]
    # link pair-wise
    pairs = [(linked[i+1],linked[i]) for i in range(len(linked)-1)]
    for pair in pairs:
        widgets.link(*pair)


# https://stackoverflow.com/questions/31517194/how-to-hide-one-specific-cell-input-or-output-in-ipython-notebook/52664156

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)    
