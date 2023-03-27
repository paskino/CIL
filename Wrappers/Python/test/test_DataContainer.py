# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
from utils import initialise_tests
import sys
import numpy as np
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry, BlockGeometry, VectorGeometry
from cil.framework import AcquisitionGeometry
from timeit import default_timer as timer
import logging
from testclass import CCPiTestClass
import functools

initialise_tests()

def dt(steps):
    return steps[-1] - steps[-2]
def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]



class TestDataContainer(CCPiTestClass):
    def create_simple_ImageData(self):
        N = 64
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        Phantom = ImageData(geometry=ig)

        x = Phantom.as_array()

        x[int(round(N/4)):int(round(3*N/4)),
          int(round(N/4)):int(round(3*N/4))] = 0.5
        x[int(round(N/8)):int(round(7*N/8)),
          int(round(3*N/8)):int(round(5*N/8))] = 1

        return (ig, Phantom)

    def create_DataContainer(self, X,Y,Z, value=1):
        steps = [timer()]
        a = value * np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        return ds

    def test_creation_nocopy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = np.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = np.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(id(a), id(ds.array))
        self.assertEqual(ds.dimension_labels, ('X', 'Y', 'Z', 'W'))

    def testGb_creation_nocopy(self):
        X, Y, Z = 512, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 3)
        ds1 = ds.copy()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))
        ds1 = ds.clone()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))

    def test_ndim(self):

        x_np = np.arange(0, 60).reshape(3,4,5)
        x_cil = DataContainer(x_np)
        self.assertEqual(x_np.ndim, x_cil.ndim)
        self.assertEqual(3, x_cil.ndim)

    def testInlineAlgebra(self):
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #ds.__iadd__( 2 )
        ds += 2
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( 2 )
        ds -= 2
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( 2 )
        ds *= 2
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( 2 )
        ds /= 2
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds1 = ds.copy()
        #ds1.__iadd__( 1 )
        ds1 += 1
        #ds.__iadd__( ds1 )
        ds += ds1
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( ds1 )
        ds -= ds1
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( ds1 )
        ds *= ds1
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( ds1 )
        ds /= ds1
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 1.)


    def test_unary_operations(self):
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = -np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])

        ds.sign(out=ds)
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], -1.)

        ds.abs(out=ds)
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds.__imul__(2)
        ds.sqrt(out=ds)
        steps.append(timer())
        self.assertEqual(ds.as_array()[0][0][0],
                         np.sqrt(2., dtype='float32'))

    def test_binary_operations(self):
        self.binary_add()
        self.binary_subtract()
        self.binary_multiply()
        self.binary_divide()

    def binary_add(self):
        X, Y, Z = 512, 512, 512
        #X, Y, Z = 1024, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()

        steps.append(timer())
        ds.add(ds1, out=out)
        steps.append(timer())
        t1 = dt(steps)
        steps.append(timer())
        ds2 = ds.add(ds1)
        steps.append(timer())
        t2 = dt(steps)
        
        #self.assertLess(t1, t2)
        self.assertEqual(out.as_array()[0][0][0], 2.)
        self.assertnpArrayEqual(out.as_array(), ds2.as_array())
        
        ds0 = ds
        dt1 = 0
        dt2 = 0
        for i in range(1):
            steps.append(timer())
            ds0.add(2, out=out)
            steps.append(timer())
            self.assertEqual(3., out.as_array()[0][0][0])

            dt1 += dt(steps)/10
            steps.append(timer())
            ds3 = ds0.add(2)
            steps.append(timer())
            dt2 += dt(steps)/10
        
        self.assertnpArrayEqual(out.as_array(), ds3.as_array())
        #self.assertLess(dt1, dt2)
        

    def binary_subtract(self):
        X, Y, Z = 512, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()

        steps.append(timer())
        ds.subtract(ds1, out=out)
        steps.append(timer())
        t1 = dt(steps)
        self.assertEqual(0., out.as_array()[0][0][0])

        steps.append(timer())
        ds2 = out.subtract(ds1)
        self.assertEqual(-1., ds2.as_array()[0][0][0])

        steps.append(timer())
        t2 = dt(steps)
        
        #self.assertLess(t1, t2)

        del ds1
        ds0 = ds.copy()
        steps.append(timer())
        ds0.subtract(2, out=ds0)
        #ds0.__isub__( 2 )
        steps.append(timer())
        
        self.assertEqual(-1., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.subtract(2)
        steps.append(timer())
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(-1., ds0.as_array()[0][0][0])
        self.assertEqual(-3., ds3.as_array()[0][0][0])


    def binary_multiply(self):
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.multiply(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        steps.append(timer())
        ds2 = ds.multiply(ds1)
        steps.append(timer())
        t2 = dt(steps)
        
        #self.assertLess(t1, t2)

        ds0 = ds
        ds0.multiply(2, out=ds0)
        steps.append(timer())
        self.assertEqual(2., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.multiply(2)
        steps.append(timer())
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(4., ds3.as_array()[0][0][0])
        self.assertEqual(2., ds.as_array()[0][0][0])
        
        ds.multiply(2.5, out=ds0)
        self.assertEqual(2.5*2., ds0.as_array()[0][0][0])


    def binary_divide(self):
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = np.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        t1 = 0 
        t2 = 0
        N=1
        for i in range(N):
            steps.append(timer())
            ds.divide(ds1, out=ds)
            steps.append(timer())
            t1 += dt(steps)/N
            steps.append(timer())
            ds2 = ds.divide(ds1)
            steps.append(timer())
            t2 += dt(steps)/N
            
        #self.assertLess(t1, t2)
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds0 = ds
        ds0.divide(2, out=ds0)
        steps.append(timer())
        self.assertEqual(0.5, ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.divide(2)
        steps.append(timer())
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(.25, ds3.as_array()[0][0][0])
        self.assertEqual(.5, ds.as_array()[0][0][0])


    def test_reverse_operand_algebra(self):
        number = 3/2
        
        X, Y, Z = 32, 64, 128
        a = np.ones((X, Y, Z), dtype='float32')
        ds = DataContainer(a * 3, False, ['X', 'Y', 'Z'])

        # rdiv
        b = number / ds
        np.testing.assert_array_almost_equal(a * 0.5, b.as_array())
        # radd
        number = 1
        b = number + ds
        np.testing.assert_array_almost_equal(a * 4, b.as_array())
        # rsub
        number = 3
        b = number - ds
        np.testing.assert_array_almost_equal(np.zeros_like(a), b.as_array())
        # rmul
        number = 1/3
        b = number * ds
        np.testing.assert_array_almost_equal(a, b.as_array())
        # rpow
        number = 2
        b = number ** ds
        np.testing.assert_array_almost_equal(a * 8, b.as_array())


    def test_creation_copy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = np.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = np.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, True, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 2)


    def test_dot(self):
        a0 = np.asarray([i for i in range(2*3*4)])
        a1 = np.asarray([2*i for i in range(2*3*4)])
                       
        ds0 = DataContainer(np.reshape(a0,(2,3,4)))
        ds1 = DataContainer(np.reshape(a1,(2,3,4)))
        
        np.testing.assert_equal(ds0.dot(ds1), a0.dot(a1))
        
        a2 = np.asarray([2*i for i in range(2*3*5)])
        ds2 = DataContainer(np.reshape(a2,(2,3,5)))
        
        # it should fail if the shape is wrong
        try:
            ds2.dot(ds0)
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
            
        n0 = (ds0 * ds1).sum()
        n1 = ds0.as_array().ravel().dot(ds1.as_array().ravel())
        self.assertEqual(n0, n1)


    def test_exp_log(self):
        a0 = np.asarray([1. for i in range(2*3*4)])
                
        ds0 = DataContainer(np.reshape(a0,(2,3,4)), suppress_warning=True)
        # ds1 = DataContainer(np.reshape(a1,(2,3,4)), suppress_warning=True)
        b = ds0.exp().log()
        np.testing.assert_allclose(ds0.as_array(), b.as_array())
        
        self.assertEqual(ds0.exp().as_array()[0][0][0], np.exp(1))
        self.assertEqual(ds0.log().as_array()[0][0][0], 0.)
        
        
    def test_ImageData(self):
        # create ImageData from geometry
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        #vol = ImageData(geometry=vgeometry)
        vol = vgeometry.allocate()
        self.assertEqual(vol.shape, (2, 3, 4))

        vol1 = vol + 1
        self.assertnpArrayEqual(vol1.as_array(), np.ones(vol.shape))

        vol1 = vol - 1
        self.assertnpArrayEqual(vol1.as_array(), -np.ones(vol.shape))

        vol1 = 2 * (vol + 1)
        self.assertnpArrayEqual(vol1.as_array(), 2 * np.ones(vol.shape))

        vol1 = (vol + 1) / 2
        self.assertnpArrayEqual(vol1.as_array(), np.ones(vol.shape) / 2)

        vol1 = vol + 1
        self.assertEqual(vol1.sum(), 2*3*4)
        vol1 = (vol + 2) ** 2
        self.assertnpArrayEqual(vol1.as_array(), np.ones(vol.shape) * 4)

        self.assertEqual(vol.number_of_dimensions, 3)
        
        ig2 = ImageGeometry (voxel_num_x=2,voxel_num_y=3,voxel_num_z=4, 
                     dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y,
                 ImageGeometry.VERTICAL])
        data = ig2.allocate()
        self.assertnpArrayEqual(np.asarray(data.shape), np.asarray(ig2.shape))
        self.assertnpArrayEqual(np.asarray(data.shape), data.as_array().shape)


    def test_ImageData_apply_circular_mask(self):
        ig = ImageGeometry(voxel_num_x=6, voxel_num_y=3, voxel_size_x=0.5, voxel_size_y = 1)
        data_orig = ig.allocate(1)

        data_masked1 = data_orig.copy()
        data_masked1.apply_circular_mask(0.8)

        self.assertEqual(data_orig.geometry, data_masked1.geometry)
        self.assertEqual(np.count_nonzero(data_masked1.array), 14)

        data_masked1 = data_orig.copy()
        data_masked1.apply_circular_mask(0.5)

        self.assertEqual(data_orig.geometry, data_masked1.geometry)
        self.assertEqual(np.count_nonzero(data_masked1.array), 8)

        data2 = data_orig.copy()
        data_masked2 = data2.apply_circular_mask(0.5, False)

        np.testing.assert_allclose(data_orig.array, data2.array)
        self.assertEqual(np.count_nonzero(data_masked2.array), 8)





    def test_AcquisitionData(self):
        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        #sino = AcquisitionData(geometry=sgeometry)
        sino = sgeometry.allocate()
        self.assertEqual(sino.shape, (2, 10, 3, 5))
           
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((2,3)).set_channels(4)                       
        data = ag.allocate()
        self.assertnpArrayEqual(np.asarray(data.shape), np.asarray(ag.shape))
        self.assertnpArrayEqual(np.asarray(data.shape), data.as_array().shape)
        
        ag2 = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((2,3)).set_channels(4)\
                                 .set_labels([AcquisitionGeometry.VERTICAL ,
                         AcquisitionGeometry.ANGLE, AcquisitionGeometry.HORIZONTAL, AcquisitionGeometry.CHANNEL])
        
        data = ag2.allocate()
        self.assertnpArrayEqual(np.asarray(data.shape), np.asarray(ag2.shape))
        self.assertnpArrayEqual(np.asarray(data.shape), data.as_array().shape)


    def test_ImageGeometry_allocate(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image = vgeometry.allocate()
        self.assertEqual(0,image.as_array()[0][0][0])
        image = vgeometry.allocate(1)
        self.assertEqual(1,image.as_array()[0][0][0])
        default_order = ['channel' , 'horizontal_y' , 'horizontal_x']
        self.assertEqual(default_order[0], image.dimension_labels[0])
        self.assertEqual(default_order[1], image.dimension_labels[1])
        self.assertEqual(default_order[2], image.dimension_labels[2])
        order = [ 'horizontal_x' , 'horizontal_y', 'channel' ]
        vgeometry.set_labels(order)
        image = vgeometry.allocate(0)
        self.assertEqual(order[0], image.dimension_labels[0])
        self.assertEqual(order[1], image.dimension_labels[1])
        self.assertEqual(order[2], image.dimension_labels[2])
        
        ig = ImageGeometry(2,3,2)
        try:
            z = ImageData(np.random.randint(10, size=(2,3)), geometry=ig)
            self.assertTrue(False)
        except ValueError as ve:
            logging.info(str (ve))
            self.assertTrue(True)

        #vgeometry.allocate('')
    def test_AcquisitionGeometry_allocate(self):
        ageometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)
        sino = ageometry.allocate(0)
        shape = sino.shape
        self.assertAlmostEqual(0.,sino.as_array()[0][0][0][0])
        self.assertAlmostEqual(0.,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        
        sino = ageometry.allocate(1)
        self.assertEqual(1,sino.as_array()[0][0][0][0])
        self.assertEqual(1,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        
        default_order = ['channel' , 'angle' ,
                         'vertical' , 'horizontal']
        self.assertEqual(default_order[0], sino.dimension_labels[0])
        self.assertEqual(default_order[1], sino.dimension_labels[1])
        self.assertEqual(default_order[2], sino.dimension_labels[2])
        self.assertEqual(default_order[3], sino.dimension_labels[3])
        order = ['vertical' , 'horizontal', 'channel' , 'angle' ]
        ageometry.set_labels(order)
        sino = ageometry.allocate(0)
        
        self.assertEqual(order[0], sino.dimension_labels[0])
        self.assertEqual(order[1], sino.dimension_labels[1])
        self.assertEqual(order[2], sino.dimension_labels[2])
        self.assertEqual(order[2], sino.dimension_labels[2])
                
        try:
            z = AcquisitionData(np.random.randint(10, size=(2,3)), geometry=ageometry)
            self.assertTrue(False)
        except ValueError as ve:
            logging.info(str(ve))
            self.assertTrue(True)


    def test_BlockGeometry_allocate_dtype(self):
        ig1 = ImageGeometry(3,3)
        ig2 = ImageGeometry(3,3, dtype=np.int16)
        bg = BlockGeometry(ig1,ig2)

        # print("The default dtype of the BlockImageGeometry is {}".format(bg.dtype))   
        self.assertEqual(bg.dtype, (np.float32, np.int16))


    def dtype_allocate_test(self, geometry):
        classname = geometry.__class__.__name__
        # print("The default dtype of the {} is {}".format(classname , geometry.dtype))
        self.assertEqual(geometry.dtype, np.float32)

        #print("Change it to complex")
        geometry.dtype = np.complex64
        self.assertEqual(geometry.dtype, np.complex64)

        geometry.dtype = np.complex128
        self.assertEqual(geometry.dtype, np.complex128)

        geometry.dtype = complex
        self.assertEqual(geometry.dtype, complex)

        #print("Test {} allocate".format(classname ))
        data = geometry.allocate()
        #print("Data dtype is now {} ".format(geometry.dtype))  
        self.assertEqual(data.dtype, geometry.dtype)
        #print("Data geometry dtype is now")
        #print(data.geometry.dtype)
        self.assertEqual(data.geometry.dtype, geometry.dtype) 

        #print("Allocate data with different dtype, e.g: np.int64 from the same {}".format(classname ))
        data = geometry.allocate(dtype=np.int64)
        self.assertEqual(data.dtype, np.int64) 
        #print("Data dtype is now {}".format(data.dtype))
        #print("Data geometry dtype is now {}".format(data.geometry.dtype))
        self.assertEqual(data.geometry.dtype, np.int64)
        self.assertEqual(data.dtype, np.int64)

        #print("The dtype of the {} remain unchanged ig.dtype =  {}".format(classname, geometry.dtype))
        self.assertEqual(geometry.dtype, complex)

        self.assertNotEqual(id(geometry), id(data.geometry))


    def test_ImageGeometry_allocate_dtype(self):        
        #print("Test ImageGeometry dtype\n")
        ig = ImageGeometry(3,3)
        self.dtype_allocate_test(ig)
    

    def test_AcquisitionGeometry_allocate_dtype(self):
        # print("Test AcquisitionGeometry dtype\n")
        # Detectors
        detectors =  10

        # Angles
        angles = np.linspace(0,180,180, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)                                
        self.dtype_allocate_test(ag)         


    def test_VectorGeometry_allocate_dtype(self):
        # print("Test VectorGeometry dtype\n")

        vg = VectorGeometry(3)
        self.dtype_allocate_test(vg)


    def complex_allocate_geometry_test(self, geometry):
        data = geometry.allocate(dtype=np.complex64)
        r = (1 + 1j*1)* np.ones(data.shape, dtype=data.dtype)
        data.fill(r)
        self.assertAlmostEqual(data.squared_norm(), data.size * 2)  
        np.testing.assert_almost_equal(data.abs().array, np.abs(r))              

        data1 = geometry.allocate(dtype=np.float32)
        try:
            data1.fill(r)
            self.assertTrue(False)
        except TypeError as err:
            logging.info(str(err))
            self.assertTrue(True)


    def test_ImageGeometry_allocate_complex(self):
        ig = ImageGeometry(2,2)
        self.complex_allocate_geometry_test(ig)


    def test_AcquisitionGeometry_allocate_complex(self):
        # Detectors
        detectors =  10

        # Angles
        angles = np.linspace(0,10,10, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)   

        self.complex_allocate_geometry_test(ag)


    def test_VectorGeometry_allocate_complex(self):
        vg = VectorGeometry(3)
        self.complex_allocate_geometry_test(vg)
        

    def test_ImageGeometry_allocate_random_same_seed(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image1 = vgeometry.allocate('random', seed=0)
        image2 = vgeometry.allocate('random', seed=0)
        np.testing.assert_allclose(image1.as_array(), image2.as_array())


    def test_AcquisitionDataSubset(self):
        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        # expected dimension_labels
        
        self.assertListEqual([AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL],
                              list(sgeometry.dimension_labels))
        sino = sgeometry.allocate()

        # test reshape
        new_order = [AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.CHANNEL , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.ANGLE]
        sino.reorder(new_order)

        self.assertListEqual(new_order, list(sino.geometry.dimension_labels))

        ss1 = sino.get_slice(vertical = 0)
        self.assertListEqual([AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.CHANNEL  ,
                 AcquisitionGeometry.ANGLE], list(ss1.geometry.dimension_labels))
        ss2 = sino.get_slice(vertical = 0, channel=0)
        self.assertListEqual([AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.ANGLE], list(ss2.geometry.dimension_labels))


    def test_ImageDataSubset(self):
        new_order = ['horizontal_x', 'channel', 'horizontal_y']


        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2, dimension_labels=new_order)
        # expected dimension_labels
        
        self.assertListEqual(new_order,
                              list(vgeometry.dimension_labels))
        vol = vgeometry.allocate()

        # test reshape
        new_order = ['channel', 'horizontal_y','horizontal_x']
        vol.reorder(new_order)

        self.assertListEqual(new_order, list(vol.geometry.dimension_labels))

        ss1 = vol.get_slice(horizontal_x = 0)
        self.assertListEqual(['channel', 'horizontal_y'], list(ss1.geometry.dimension_labels))

        vg = ImageGeometry(3,4,5,channels=2)
        self.assertListEqual([ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X],
                              list(vg.dimension_labels))
        ss2 = vg.allocate()
        ss3 = vol.get_slice(channel=0)
        self.assertListEqual([ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X], list(ss3.geometry.dimension_labels))


    def test_DataContainerChaining(self):
        dc = self.create_DataContainer(256,256,256,1)

        dc.add(9,out=dc)\
          .subtract(1,out=dc)
        self.assertEqual(1+9-1,dc.as_array().flatten()[0])


    def test_reduction(self):
        dc = self.create_DataContainer(2,2,2,value=1)
        sqnorm = dc.squared_norm()
        norm = dc.norm()
        self.assertEqual(sqnorm, 8.0)
        np.testing.assert_almost_equal(norm, np.sqrt(8.0), decimal=7)
    

    def test_reduction_mean(self):
        ig = ImageGeometry(2,2)
        data = ig.allocate(0)
        np_arr = data.as_array()
        np_arr[0][0] = 0
        np_arr[0][1] = 1
        np_arr[1][0] = 2
        np_arr[1][1] = 3
        data.fill(np_arr)

        mean = data.mean()
        expected = np.float64(0+1+2+3)/np.float64(4)
        np.testing.assert_almost_equal(mean, expected)
        
        
    def test_multiply_out(self):
        ig = ImageGeometry(10,11,12)
        u = ig.allocate()
        a = np.ones(u.shape)
        
        u.fill(a)
        
        np.testing.assert_array_equal(a, u.as_array())
        
        #u = ig.allocate(ImageGeometry.RANDOM_INT, seed=1)
        l = functools.reduce(lambda x,y: x*y, (10,11,12), 1)
        
        a = np.zeros((l, ), dtype=np.float32)
        for i in range(l):
            a[i] = np.sin(2 * i* 3.1415/l)
        b = np.reshape(a, u.shape)
        u.fill(b)
        np.testing.assert_array_equal(b, u.as_array())
        
        u.multiply(2, out=u)
        c = b * 2
        np.testing.assert_allclose(u.as_array(), c)


    def test_sapyb_datacontainer_f(self):
        #a vec, b vec
        
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(dtype=np.float32)                                                     
        d2 = ig.allocate(dtype=np.float32)   
        a = ig.allocate(dtype=np.float32)                                                  
        b = ig.allocate(dtype=np.float32)         

        d1.fill(np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32))
        d2.fill(np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32))
        a.fill(1.0/d1.as_array())                                                  
        b.fill(-1.0/d2.as_array())   

        out = ig.allocate(-1,dtype=np.float32)                                                 
        # equals to 1 + -1 = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_scalar_f(self):
        # a,b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=np.float32)                                                     
        d2 = ig.allocate(2, dtype=np.float32)   
        a = 2.
        b = -1.

        out = ig.allocate(-1,dtype=np.float32)                                                 
        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        np.testing.assert_array_equal(res, d2.as_array())


    def test_sapyb_datacontainer_scalar_f(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1., dtype=np.complex64)                                                     
        d2 = ig.allocate(2.,dtype=np.complex64)   
        a = 2.+2j                                                
        b = ig.allocate(-1.-1j, dtype=np.complex64)         

        out = ig.allocate(-1,dtype=np.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        
        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())


    def test_sapyb_scalar_c(self):
        # a, b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=np.complex64)                                                     
        d2 = ig.allocate(2, dtype=np.complex64)   
        a = 2.+2j
        b = -1.-1j

        out = ig.allocate(-1,dtype=np.complex64)                                                 
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        np.testing.assert_array_equal(res, d2.as_array())


    def test_sapyb_datacontainer_c(self):
        #a vec, b vec
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(dtype=np.complex64)                                                     
        d2 = ig.allocate(dtype=np.complex64)   
        a = ig.allocate(dtype=np.complex64)                                                  
        b = ig.allocate(dtype=np.complex64)         

        arr = np.empty(ig.shape, dtype=np.complex64)
        arr.real = np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32)
        arr.imag = np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32)

        d1.fill(arr)

        arr.imag = -1* arr.imag
        d2.fill(arr)

        a.fill(d2.as_array())                                                  
        b.fill(d1.as_array())   

        out = ig.allocate(-1,dtype=np.complex64)
        # equals to d1^ * d1 + d2^*d2 = d1**2 + d2**2 = 2* arr.norm = 2 * (arr.real **2 + arr.imag **2)
        out = d1.sapyb(a,d2,b)
        res = 2* (arr.real * arr.real + arr.imag * arr.imag)
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_datacontainer_scalar_c(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1., dtype=np.complex64)                                                     
        d2 = ig.allocate(2.,dtype=np.complex64)   
        a = 2.+2j                                                
        b = ig.allocate(-1.-1j, dtype=np.complex64)         


        out = ig.allocate(-1,dtype=np.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        
        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())


    def test_sapyb_scalar_f_c(self):
        # a,b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=np.complex64)                                                     
        d2 = ig.allocate(2, dtype=np.float32)   
        a = 2.+1j
        b = -1.

        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array()) + 1j
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        try:
            with self.assertRaises(np.core._exceptions.UFuncTypeError) as context:
                d1.sapyb(a,d2,b, out=d2)
        except AttributeError as ae:
            logging.info ("Probably np version too low: {}".format(ae))

        # print ("Exception thrown:", str(context.exception))
        
        # out is complex
        # d1.fill(1+0j)
        d2.fill(2)
        d1.sapyb(a,d2,b,out=d1)
        # 2+1j * [1+0j] -1 * [2]
        np.testing.assert_array_equal(1j * np.ones_like(d1.as_array()), d1.as_array())
            

    def test_min(self):
        ig = ImageGeometry(10,10)     
        a = np.asarray(np.linspace(-10,10, num=100, endpoint=True), dtype=np.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)                                                     
        d1.fill(a)                                                     
        self.assertAlmostEqual(d1.min(), -10.)


    def test_max(self):
        ig = ImageGeometry(10,10)     
        a = np.asarray(np.linspace(-10,10, num=100, endpoint=True), dtype=np.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)                                                     
        d1.fill(a)                                                     
        self.assertAlmostEqual(d1.max(), 10.)


    def test_size(self):
        ig = ImageGeometry(10,10)     
        d1 = ig.allocate(1)                                                     
                                                
        self.assertEqual( d1.size, 100 )
        
        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        ad = sgeometry.allocate()

        self.assertEqual( ad.size, 3*5*10*2 )
    

    def test_negation(self):
        X, Y, Z = 256, 512, 512
        a = np.ones((X, Y, Z), dtype='int32')
        
        ds = - DataContainer(a, False, ['X', 'Y', 'Z'])
        
        np.testing.assert_array_equal(ds.as_array(), -a)


    def test_fill_dimension_ImageData(self):
        ig = ImageGeometry(2,3,4)
        u = ig.allocate(0)
        a = np.ones((4,2))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        
        data = u.as_array()
        axis_number = u.get_dimension_axis('horizontal_y')
        
        u.fill(a, horizontal_y=0)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=1).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=1).as_array(), 2 * a)
        
        b = u.get_slice(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=2).as_array(), 3 * a)


    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        a = np.ones((4,2))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        
        data = u.as_array()
        axis_number = u.get_dimension_axis('horizontal_y')
        
        u.fill(a, horizontal_y=0)
        np.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.subset(horizontal_y=1).as_array(), 2 * a)
        
        b = u.subset(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        np.testing.assert_array_equal(u.subset(horizontal_y=2).as_array(), 3 * a)

        # slice with 2 axis
        a = np.ones((2,))
        u.fill(a, horizontal_y=1, vertical=0)
        np.testing.assert_array_equal(u.subset(horizontal_y=1, vertical=0).as_array(), a)


    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        # (2, 5, 3, 4)
        a = np.ones((2,5))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        b = u.get_slice(channel=0, vertical=0)
        data = u.as_array()
        
        u.fill(a, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), a)

        u.fill(2, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), 2 * a)

        u.fill(2, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), 2 * a)
        
        b = u.get_slice(channel=0, vertical=0)
        b.fill(3)
        u.fill(b, channel=1, vertical=1)
        np.testing.assert_array_equal(u.get_slice(channel=1, vertical=1).as_array(), 3 * a)

