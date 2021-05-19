""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

from __future__ import division
import numpy as np
from scipy.linalg import expm, norm

########################################################################################################################
################################### MULTI DIMENSIONAL SUPPORT ##########################################################
def getGrid(siz):
    """ Returns grid with coordinates from -siz[0]/2 : siz[0]/2, -siz[1]/2 : siz[1]/2, ...."""
    space = [np.linspace( -((N+1)//2)+1, (N//2), N ) for N in siz]
    mesh = np.meshgrid( *space, indexing='ij' )
    mesh = [np.expand_dims( ax.ravel(), 0) for ax in mesh]

    return np.concatenate(mesh).reshape([len(siz)]+list(siz))

def coordinate_map(cube):

    #Linspace for cubec
    ds = cube.shape
    space = [np.linspace(0, s-1, s, dtype='uint16') for s in ds]

    # Grid for cube
    grid = np.meshgrid(*space, indexing='ij')

    #Select coordinates from grid
    map = [x[cube.astype('bool')] for x in grid]

    #Return
    map = [np.expand_dims(m.flatten(), 0) for m in map]
    return np.concatenate(map, axis=0)

########################################################################################################################
################################### 1D/2D/3D SUPPORT ###################################################################

def linear_interpolation(input_array, output_inds, boundary_correction = True, boundary_val=0, out_shape=None):
    if input_array.ndim == 1:
        return _linear_interpolation_1D(input_array, output_inds, boundary_correction=boundary_correction, out_shape=out_shape)
    elif input_array.ndim == 2:
        return linear_interpolation_2D(input_array, output_inds, boundary_correction=boundary_correction, out_shape=out_shape)
    elif input_array.ndim == 3:
        return _linear_interpolation_3D(input_array, output_inds, boundary_correction = boundary_correction, boundary_val=boundary_val,out_shape=out_shape)

def nearest_interpolation(input_array, output_inds, boundary_correction = True, boundary_val=0, out_shape=None):
    if input_array.ndim == 1:
        return _nearest_interpolation_1D(input_array, output_inds, boundary_correction=boundary_correction, out_shape=out_shape)
    elif input_array.ndim == 2:
        return _nearest_interpolation_2D(input_array, output_inds, boundary_correction=boundary_correction, boundary_val=boundary_val,out_shape=out_shape)
    elif input_array.ndim == 3:
        return _nearest_interpolation_3D(input_array, output_inds, boundary_correction = boundary_correction,out_shape=out_shape)

########################################################################################################################
################################### 3D SUPPORT #########################################################################


def getCoordinatesFromCube(label_cube):

    ds = label_cube.shape

    # Coordinates for cube
    x0, x1, x2 = np.meshgrid(np.linspace(0, ds[0] - 1, ds[0], dtype='uint16'),
                             np.linspace(0, ds[1] - 1, ds[1], dtype='uint16'),
                             np.linspace(0, ds[2] - 1, ds[2], dtype='uint16'),
                             indexing='ij')
    label_cube = label_cube.ravel()
    x0 = x0.ravel()
    x1 = x1.ravel()
    x2 = x2.ravel()
    x0 = x0[label_cube.astype('bool')]
    x1 = x1[label_cube.astype('bool')]
    x2 = x2[label_cube.astype('bool')]

    x0 = np.expand_dims(x0, 0)
    x1 = np.expand_dims(x1, 0)
    x2 = np.expand_dims(x2, 0)
    return np.concatenate((x0, x1, x2), axis=0)

def _linear_interpolation_3D(input_array, indices, boundary_correction = True, boundary_val=0, out_shape=None):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    output = np.empty(indices[0].shape)
    ind_0 = indices[0,:]
    ind_1 = indices[1,:]
    ind_2 = indices[2,:]

    N0, N1, N2 = input_array.shape

    x0_0 = ind_0.astype(np.integer)
    x1_0 = ind_1.astype(np.integer)
    x2_0 = ind_2.astype(np.integer)
    x0_1 = x0_0 + 1
    x1_1 = x1_0 + 1
    x2_1 = x2_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0)   | (x0_1 < 0)   | (x1_0 < 0)   | (x1_1 < 0)   | (x2_0 < 0)   | (x2_1 < 0) | \
                            (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1) | (x2_0 >= N2) | (x2_1 >= N2)

        x0_0[inds_out_of_range] = 0
        x1_0[inds_out_of_range] = 0
        x2_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0
        x1_1[inds_out_of_range] = 0
        x2_1[inds_out_of_range] = 0


    w0 = ind_0 - x0_0
    w1 = ind_1 - x1_0
    w2 = ind_2 - x2_0
    #Replace by this...
    #input_array.take(np.array([x0_0, x1_0, x2_0]))
    #For increased speed
    output = (input_array[x0_0, x1_0, x2_0] * (1 - w0) * (1 - w1) * (1 - w2) +
              input_array[x0_1, x1_0, x2_0] * np.abs(w0 * (1 - w1) * (1 - w2)) +
              input_array[x0_0, x1_1, x2_0] * (1 - w0) * w1 * (1 - w2) +
              input_array[x0_0, x1_0, x2_1] * (1 - w0) * (1 - w1) * w2 +
              input_array[x0_1, x1_0, x2_1] * w0 * (1 - w1) * w2 +
              input_array[x0_0, x1_1, x2_1] * (1 - w0) * w1 * w2 +
              input_array[x0_1, x1_1, x2_0] * w0 * w1 * (1 - w2) +
              input_array[x0_1, x1_1, x2_1] * w0 * w1 * w2)
    output.setflags(write=1)
    if boundary_correction:
        output[inds_out_of_range] = boundary_val

    if out_shape is not None:
        output = np.reshape(output, out_shape)

    return output

def _nearest_interpolation_3D(input_array, indices, boundary_correction = True, out_shape=None) :
    x_indices = indices[0,:].astype(np.integer)
    y_indices = indices[1,:].astype(np.integer)
    z_indices = indices[2,:].astype(np.integer)

    x0 = x_indices.astype(np.integer)
    x1 = y_indices.astype(np.integer)
    x2 = z_indices.astype(np.integer)

    if boundary_correction:
        N0, N1, N2 = input_array.shape
        # put all samples outside datacube to 0
        inds_out_of_range = (x0 < 0)  | (x1 < 0)  | (x2 < 0)  |\
                            (x0 >= N0)| (x1 >= N1)| (x2 >= N2)

        x0[inds_out_of_range] = 0
        x1[inds_out_of_range] = 0
        x2[inds_out_of_range] = 0


    output = input_array[x0, x1, x2]
    output.setflags(write=1)

    if boundary_correction:
        output[inds_out_of_range] = 0

    if out_shape is not None:
        output = np.reshape(output, out_shape)


    return output

def rotate_about_axis_3D(theta, axis, grid):
    theta = np.deg2rad(theta)
    #Make axis a vector
    if type(axis) == type(1):
        if axis == 0:
            axis = [1, 0, 0]
        elif axis==1:
            axis = [0, 1, 0]
        elif axis==2:
            axis = [0, 0, 1]
    axis = np.array(axis)

    rot_mat = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
    rot_mat  =np.expand_dims(rot_mat,2)
    grid = np.transpose( np.expand_dims(grid,2), [0,2,1])

    return np.einsum('ijk,jik->ik',rot_mat,grid)


########################################################################################################################
################################### 2D SUPPORT #########################################################################
def _nearest_interpolation_2D(input_array, indices, boundary_correction=True, boundary_val = 0, out_shape=None):

    x_indices = indices[0,:].astype(np.integer)
    y_indices = indices[1,:].astype(np.integer)

    x0 = x_indices.astype(np.integer )
    x1 = y_indices.astype(np.integer)

    if boundary_correction:
        N0, N1 = input_array.shape
        not_valids =  (x0 < 0) |  (x1 < 0)  |  (x0 >= N0) | (x1 >= N1)
        x0[not_valids] = 0;
        x1[not_valids] = 0;

    output = input_array[x0, x1]
    output.setflags(write=1)

    if boundary_correction:
        output[not_valids] = boundary_val

    if out_shape is not None:
        output = np.reshape(output, out_shape)


    return output


def linear_interpolation_2D(input_array, indices, outside_val=0, boundary_correction=True, out_shape=None):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    output = np.empty(indices[0].shape)
    ind_0 = indices[0,:]
    ind_1 = indices[1,:]

    N0, N1 = input_array.shape

    x0_0 = ind_0.astype(np.integer)
    x1_0 = ind_1.astype(np.integer)
    x0_1 = x0_0 + 1
    x1_1 = x1_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) | (x1_0 < 0) | (x1_1 < 0) |  \
                            (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1)

        x0_0[inds_out_of_range] = 0
        x1_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0
        x1_1[inds_out_of_range] = 0

    w0 = ind_0 - x0_0
    w1 = ind_1 - x1_0
    # Replace by this...
    # input_array.take(np.array([x0_0, x1_0, x2_0]))
    output = (input_array[x0_0, x1_0] * (1 - w0) * (1 - w1)  +
              input_array[x0_1, x1_0] * w0 * (1 - w1)  +
              input_array[x0_0, x1_1] * (1 - w0) * w1  +
              input_array[x0_1, x1_1] * w0 * w1 )

    output.setflags(write=1)


    if boundary_correction:
        output[inds_out_of_range] = 0

    if out_shape is not None:
        output = np.reshape(output, out_shape)

    return output

def rotate_2D(grid, theta):
    y = grid[1, :, :] * np.sin(theta) + grid[0, :, :] * np.cos(theta)
    x = grid[1, :, :] * np.cos(theta) - grid[0, :, :] * np.sin(theta)

    grid[0, :, :] = y
    grid[1, :, :] = x
    return grid
########################################################################################################################
###################################1D SUPPORT #########################################################################
def _nearest_interpolation_1D(input_array, indices, outside_val = 0, out_shape=None):
    x_indices = indices.astype(np.integer)

    x0 = x_indices.astype(np.integer )

    N0, N1 = input_array.shape
    not_valids = np.logical_or( x0 >N1, x0<0)

    x0[not_valids] = 0
    output = input_array[x0]
    output[not_valids] = outside_val

    if out_shape is not None:
        output = np.reshape(output, out_shape)


    return output

def _linear_interpolation_1D(input_array, indices, outside_val=0, boundary_correction=True, out_shape=None):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    ind_0 = indices[:]

    N0 = input_array.shape

    x0_0 = ind_0.astype(np.integer)
    x0_1 = x0_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) |   \
                            (x0_0 >= N0) | (x0_1 >= N0)

        x0_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0

    w0 = ind_0 - x0_0
    # Replace by this...
    # input_array.take(np.array([x0_0, x1_0, x2_0]))
    output = (input_array[x0_0] * (1 - w0)   +
              input_array[x0_1] * w0)

    if boundary_correction:
        output[inds_out_of_range] = 0

        if out_shape is not None:
            output = np.reshape(output, out_shape)

    return output


