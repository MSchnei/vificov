# -*- coding: utf-8 -*-
"""All utilities for vificov."""

# Visual Field Coverage (ViFiCov) visualization in python.

# Part of vificov library
# Copyright (C) 2018  Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy as sp
import warnings
import nibabel as nb


def loadNiiData(strPathNii, typPrc=None):
    """Load nii data from a single nii file.

    Parameters
    ----------
    strPathNii : str,
        Path to nii file
    typPrc: type or nontype, optional
        Precision with which nii data should be loaded
    Returns
    -------
    aryDataNii : numpy array
        Array with the nii data.
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    """
    # Load nii file:
    objNii = nb.load(strPathNii)

    # Load data into array:
    if typPrc is None:
        aryDataNii = np.asarray(objNii.dataobj)
    else:
        aryDataNii = np.asarray(objNii.dataobj).astype(typPrc)

    # Get headers:
    objHdr = objNii.header

    # Get 'affine':
    aryAff = objNii.affine

    return aryDataNii, objHdr, aryAff


def loadNiiDataExt(lstFunc,
                   lstFlsMsk=None):
    """Load nii data from multiple nii files, with optional mask argument.

    Parameters
    ----------
    lstFunc : list,
        list of str with file names of nii files
    lstFlsMsk : list, optional
        list of str with paths to 3D nii files that can act as mask/s
    Returns
    -------
    lstPrmAry : list
        The list will contain as many numpy arrays as masks were provided.
        Each array is 2D with shape [nr voxel in mask, nr nii files in lstFunc]
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    """

    # load parameter/functional maps into a list
    lstPrm = [None] * len(lstFunc)
    for ind, path in enumerate(lstFunc):
        aryFnc = loadNiiData(path, typPrc=np.float32)[0]
        lstPrm[ind] = aryFnc

    # load mask/s if available
    if lstFlsMsk is not None:
        lstMsk = [None] * len(lstFlsMsk)
        for ind, path in enumerate(lstFlsMsk):
            aryMsk = loadNiiData(path, typPrc=np.bool)[0]
            lstMsk[ind] = aryMsk
    else:
        print('------------No masks were provided')

    if lstFlsMsk is None:
        # if no mask was provided we just flatten all parameter array in list
        # and return resulting list
        lstPrmAry = [ary.flatten() for ary in lstPrm]
    else:
        # if masks are available, we loop over masks and then over parameter
        # maps to extract selected voxels and parameters
        lstPrmAry = [None] * len(lstFlsMsk)
        for indLst, aryMsk in enumerate(lstMsk):
            # prepare array that will hold parameter values of selected voxels
            aryPrmSel = np.empty((np.sum(aryMsk), len(lstFunc)),
                                 dtype=np.float32)
            # loop over different parameter maps
            for indAry, aryPrm in enumerate(lstPrm):
                # get voxels specific to this mask
                aryPrmSel[:, indAry] = aryPrm[aryMsk, ...]
            # put array away in list, if only one parameter map was provided
            # the output will be squeezed
            lstPrmAry[indLst] = np.squeeze(aryPrmSel)

    # also get header object and affine array
    # we simply take it for the first functional nii file, cause that is the
    # only file that has to be provided by necessity
    objHdr, aryAff = loadNiiData(lstFunc[0])[1:]

    return lstPrmAry, objHdr, aryAff


def rmp_rng(aryVls, varNewMin, varNewMax, varOldThrMin=None,
            varOldAbsMax=None):
    """Remap values in an array from one range to another.

    Parameters
    ----------
    aryVls : 1D numpy array
        Array with values that need to be remapped.
    varNewMin : float
        Desired minimum value of new, remapped array.
    varNewMax : float
        Desired maximum value of new, remapped array.
    varOldThrMin : float
        Theoretical minimum of old distribution. Can be specified if this
        theoretical minimum does not occur in empirical distribution but
        should be considered nontheless.
    varOldThrMin : float
        Theoretical maximum of old distribution. Can be specified if this
        theoretical maximum does not occur in empirical distribution but
        should be considered nontheless.

    Returns
    -------
    aryVls : 1D numpy array
        Array with remapped values.

    """
    if varOldThrMin is None:
        varOldMin = aryVls.min()
    else:
        varOldMin = varOldThrMin
    if varOldAbsMax is None:
        varOldMax = aryVls.max()
    else:
        varOldMax = varOldAbsMax

    aryNewVls = np.empty((aryVls.shape), dtype=aryVls.dtype)
    for ind, val in enumerate(aryVls):
        aryNewVls[ind] = (((val - varOldMin) * (varNewMax - varNewMin)) /
                          (varOldMax - varOldMin)) + varNewMin

    return aryNewVls


def rmp_deg_pixel_xys(vecX, vecY, vecPrfSd, tplPngSize,
                      varExtXmin, varExtXmax, varExtYmin, varExtYmax):
    """Remap x, y, sigma parameters from degrees to pixel.

    Parameters
    ----------
    vecX : 1D numpy array
        Array with possible x parametrs in degree
    vecY : 1D numpy array
        Array with possible y parametrs in degree
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in degree
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space in pixel (width, height).
    varExtXmin : float
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float
        Extent of visual space from centre in positive x-direction (width)
    varExtYmin : int
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float
        Extent of visual space from centre in positive y-direction (height)
    Returns
    -------
    vecX : 1D numpy array
        Array with possible x parametrs in pixel
    vecY : 1D numpy array
        Array with possible y parametrs in pixel
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in pixel

    """
    # Remap modelled x-positions of the pRFs:
    vecXpxl = rmp_rng(vecX, 0.0, (tplPngSize[0] - 1), varOldThrMin=varExtXmin,
                      varOldAbsMax=varExtXmax)

    # Remap modelled y-positions of the pRFs:
    vecYpxl = rmp_rng(vecY, 0.0, (tplPngSize[1] - 1), varOldThrMin=varExtYmin,
                      varOldAbsMax=varExtYmax)

    # We calculate the scaling factor from degrees of visual angle to
    # pixels separately for the x- and the y-directions (the two should
    # be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in ' + \
        'stimulus space (in degrees of visual angle) and the ' + \
        'ratio of X and Y dimensions in the upsampled visual space' + \
        'do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Convert prf sizes from degrees of visual angles to pixel
    vecPrfSdpxl = np.multiply(vecPrfSd, varDgr2PixX)
    
    # Return new values in column stack.
    # Since values are now in pixel, they should be integer
    return np.column_stack((vecXpxl, vecYpxl, vecPrfSdpxl)).astype(np.int32)


def crt_2D_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field in pixel.
    varSizeY : int, positive
        Height of the visual field in pixel.
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1] mathworld.wolfram.com/GaussianFunction.html

    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # create x and y in meshgrid:
    aryX, aryY = sp.mgrid[0:varSizeX, 0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX)) + np.square((aryY - varPosY))) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * np.square(varSd))
    
    # because we assume later (when plugging in the winner parameters) that the
    # origin of the created 2D Gaussian was in the lower left and that the
    # first axis of the array indexes the left-right direction of the screen
    # and the second axis indexes the the top-down direction of the screen,
    # we rotate by 90 degrees clockwise
    aryGauss = np.rot90(aryGauss, k=1)

    return aryGauss


def crt_fov(aryPrm, tplVslSpcPix):
    """Create field of view for given winner x,y,sigma parameters.

    Parameters
    ----------
    aryPrm : 2D numpy array, shape [number of voxels, 3]
        Array with x, y, and sigma winner parameters for all voxels included in
        a given ROI
    tplVslSpcPix : tuple
        Tuple with the (width, height) of the visual field in pixel.

    Returns
    -------
    aryAddGss : 2d numpy array, shape [width, height]
        Visual field coverage using the additive method.
    aryMaxGss : 2d numpy array, shape [width, height]
        Visual field coverage using maximum method.

    References
    -------
    [1]

    """

    # Prepare image for additive Gaussian
    aryAddGss = np.zeros((tplVslSpcPix))
    # Prepare image for max Gaussian
    aryMaxGss = np.zeros((tplVslSpcPix))

    # Loop over voxels
    varDivCnt = 0
    for indVxl, vecVxlPrm in enumerate(aryPrm):
        # Extract winner parameters for this voxel
        varPosX, varPosY, varSd = vecVxlPrm[0], vecVxlPrm[1], vecVxlPrm[2]
        # Do not continue the for-loop for voxels that have a standard
        # deviation of 0 pixels
        if np.isclose(varSd, 0, atol=1e-04):
            continue
        else:
            # Recreate the winner 2D Gaussian
            aryTmpGss = crt_2D_gauss(tplVslSpcPix[0], tplVslSpcPix[1],
                                     varPosX, varPosY, varSd)
            if np.sum(np.isnan(aryTmpGss)) > 1:
                warnings.warn("NaN value encountered in 2D Gaussian")
            # Add Gaussians for this region
            aryAddGss += aryTmpGss
            # Replace pixels for which aryTmpGss is greater than aryMaxGss
            aryTmpGssNrm = np.divide(aryTmpGss, aryTmpGss.max())
            lgcMaxGss = np.greater(aryTmpGssNrm, aryMaxGss)
            aryMaxGss[lgcMaxGss] = np.copy(aryTmpGssNrm[lgcMaxGss])
            # Add to division couner
            varDivCnt += 1

    # Divide by total number of Gaussians that were included
    aryAddGss /= varDivCnt
    
    return aryAddGss, aryMaxGss


def bootstrap_resample(aryX, varLen=None):
    """Perform resampling via bootstrapping for an input array.

    Parameters
    ----------
    aryX : 1D numpy array
        Data to be resampled.
    varLen : int, optional
        Length of bootsrapped sample. Equal to len(aryX) if varLen==None.

    Returns
    -------
    aryRsm : 1D numpy array
        Bootstrapped sample of the input array.

    References
    -------
    [1] Modified from: https://gist.github.com/aflaxman/6871948
    
    """
    if varLen == None:
        varLen = len(aryX)
        
    resample_i = np.floor(np.random.rand(varLen)*len(aryX)).astype(int)
    aryRsm = aryX[resample_i]
    return aryRsm


class cls_set_config(object):
    """
    Set config parameters from dictionary into local namespace.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.

    """

    def __init__(self, dicCnfg):
        """Set config parameters from dictionary into local namespace."""
        self.__dict__.update(dicCnfg)
