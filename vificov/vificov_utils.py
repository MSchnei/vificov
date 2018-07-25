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


def prep_func(lstPathNiiMask, lstPathNiiFunc, strPrepro=None):
    """
    Load & prepare functional data.

    Parameters
    ----------
    lstPathNiiMask: list
        List of paths to masks used to restrict pRF model finding. Only voxels
        with a value other than zero in the mask are considered.
    lstPathNiiFunc : list
        List of paths of functional data (3D or 4D nii files).
    strPrepro : NoneType or string
        Flag to determine the preprocessing that will be performed on files.
        Accepeted options are: None, 'demean', 'psc', or 'zscore'.

    Returns
    -------
    aryLgcMsk : np.array
        3D numpy array with logial values. Externally supplied mask (e.g grey
        matter mask). Voxels that are `False` in the mask are excluded.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of mask nii data.
    lstFuncOut : list
        List containing 2D numpy arrays with prepared functional data, of the
        form aryFunc[voxelCount, time]. There will be as many 2D arrays as
        masks were provided in the list.
    tplNiiShp : tuple
        Spatial dimensions of input nii data (number of voxels in x, y, z
        direction). The data are reshaped during preparation, this
        information is needed to fit final output into original spatial
        dimensions.

    Notes
    -----
    Functional data is loaded from disk. The functional data is reshaped, into
    the form aryFunc[voxel, time]. A mask is applied (externally supplied, e.g.
    a grey matter mask). Subsequently, the functional data is pre-processed.
    """

    # prepare output list
    lstFuncOut = [None] * len(lstPathNiiMask)

    # loop over different masks that were provided by the user
    for indMsk, strPathNiiMask in enumerate(lstPathNiiMask):

        print('------Mask number ' + str(indMsk+1))

        # Load mask (to restrict model fitting) as boolean:
        aryLgcMsk, hdrMsk, aryAff = loadNiiData(strPathNiiMask, typPrc=np.bool)

        # Dimensions of nii data:
        tplNiiShp = aryLgcMsk.shape

        # List for arrays with functional data (possibly several runs):
        lstFunc = []

        # Number of runs:
        varNumRun = len(lstPathNiiFunc)

        # Loop through runs and load data:
        for idxRun in range(varNumRun):

            print(('---------Prepare run ' + str(idxRun + 1)))

            # Load 4D nii data:
            aryTmpFunc, _, _ = loadNiiData(lstPathNiiFunc[idxRun])

            # Apply mask:
            aryTmpFunc = aryTmpFunc[aryLgcMsk, ...]

            # make sure that aryTmpFunc is two-dimensional
            if len(aryTmpFunc.shape) == 1:
                aryTmpFunc = aryTmpFunc.reshape(-1, 1)

            # perform preprocessing, if desired by user
            if strPrepro == 'demean':
                # De-mean functional data:
                print('------------Demean')
                aryTmpFunc = np.subtract(
                    aryTmpFunc, np.mean(aryTmpFunc,
                                        axis=1,
                                        dtype=np.float32)[:, None])

            if strPrepro == 'psc':
                # Get percent signal change of functional data:
                print('------------Get percent signal change')
                aryTmpStd = np.std(aryTmpFunc, axis=-1)
                aryTmpMean = np.mean(aryTmpFunc, axis=-1)
                aryTmpLgc = np.greater(aryTmpStd, np.array([0.0]))
                aryTmpFunc[aryTmpLgc, :] = np.divide(
                    aryTmpFunc[aryTmpLgc, :],
                    aryTmpMean[aryTmpLgc, None]) * 100 - 100

            if strPrepro == 'zscore':
                # Score functional data:
                print('------------Zscore')
                aryTmpFunc = np.subtract(aryTmpFunc,
                                         np.mean(aryTmpFunc,
                                                 axis=1,
                                                 dtype=np.float32)[:, None])
                aryTmpStd = np.std(aryTmpFunc, axis=-1)
                aryTmpLgc = np.greater(aryTmpStd, np.array([0.0]))
                aryTmpFunc[aryTmpLgc, :] = np.divide(
                    aryTmpFunc[aryTmpLgc, :], aryTmpStd[aryTmpLgc, None])

            # Put prepared functional data of current run into list:
            lstFunc.append(aryTmpFunc)
            del(aryTmpFunc)

        # Put functional data from separate runs into one array. 2D array of
        # the form aryFunc[voxelCount, time]
        aryFunc = np.concatenate(lstFunc, axis=1).astype(np.float32,
                                                         copy=False)
        del(lstFunc)

        # Put functional array for this paricular mask away to output list
        lstFuncOut[indMsk] = aryFunc

    return aryLgcMsk, hdrMsk, aryAff, lstFuncOut, tplNiiShp


def loadNiiPrm(lstFunc, lstFlsMsk=None):
    """Load parameters from multiple nii files, with optional mask argument.

    Parameters
    ----------
    lstFunc : list,
        list of str with file names of 3D nii files
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
    varExtYmin : float
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
    varDgr2PixX = np.divide(tplPngSize[0], (varExtXmax - varExtXmin))
    varDgr2PixY = np.divide(tplPngSize[1], (varExtYmax - varExtYmin))

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

    # Prepare image for additive and max Gaussian
    # use np.rot90 to make sure array is compatible with result of crt_2D_gauss
    aryAddGss = np.rot90(np.zeros((tplVslSpcPix)), k=1)
    aryMaxGss = np.rot90(np.zeros((tplVslSpcPix)), k=1)

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


def crt_prj(aryPrm, aryStatsMap, tplVslSpcPix):

    # Prepare image stack for additive Gaussian and projection
    # use np.rot90 to make sure array is compatible with result of crt_2D_gauss
    aryAddGss = np.rot90(np.zeros((tplVslSpcPix), dtype=np.float32), k=1)
    aryAddPrj = np.rot90(np.zeros((tplVslSpcPix + (aryStatsMap.shape[-1],)),
                                  dtype=np.float32), k=1, axes=(0, 1))

    # Loop over voxels
    for indVxl, (vecVxlPrm, aryVxlMap) in enumerate(zip(aryPrm, aryStatsMap)):
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
            # Add Gaussians for this region
            aryAddGss += aryTmpGss
            # Create the projection of stats map into visual field
            aryTmpPrj = np.multiply(aryTmpGss[:, :, None],
                                    aryVxlMap)
            # Add projection for this region
            aryAddPrj += aryTmpPrj

    # Normalize the projection
    aryPrj = np.divide(aryAddPrj, aryAddGss[:, :, None])

    return aryPrj


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
