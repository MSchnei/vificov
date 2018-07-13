#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:19:03 2018

@author: Marian
"""
import numpy as np
import scipy as sp
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
            lstPrmAry[indLst] = aryPrmSel

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


def rmp_deg_pixel_x_y_s(vecX, vecY, vecPrfSd, tplPngSize,
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

    return np.column_stack((vecXpxl, vecYpxl, vecPrfSdpxl))


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

    return aryGauss


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
