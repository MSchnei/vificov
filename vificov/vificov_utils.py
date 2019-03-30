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

import warnings
import matplotlib
import numpy as np
import scipy as sp
import nibabel as nb
import multiprocessing as mp
import matplotlib.pyplot as plt


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
        matter mask). Voxels that are 'False' in the mask are excluded.
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
                aryTmpMean = np.mean(aryTmpFunc, axis=-1)
                aryTmpLgc = np.greater(aryTmpMean, np.array([0.0]))
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


    # Values from the pyprf estimation assume scientific convention and
    # orientation of 2D Gaussian images with the origin in the lower left
    # corner. x-axis occupies width and y-axis occupies the height dimension.
    # We also assume that the first dimension that the user provides
    # indexes x and the second indexes the y-axis. Since python is column
    # major (i.e. first indexes columns, only then rows), we need to rotate
    # our array by 90 degrees rightward (k=3). This will insure that with
    # the 0th axis we index the scientific x-axis and higher values move us to
    # the right on that x-axis. It will also ensure that the 1st
    # python axis indexes the scientific y-axis and higher values will 
    # move us up. However, because of the way that matplotlib displays images,
    # where higher indices on the 0th axis are displayed lower, instead of
    # turning our array by 90 degrees rightward (k=3), we turn it leftward
    # (k=1). This effectively mirrors the image top-down. 

    aryGauss = np.rot90(aryGauss, k=1)

    return aryGauss


def cnvl_2D_gauss(idxPrc, aryMdlParamsChnk, arySptExpInf, tplPngSize, queOut):
    """Spatially convolve input with 2D Gaussian model.

    Parameters
    ----------
    idxPrc : int
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryMdlParamsChnk : 2d numpy array, shape [n_models, n_model_params]
        Array with the model parameter combinations for this chunk.
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2.
        Pixel dimensions of the visual space (width, height).
    queOut : multiprocessing.queues.Queue
        Queue to put the results on. If this is None, the user is not running
        multiprocessing but is just calling the function
    Returns
    -------
    data : 2d numpy array, shape [n_models, n_conditions]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = aryMdlParamsChnk.shape[0]

    # Number of conditions / time points of the input data
    varNumLstAx = arySptExpInf.shape[-1]

    # Output array with results of convolution:
    aryOut = np.zeros((varChnkSze, varNumLstAx))

    # Loop through combinations of model parameters:
    for idxMdl in range(0, varChnkSze):

        # Spatial parameters of current model:
        varTmpX = aryMdlParamsChnk[idxMdl, 0]
        varTmpY = aryMdlParamsChnk[idxMdl, 1]
        varTmpSd = aryMdlParamsChnk[idxMdl, 2]

        # Create pRF model (2D):
        aryGauss = crt_2D_gauss(tplPngSize[0],
                                tplPngSize[1],
                                varTmpX,
                                varTmpY,
                                varTmpSd)

        # Multiply pixel-time courses with Gaussian pRF models:
        aryCndTcTmp = np.multiply(arySptExpInf, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'.
        aryCndTcTmp = np.sum(aryCndTcTmp, axis=(0, 1))

        # Put model time courses into function's output with 2d Gaussian
        # arrray:
        aryOut[idxMdl, :] = aryCndTcTmp

    if queOut is None:
        # if user is not using multiprocessing, return the array directly
        return aryOut

    else:
        # Put column with the indices of model-parameter-combinations into the
        # output array (in order to be able to put the pRF model time courses
        # into the correct order after the parallelised function):
        lstOut = [idxPrc,
                  aryOut]

        # Put output to queue:
        queOut.put(lstOut)


def crt_mdl_rsp(arySptExpInf, tplPngSize, aryMdlParams, varPar):
    """Create responses of 2D Gauss models to spatial conditions.

    Parameters
    ----------
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space (width, height).
    aryMdlParams : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, 3]
        Model parameters (x, y, sigma) for all models.
    varPar : int, positive
        Number of cores to parallelize over.

    Returns
    -------
    aryMdlCndRsp : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions.

    """

    if varPar == 1:
        # if the number of cores requested by the user is equal to 1,
        # we save the overhead of multiprocessing by calling aryMdlCndRsp
        # directly
        aryMdlCndRsp = cnvl_2D_gauss(0, aryMdlParams, arySptExpInf,
                                     tplPngSize, None)

    else:

        # The long array with all the combinations of model parameters is put
        # into separate chunks for parallelisation, using a list of arrays.
        lstMdlParams = np.array_split(aryMdlParams, varPar)

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Empty list for results from parallel processes (for pRF model
        # responses):
        lstMdlTc = [None] * varPar

        # Empty list for processes:
        lstPrcs = [None] * varPar

        print('---------Running parallel processes')

        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvl_2D_gauss,
                                         args=(idxPrc, lstMdlParams[idxPrc],
                                               arySptExpInf, tplPngSize, queOut
                                               )
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Start processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(0, varPar):
            lstMdlTc[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].join()

        print('---------Collecting results from parallel processes')
        # Put output arrays from parallel process into one big array
        lstMdlTc = sorted(lstMdlTc)
        aryMdlCndRsp = np.empty((0, arySptExpInf.shape[-1]))
        for idx in range(0, varPar):
            aryMdlCndRsp = np.concatenate((aryMdlCndRsp, lstMdlTc[idx][1]),
                                          axis=0)

        # Clean up:
        del(lstMdlParams)
        del(lstMdlTc)

    return aryMdlCndRsp.astype(np.float32)


def get_strd_ind(varInd, varVslSpcPix, varStrdWdth=1):
    """Given an index, create striding window for indexing.

    Parameters
    ----------
    varInd : integer
        Integer on which the striding window will center.
    varVslSpcPix : integer
        Integer with the width or height of the visual field in pixel.
    varStrdWdth : integer
        Integer that describes the width of the striding window.

    Returns
    -------
    lstStrdInd : list
        List of striding indices.

    """
    
    # Make sure varInd and varStrdWdth are integers
    varInd = int(varInd)
    varStrdWdth = int(varStrdWdth)
    
    # Create list around centre index.
    lstStrdInd = np.arange(varInd-varStrdWdth, varInd+varStrdWdth+1)
    
    # Exclude indices that go out of the image array
    lstStrdInd = lstStrdInd[np.logical_and(lstStrdInd >= 0,
                                           lstStrdInd < varVslSpcPix)]
    
    return lstStrdInd


def get_bin_prf_ima(tplCentre, tplVslSpcPix, varSd=1):
    """Create create binary pRF image.

    Parameters
    ----------
    tplCentre : tuple
        Pixel on which the pRF centers.
    tplVslSpcPix : tuple
        Tuple with the width and height of the visual field in pixel.
    varSd : integer
        Integer that describes the size of the pRF in pixel.

    Returns
    -------
    aryBinPrfIma : 2D numpy array
        Binary pRF image.

    """
    # Sort input and make sure it is integer value
    varPosX, varPosY = int(tplCentre[0]), int(tplCentre[1])
    varSizeX, varSizeY = int(tplVslSpcPix[0]), int(tplVslSpcPix[1])

    # Create x and y in meshgrid:
    aryX, aryY = sp.mgrid[0:varSizeX, 0:varSizeY]

    # The actual creation of the Gaussian array:    
    aryR = np.sqrt((aryX - varPosX)**2+(aryY - varPosY)**2)
    
    return np.less_equal(aryR, varSd).astype(np.int8)



def crt_cntr_dot(aryPrm, tplVslSpcPix, lgcNrm=False):
    """Create image for pRF centre dots.
    
    Parameters
    ----------
    aryPrm : 2D numpy array, shape [number of voxels, 3]
        Array with x, y, and sigma winner parameters for all voxels included in
        a given ROI
    tplVslSpcPix : tuple
        Tuple with the (width, height) of the visual field in pixel.
    lgcNrm : boolean
        Should returned array be normalizes such that max value is 1?

    Returns
    -------
    aryCntrDts : 2D numpy array
        Image of pRF centers. Values of 1 where centre and zero values
        elsewhere.

    """

    # Prepare image for pRF centre dots
    aryCntrDts = np.zeros((tplVslSpcPix), dtype=np.int8)
    
    for indVxl, vecVxlPrm in enumerate(aryPrm):
        # Extract x and y winner parameters for this voxel
        varPosX, varPosY = vecVxlPrm[0], vecVxlPrm[1]
        
        # Add 1 to pixel at pRF centre (and, if desired, surrounding pixels)
        aryCntrDts[get_strd_ind(varPosX, tplVslSpcPix[0])[:, np.newaxis],
                   get_strd_ind(varPosY, tplVslSpcPix[1])] += 1

    # Use np.rot90 to make sure array is compatible with result of crt_2D_gauss
    aryCntrDts = np.rot90(aryCntrDts, k=1)
    
    # If desired by user, normalize the array
    if lgcNrm:
        aryCntrDts[aryCntrDts>0] = 1
    
    return aryCntrDts


def crt_fov(aryPrm, arySptExpInf, tplVslSpcPix):
    """Create field of view for given winner x,y,sigma parameters.

    Parameters
    ----------
    aryPrm : 2D numpy array, shape [number of voxels, 3]
        Array with x, y, and sigma winner parameters for all voxels included in
        a given ROI.
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
        This is required to determine maximum response of pRF for Kay method.
    tplVslSpcPix : tuple
        Tuple with the (width, height) of the visual field in pixel.

    Returns
    -------
    aryAddGss : 2d numpy array, shape [width, height]
        Visual field coverage using the additive method.
    aryMaxGss : 2d numpy array, shape [width, height]
        Visual field coverage using maximum method with normalized Gaussian.
    aryKayGss : 2d numpy array, shape [width, height]
        Visual field coverage using Kay method.

    Notes
    ----------
    [1] Each of the returned arrays fro visual field coverage is created
        according to a different emthod that was described in the literature.
        See references, which are in respective order.

    References
    ----------
    [1] Kok, P., Bains, L. J., Van Mourik, T., Norris, D. G., & De Lange, F. P.
        (2016). Selective activation of the deep layers of the human primary
        visual cortex by top-down feedback. Current Biology, 26(3), 371–376.
    [2] Amano, K., Wandell, B. A., & Dumoulin, S. O. (2009). Visual field maps,
        population receptive field sizes, and visual field coverage in the
        human MT+ complex. Journal of Neurophysiology, 102(5), 2704–18.
    [3] Kay, K. N., Weiner, K. S., & Grill-Spector, K. (2015). Attention
        reduces spatial uncertainty in human ventral temporal cortex.
        Current Biology, 25(5), 595–600.

    """

    # Prepare image for additive and max Gaussian
    # use np.rot90 to make sure array is compatible with result of crt_2D_gauss
    aryAddGss = np.rot90(np.zeros((tplVslSpcPix)), k=1)
    aryMaxGss = np.rot90(np.zeros((tplVslSpcPix)), k=1)
    aryKayGss = np.rot90(np.zeros((tplVslSpcPix)), k=1)

    # Upsample spatial information if necessary
    varFctUps = np.divide(tplVslSpcPix[0], arySptExpInf.shape[0])
    errorMsg = 'Desired pixel size needs to be multiple of arySptExpInf dim.'
    assert varFctUps % varFctUps == 0.0, errorMsg
    varFctUps = int(varFctUps)
    arySptExpInfUps = np.kron(arySptExpInf, np.ones((varFctUps, varFctUps, 1),
                              dtype=arySptExpInf.dtype))
    # Get maximum predicted response for the stimuli in the model fitting
    vecMaxRsp = np.max(crt_mdl_rsp(arySptExpInfUps, tplVslSpcPix, aryPrm, 10),
                       axis=1)

    # Loop over voxels
    varDivCnt = 0
    for indVxl, vecVxlPrm in enumerate(aryPrm):
        # Extract winner parameters for this voxel
        varPosX, varPosY, varSd = vecVxlPrm[0], vecVxlPrm[1], vecVxlPrm[2]
        # Do not continue the for-loop for voxels that have a standard
        # deviation of 0 pixels
        if np.isclose(varSd, 0, atol=1e-04):
            warnings.warn("Voxel skipped because SD equals 0")
        else:
            # Recreate the winner 2D Gaussian
            aryTmpGss = crt_2D_gauss(tplVslSpcPix[0], tplVslSpcPix[1],
                                     varPosX, varPosY, varSd)
            if np.sum(np.isnan(aryTmpGss)) > 1:
                warnings.warn("NaN value encountered in 2D Gaussian")


            # Normalize such that the maximum pixel has value 1.0
            aryTmpGssNrm = np.divide(aryTmpGss, aryTmpGss.max())
            
            # Add Gaussians for this region
            aryAddGss += aryTmpGssNrm

            # Find pixels where value has never been that high before
            lgcMaxGss = np.greater(aryTmpGssNrm, aryMaxGss)
            # Copy values for those pixels
            aryMaxGss[lgcMaxGss] = np.copy(aryTmpGssNrm[lgcMaxGss])
            
            # Implement Kay method
            aryKayGss += np.divide(get_bin_prf_ima((varPosX, varPosY),
                                                   tplVslSpcPix,
                                                   varSd=2*varSd),
                                   np.sqrt(vecMaxRsp[indVxl]))

            # Add to division couner
            varDivCnt += 1

    # Divide by total number of Gaussians that were included
    aryAddGss /= varDivCnt
    aryKayGss /= varDivCnt

    return aryAddGss, aryMaxGss, aryKayGss


def calc_ovlp(aryPrm, lstTmplIma, tplVslSpcPix):
    """Calculate overlap between 2d Gauss for given winner x,y,sigma
       parameters and given input images in a list.

    Parameters
    ----------
    aryPrm : 2D numpy array, shape [number of voxels, 3]
        Array with x, y, and sigma winner parameters for all voxels included in
        a given ROI
    lstTmplIma : list
        List with images. Each image should be a 2D numpy array with same
        dimensions as tplVslSpcPix.
    tplVslSpcPix : tuple
        Tuple with the (width, height) of the visual field in pixel.

    Returns
    -------
    aryOvlp : numpy array, shape [number of voxels, number of images in list]
       Overlap between 2d Gauss for given winner x,y,sigma parameters and input
       images.

    Notes
    -------
    [1] This is a helper function that is currently not used within the vificov
        package itself but was needed for a different analysis.

    """

    # Prepare array for resulting overlap results
    aryOvlp = np.zeros((aryPrm.shape[0], len(lstTmplIma)), dtype=np.float32)
    
    # loop over input images in list
    for indIma, imaTmpl in enumerate(lstTmplIma):
        # loop over voxels
        for indVxl, vecVxlPrm in enumerate(aryPrm):
            # Extract winner parameters for this voxel
            varPosX, varPosY, varSd = vecVxlPrm[0], vecVxlPrm[1], vecVxlPrm[2]
            # Recreate the winner 2D Gaussian
            aryTmpGss = crt_2D_gauss(tplVslSpcPix[0], tplVslSpcPix[1],
                                     varPosX, varPosY, varSd)
            # calculate the overlap with
            aryOvlp[indVxl, indIma] = np.sum(
                np.multiply(aryTmpGss, imaTmpl.astype(np.int8)), axis=(0, 1))

    return aryOvlp


def crt_prj(aryPrm, aryStatsMap, tplVslSpcPix):
    """Create projection of statistical map(s) into visual field.

    Parameters
    ----------
    aryPrm : 2D numpy array, shape [number voxels, 3]
        Array with x, y, and sigma winner parameters for all voxels included in
        a given ROI
    aryStatsMap : 2D numpy array, shape [number voxels, time / number of maps]
        2D numpy array with stats map / prepared functional data.
    tplVslSpcPix : tuple
        Tuple with the (width, height) of the visual field in pixel.

    Returns
    -------
    aryPrj : 3D numpy array, shape [tplVslSpcPix, time / number of maps]
       Projection of statistical map(s) into visual field.
    aryAddPrj : 3D numpy array, shape [tplVslSpcPix, time / number of maps]
       Unnormalized projection of statistical map(s) into visual field.
    aryAddGss : 2D numpy array, shape [tplVslSpcPix]
       Normalizing denominator of map(s) projection into visual field.

    """

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
    # The 1 is added to make the normalization stable, otherwise in areas of
    # the visual field that are not covered by any voxels division would be by
    # a number close to zero, resulting in extremely large numbers
#    aryPrj = np.divide(aryAddPrj, np.add(aryAddGss, 1)[:, :, None])
    aryPrj = np.divide(aryAddPrj, aryAddGss[:, :, None])


    return aryPrj, aryAddPrj, aryAddGss


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


def shift_cmap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
      cmap : matplotlib colormap object
          The matplotlib colormap to be altered.
      start : float, default 0.0
          Offset from lowest point in the colormap's range.
          Should be between 0.0 and 'midpoint'.
      midpoint : float, default 0.5
          The new center of the colormap. Defaults of 0.5 means no shift.
          Should be between 0.0 and 1.0. In general, this should be
          1 - vmax / (vmax + abs(vmin)). For example if your data range from
          -15.0 to +5.0 and you want the center of the colormap at 0.0,
          'midpoint' should be set to  1 - 5/(5 + 15)) or 0.75
      stop : float, default 1.0
          Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          'midpoint' and 1.0.

    Returns
    -------
    newcmap : matplotlib colormap object
          The new matplotlib colormap.

    References
    -------
    [1] Taken from stackoverflow:
        https://stackoverflow.com/questions/7404116/defining-the-midpoint-
        of-a-colormap-in-matplotlib

    """

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
        }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class cls_set_config(object):
    """
    Set config parameters from dictionary into local namespace.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, 'dicCnfg['varTr']' contains a float, such as
        '2.94'.

    """

    def __init__(self, dicCnfg):
        """Set config parameters from dictionary into local namespace."""
        self.__dict__.update(dicCnfg)
