# -*- coding: utf-8 -*-
"""Main function for vificov, which is called from command line."""

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

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vificov.load_config import load_config
from vificov.vificov_utils import (cls_set_config, loadNiiPrm, crt_fov,
                                   rmp_deg_pixel_xys, bootstrap_resample,
                                   crt_cntr_dot, prep_func, crt_prj,
                                   shift_cmap)


def run_vificov(strCsvCnfg, lgcGnPrm=True):
    ###########################################################################
#    # debugging
    # strCsvCnfg = '/media/sf_D_DRIVE/MotionQuartet/Tools/P3/Prf/b_11b_config_vificov_2pred.csv'
    ###########################################################################
    # %% Load parameters and files

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Load x values, y and sigma values for all region of interests that were
    # provided as masks
    print('---Load provided parameter maps')
    lstPrmAry, objHdr, aryAff = loadNiiPrm(cfg.lstPathNiiPrm,
                                           lstFlsMsk=cfg.lstPathNiiMask)

    # Deduce number of region of interest
    cfg.varNumRois = len(lstPrmAry)
    print('------Number of ROIs found: ' + str(cfg.varNumRois))

    # Load threshold map, if desired by user
    if cfg.strPathNiiThr:
        # Get threshold values
        lstThr = loadNiiPrm([cfg.strPathNiiThr],
                            lstFlsMsk=cfg.lstPathNiiMask)[0]
        # Turn threshold values into boolean arrays by checking if they are
        # above the threshold specified by the user
        for ind, aryThr in enumerate(lstThr):
            lstThr[ind] = np.greater_equal(aryThr, cfg.varThr)

    # Load stats values, if desired by user
    if cfg.lstPathNiiStats[0]:
        print('---Load provided statistical maps')
        lstStatMaps = prep_func(cfg.lstPathNiiMask, cfg.lstPathNiiStats,
                                strPrepro=cfg.strPrepro)[3]

    # Apply threshold map, if desired by user
    if cfg.strPathNiiThr:
        print('---Exclude voxels based on threshold map')
        print('------Threshold is set to: ' + str(cfg.varThr))
        for ind, (aryPrm, aryThr) in enumerate(zip(lstPrmAry, lstThr)):
            # Check how many voxels before selection
            varNumVxlBfr = aryPrm.shape[0]
            # apply threshold boolean to exclude voxels
            lstPrmAry[ind] = aryPrm[aryThr, ...]
            # apply threshold to stats map, if they were provided
            if cfg.lstPathNiiStats[0]:
                aryMap = lstStatMaps[ind]
                lstStatMaps[ind] = aryMap[aryThr, ...]
            # Check how many voxels were excluded
            varNumVxlExl = varNumVxlBfr - lstPrmAry[ind].shape[0]
            # print number of voxels included and excluded
            print('------Number of voxels excluded in ROI ' + str(ind+1) +
                  ': ' + str(varNumVxlExl))

    # Check how many voxels are left in ROIs and provide info to user
    print('---Counting voxels in provided ROIs:')
    for ind, aryPrm in enumerate(lstPrmAry):
        # Check how many voxels before selection
        varNumVxlIncl = aryPrm.shape[0]
        print('------Number of voxels now included in ROI ' + str(ind+1) +
              ': ' + str(varNumVxlIncl))

    # Load information about spatial apertures that were presented in
    # experiment to obtain pRF parameters. This is needed to create the visual
    # field coverage according to Kay method.
    arySptExpInf = np.load(cfg.strSptExpInf)

    # %% Convert from degree to pixel

    # Convert parameter maps that were provided in degrees of visual angle
    # to parameters in pixels, since this will be the relevant unit for the
    # visual field projection

    for ind, aryPrm in enumerate(lstPrmAry):
        # remap values
        lstPrmAry[ind] = rmp_deg_pixel_xys(aryPrm[:, 0],
                                           aryPrm[:, 1],
                                           aryPrm[:, 2],
                                           cfg.tplVslSpcPix,
                                           cfg.varXminDeg,
                                           cfg.varXmaxDeg,
                                           cfg.varYminDeg,
                                           cfg.varYmaxDeg)

    # %% Create visual field coverage images
    print('---Create visual field coverage images')

    # Prepare list for additive and maximum Gaussian output
    lstAddGss = [None] * len(lstPrmAry)
    lstMaxGss = [None] * len(lstPrmAry)
    lstKayGss = [None] * len(lstPrmAry)

    # Loop over ROIs
    for indRoi, aryPrm in enumerate(lstPrmAry):
        print('------for ROI ' + str(indRoi+1))

        # Run function to create visual field coverage
        # Return both the result of the additive and maximum method
        aryAddGss, aryMaxGss, aryKayGss = crt_fov(aryPrm,
                                                  arySptExpInf,
                                                  cfg.tplVslSpcPix)

        # Put outputs away to list
        lstAddGss[indRoi] = aryAddGss
        lstMaxGss[indRoi] = aryMaxGss
        lstKayGss[indRoi] = aryKayGss

    # %% Create images with small dot for every pRF centre
    print('---Create images with small dot for every pRF centre')

    # Prepare list for images with small dot for every pRF centre
    lstCntrDts = [None] * len(lstPrmAry)

    # Loop over ROIs
    for indRoi, aryPrm in enumerate(lstPrmAry):
        print('------for ROI ' + str(indRoi+1))

        # Create images
        aryCntrDts = crt_cntr_dot(aryPrm, cfg.tplVslSpcPix, lgcNrm=True)

        # Put outputs away to list
        lstCntrDts[indRoi] = aryCntrDts


    # %% Bootstrap the visual field coverage, if desired by user

    if cfg.varNumBts > 0:
        print('---Create bootstrapped visual field coverage images')

        # prepare list for additive and maximum Gaussian output
        lstBtsAddGss = [None] * len(lstPrmAry)
        lstBtsMaxGss = [None] * len(lstPrmAry)
        lstBtsKayGss = [None] * len(lstPrmAry)

        # Loop over ROIs
        for indRoi, aryPrm in enumerate(lstPrmAry):
            print('------for ROI ' + str(indRoi+1))

            # initialize arrays that can function as accumulators of the
            # visual field coverage map created on every bootstrap fold
            # use np.rot90 to make sure array is compatible with result of
            # crt_fov and crt_2D_gauss

            aryBtsAddGss = np.rot90(np.zeros((cfg.tplVslSpcPix)), k=1)
            aryBtsMaxGss = np.rot90(np.zeros((cfg.tplVslSpcPix)), k=1)
            aryBtsKayGss = np.rot90(np.zeros((cfg.tplVslSpcPix)), k=1)

            # get number of voxels in ROI
            varNumVxl = aryPrm.shape[0]
            for indFld in range(cfg.varNumBts):
                print('---------Run bootstrap fold ' + str(indFld+1))
                # get indices for voxels that will be sampled in this fold
                arySmpl = bootstrap_resample(np.arange(varNumVxl))
                # for the selected voxels, get the winner parameters
                aryPrmRsm = aryPrm[arySmpl, :]
                # for these winner parameters get the visual field coverage
                aryFldAddGss, aryFldMaxGss, aryFldKayGss = crt_fov(
                    aryPrmRsm, arySptExpInf, cfg.tplVslSpcPix)
                # add aryFldAddGss, aryFldMaxGss and aryFldKayGss up over folds
                aryBtsAddGss += aryFldAddGss
                aryBtsMaxGss += aryFldMaxGss
                aryBtsKayGss += aryFldKayGss

            # Put away the mean bootstrap visual field map for this particular
            # ROI
            lstBtsAddGss[indRoi] = np.divide(aryBtsAddGss,
                                             float(cfg.varNumBts))
            lstBtsMaxGss[indRoi] = np.divide(aryBtsMaxGss,
                                             float(cfg.varNumBts))
            lstBtsKayGss[indRoi] = np.divide(aryBtsKayGss,
                                             float(cfg.varNumBts))

    # %% Project stats map into the visual field

    if cfg.lstPathNiiStats[0]:
        print('---Project stats map into the visual field')

        # Prepare list for normalized projections
        lstPrj = [None] * len(lstStatMaps)
        lstUnnrmPrj = [None] * len(lstStatMaps)
        lstNrmDen = [None] * len(lstStatMaps)

        # Loop over ROIs
        for indRoi, (aryPrm, aryMap) in enumerate(zip(lstPrmAry, lstStatMaps)):
            print('------for ROI ' + str(indRoi+1))

            aryPrj, aryUnnrmPrj, aryNrmDen = crt_prj(aryPrm, aryMap,
                                                     cfg.tplVslSpcPix)

            # Put projection away to list
            lstPrj[indRoi] = aryPrj
            lstUnnrmPrj[indRoi] = aryUnnrmPrj
            lstNrmDen[indRoi] = aryNrmDen

    # %% Save visual field coverage, and ptn boostrapped images and projections

    # Loop over different regions
    for ind in range(len(lstAddGss)):
        print('---Save files to disk for ROI ' + str(ind+1))

        # Derive file name
        strPthFln = os.path.basename(
            os.path.splitext(cfg.lstPathNiiMask[ind])[0])
        # if it was a nii.gz file, get rid of .nii leftover
        if strPthFln[-4:] == '.nii':
            strPthFln = strPthFln[:-4]

        # Derive output path
        strPthImg = cfg.strPathOut + '_' + strPthFln

        # %% Save visual field coverage
        print('------Save visual field coverage images')

        # get arrays
        aryAddGss = lstAddGss[ind]
        aryMaxGss = lstMaxGss[ind]
        aryKayGss = lstKayGss[ind]

        # Save visual field projections as images
        varVmin = 0.0

        # Save image from aryAddGss to disk
        plt.imsave(strPthImg + '_FOV_add.png', aryAddGss, cmap='plasma',
                   format="png", vmin=varVmin,
                   vmax=0.35)
        # Save image from aryMaxGss to disk
        plt.imsave(strPthImg + '_FOV_max.png', aryMaxGss, cmap='magma',
                   format="png", vmin=varVmin, vmax=1.0)
        # Save image from aryMaxGss to disk
        plt.imsave(strPthImg + '_FOV_kay.png', aryKayGss, cmap='bone',
                   format="png", vmin=varVmin, vmax=1.0)

#        # Generate figure
#        fig, ax = plt.subplots() # Create a figure with a single axes.
#        im = ax.imshow(aryKayGss, cmap='bone', vmin=varVmin, vmax=1.0)
#        # Display the image data
#        cbar = fig.colorbar(im)
#        fig.savefig(strPthImg + '_FOV_Kay_colorbar.svg')
#
#        # Generate figure
#        fig, ax = plt.subplots() # Create a figure with a single axes.
#        im = ax.imshow(aryAddGss, cmap='plasma', vmin=varVmin,
#                       vmax=0.35)
#        # Display the image data
#        cbar = fig.colorbar(im)
#        fig.savefig(strPthImg + '_FOV_add_colorbar.svg')


#        # Get pRF centre image
#        aryCntrDts = lstCntrDts[ind]
#        # Mask pixels that have no pRF centre
#        aryCntrDtsMsk = np.ma.masked_where(aryCntrDts < 1.0, aryCntrDts)
#
#        # Generate figure
#        fig = plt.figure(frameon=False)
#        ax = plt.Axes(fig, [0., 0., 1., 1.])
#        ax.set_axis_off()
#        fig.add_axes(ax)
#        im1 = ax.imshow(aryAddGss, cmap='bone', vmin=varVmin,
#                        vmax=varVmax)
#        ax.imshow(aryCntrDtsMsk, cmap='cool', vmin=0.0,
#                  vmax=1.0, extent=im1.get_extent())
#        fig.savefig(strPthImg + '_FOV_add_prfCentre.png')


        # %% Save bootstrapped visual field projections as images
        if cfg.varNumBts > 0:

            print('------Save bootstrapped visual field projections as images')
            aryBtsAddGss = lstBtsAddGss[ind]
            aryBtsMaxGss = lstBtsMaxGss[ind]
            aryBtsKayGss = lstBtsKayGss[ind]

            plt.imsave(strPthImg + '_FOV_add_btsrp.png', aryBtsAddGss,
                       cmap='plasma', format="png", vmin=0.0,
                       vmax=np.percentile(aryAddGss, 75))
            plt.imsave(strPthImg + '_FOV_max_btsrp.png', aryBtsMaxGss,
                       cmap='magma', format="png", vmin=0.0, vmax=1.0)
            plt.imsave(strPthImg + '_FOV_kay_btsrp.png', aryBtsKayGss,
                       cmap='bone', format="png", vmin=0.0, vmax=0.5)

        # %% Save projections of statistical maps as npz/nii/png files
        if cfg.lstPathNiiStats[0]:
            print('------Save projections of statistical maps')

            # get projections for this ROI
            aryPrj = lstPrj[ind]
            aryUnnrmPrj = lstUnnrmPrj[ind]
            aryNrmDen = lstNrmDen[ind]

            # save arrays in list as npz files
            print('------Save as npz files')
            np.savez(strPthImg, aryPrj=aryPrj, aryUnnrmPrj=aryUnnrmPrj,
                     aryNrmDen=aryNrmDen)

            # save arrays as nii file
            print('------Save as nii files')
            imgNii = nb.Nifti1Image(aryPrj, affine=np.eye(4))
            nb.save(imgNii, strPthImg + '.nii')

            # save projections as images
            print('------Save as png files')
            # get 5th percentile and 95th percentile to set limits to colormap
            varVmin = np.percentile(aryPrj.ravel(), 5, axis=0)
            varVmax = np.percentile(aryPrj.ravel(), 95, axis=0)
            print('---------Minimum threshold: ' + str(varVmin))
            print('---------Maximum threshold: ' + str(varVmax))

            # loop over different projections
            for indPrj in range(aryPrj.shape[-1]):
                # get particular image projection
                imaPrj = aryPrj[..., indPrj]
                if indPrj < 10:
                    strPrnt = '_prjIma_000' + str(indPrj)
                elif indPrj < 100:
                    strPrnt = '_prjIma_00' + str(indPrj)
                elif indPrj < 1000:
                    strPrnt = '_prjIma_0' + str(indPrj)
                elif indPrj < 10000:
                    strPrnt = '_prjIma_' + str(indPrj)
                # Create shifted colormap such that it centers on zero
                varCnt = 1 - varVmax / (varVmax + abs(varVmin))
                # calculate zero centre
                objCmpa = shift_cmap(cm.coolwarm, midpoint=varCnt)
                # save image
                plt.imsave(strPthImg + strPrnt + '.png', imaPrj,
                           cmap=objCmpa, format="png", vmin=varVmin,
                           vmax=varVmax)

    # %% Print done statement.
    print('---Done.')
