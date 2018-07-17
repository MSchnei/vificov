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
import matplotlib.pyplot as plt
from vificov.load_config import load_config
from vificov.vificov_utils import (cls_set_config, loadNiiDataExt, crt_fov,
                                   rmp_deg_pixel_xys, bootstrap_resample)


def run_vificov(strCsvCnfg):
    ###########################################################################
    ## debugging
    #strCsvCnfg = '/home/marian/Documents/Git/vificov/vificov/config_custom.csv'
    ###########################################################################
    # %% Load parameters and files

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Load x values, y and sigma values for all region of interests that were
    # provided as masks
    print('---Load provided parameter maps')
    lstPrmAry, objHdr, aryAff = loadNiiDataExt(cfg.lstPathNiiPrm,
                                               lstFlsMsk=cfg.lstPathNiiMask)

    # Deduce number of region of interest
    cfg.varNumRois = len(lstPrmAry)
    print('------Number of ROIs found: ' + str(cfg.varNumRois))

    # Load threshold map, if desired by user
    if cfg.strPathNiiThr:
        # Get threshold values
        lstThr = loadNiiDataExt([cfg.strPathNiiThr],
                                lstFlsMsk=cfg.lstPathNiiMask)[0]
        # Turn threshold values into boolean arrays by checking if they are
        # above the threshold specified by the user
        for ind, aryThr in enumerate(lstThr):
            lstThr[ind] = np.greater_equal(aryThr, cfg.varThr)

    # Apply threshold map, if desired by user
    if cfg.strPathNiiThr:
        print('---Exclude voxels based on threshold map')
        print('------Threshold is set to: ' + str(cfg.varThr))
        for ind, (aryPrm, aryThr) in enumerate(zip(lstPrmAry, lstThr)):
            # Check how many voxels before selection
            varNumVxlBfr = aryPrm.shape[0]
            # apply threshold boolean to exclude voxels
            lstPrmAry[ind] = aryPrm[aryThr, ...]
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
                                           int(cfg.varXminDeg),
                                           int(cfg.varXmaxDeg),
                                           int(cfg.varYminDeg),
                                           int(cfg.varYmaxDeg))

    # %% Create visual field coverage images
    print('---Create visual field coverage images')

    # prepare list for additive and maximum Gaussian output
    lstAddGss = [None] * len(lstPrmAry)
    lstMaxGss = [None] * len(lstPrmAry)

    # Loop over ROIs
    for indRoi, aryPrm in enumerate(lstPrmAry):
        print('------for ROI ' + str(indRoi+1))

        # Run functyion to create visual field coverage
        # Return both the reulst of the additive and maximum method
        aryAddGss, aryMaxGss = crt_fov(aryPrm, cfg.tplVslSpcPix)

        # Put outputs away to list
        lstAddGss[indRoi] = aryAddGss
        lstMaxGss[indRoi] = aryMaxGss

    # %% Bootstrap the visual field coverage, if desired by user
    
    if cfg.varNumBts > 0:
        print('---Create bootstrapped visual field coverage images')

        # prepare list for additive and maximum Gaussian output
        lstBtsAddGss = [None] * len(lstPrmAry)
        lstBtsMaxGss = [None] * len(lstPrmAry)

        # Loop over ROIs
        for indRoi, aryPrm in enumerate(lstPrmAry):
            print('------for ROI ' + str(indRoi+1))

            # initialize arrays that can function as accumulators of the
            # visual field coverage map created on every bootstrap fold
            aryBtsAddGss = np.zeros((cfg.tplVslSpcPix))
            aryBtsMaxGss = np.zeros((cfg.tplVslSpcPix))

            # get number of voxels in ROI
            varNumVxl = aryPrm.shape[0]
            for indFld in range(cfg.varNumBts):
                print('---------Run bootstrap fold ' + str(indFld+1))
                # get indices for voxels that will be sampled in this fold
                arySmpl = bootstrap_resample(np.arange(varNumVxl))
                # for the selected voxels, get the winner parameters
                aryPrmRsm = aryPrm[arySmpl, :]
                # for these winner parameters get the visual field coverage
                aryFldAddGss, aryFldMaxGss = crt_fov(aryPrmRsm,
                                                     cfg.tplVslSpcPix)
                # add aryFldAddGss and aryFldMaxGss up over folds
                aryBtsAddGss += aryFldAddGss
                aryBtsMaxGss += aryFldMaxGss

            # Put away the mean bootstrap visual field map for this particular
            # ROI
            lstBtsAddGss[indRoi] = np.divide(aryBtsAddGss,
                                             float(cfg.varNumBts))
            lstBtsMaxGss[indRoi] = np.divide(aryBtsMaxGss,
                                             float(cfg.varNumBts))

    # %% Save visual field coverage images to disk
    print('---Save visual field coverage images to disk')

    for ind in range(len(lstAddGss)):
        
        # get arrays
        aryAddGss = lstAddGss[ind]
        aryMaxGss = lstMaxGss[ind]
        aryBtsAddGss = lstBtsAddGss[ind]
        aryBtsMaxGss = lstBtsMaxGss[ind]

        # Derive file name
        strPthFln = os.path.basename(
            os.path.splitext(cfg.lstPathNiiMask[ind])[0])
        # Derive output path
        strPthImg = cfg.strPathOut + '_' + strPthFln

        # save arrays as images
        plt.imsave(strPthImg + '_FOV_add.png', aryAddGss, cmap='viridis',
                   format="png", vmin=0.0, vmax = np.percentile(aryAddGss, 95))
        plt.imsave(strPthImg + '_FOV_max.png', aryMaxGss, cmap='magma',
                   format="png", vmin=0.0, vmax = 1.0)

        plt.imsave(strPthImg + '_FOV_add_btsrp.png', aryBtsAddGss,
                   cmap='viridis', format="png", vmin=0.0,
                   vmax = np.percentile(aryAddGss, 95))
        plt.imsave(strPthImg + '_FOV_max_btsrp.png', aryBtsMaxGss,
                   cmap='magma', format="png", vmin=0.0, vmax = 1.0)

    # %% Print done statement.
    print('---Done.')
