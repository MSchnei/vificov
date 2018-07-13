#!/usr/bin/env python2
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
from vificov.vificov_utils import (cls_set_config, loadNiiDataExt,
                                   rmp_deg_pixel_x_y_s, crt_2D_gauss)


def run_vificov(strCsvCnfg):
    ###########################################################################
    ## debugging
    #strCsvCnfg = '/home/marian/Documents/Git/vificov/config_custom.csv'
    ###########################################################################
    # %% Load parameters and files

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # load x values, y and sigma values for all region of interests that were
    # provided as masks
    lstPrmAry, objHdr, aryAff = loadNiiDataExt(cfg.lstPathNiiPrm,
                                               lstFlsMsk=cfg.lstPathNiiMask)

    # deduce number of region of interest
    cfg.varNumRois = len(lstPrmAry)

    # %% convert from degree to pixel

    # convert parameter maps gthat were provided in degrees of visual angle
    # to parameters in pixels, since this will be the relevant unit for the
    # visual field projection

    for ind, aryPrm in enumerate(lstPrmAry):
        # remap values
        lstPrmAry[ind] = rmp_deg_pixel_x_y_s(aryPrm[:, 0], aryPrm[:, 1],
                                             aryPrm[:, 2],
                                             cfg.tplVslSpcPix,
                                             int(cfg.varXminDeg),
                                             int(cfg.varXmaxDeg),
                                             int(cfg.varYminDeg),
                                             int(cfg.varYmaxDeg))

    # %% create visual field coverage images
    lstViFi = [None] * len(lstPrmAry)

    # loop over ROIs
    for indRoi, aryPrm in enumerate(lstPrmAry):

        # prepare image
        aryViFi = np.zeros((cfg.tplVslSpcPix))
        # summary image
        allImas = np.zeros((cfg.tplVslSpcPix[0], cfg.tplVslSpcPix[1],
                            aryPrm.shape[0]))

        # loop over voxels
        counter = 0
        for indVxl, vecVxlPrm in enumerate(aryPrm):
            # extract winner parameters for this voxel
            varPosX, varPosY, varSd = vecVxlPrm[0], vecVxlPrm[1], vecVxlPrm[2]
            if np.isclose(varSd, 0, atol=1e-04):
                pass
            else:
                # recreate the winner 2D Gaussian
                aryTmpGss = crt_2D_gauss(cfg.tplVslSpcPix[0],
                                         cfg.tplVslSpcPix[1],
                                         varPosX, varPosY, varSd)
                if np.sum(np.isnan(aryTmpGss)) > 0:
                    print('NaN NaN NaN in voxel ' + str(indVxl))
                # add Gaussians for this region
                aryViFi += aryTmpGss
                # add to couner
                counter += 1

            # put away allImas
            allImas[..., indVxl] = aryTmpGss

        # divide by total number of Gaussians that were added
        # aryViFi /= aryPrm.shape[0]
        aryViFi /= counter

        # put way to list
        lstViFi[indRoi] = aryViFi

    # save visual field coverage images to disk
    for ind, aryViFi in enumerate(lstViFi):

        strPthFln = os.path.basename(
            os.path.splitext(cfg.lstPathNiiMask[ind])[0])

        strPthImg = cfg.strPathOut + '_' + strPthFln
        plt.imsave(strPthImg, aryViFi, cmap='viridis', format="png")
