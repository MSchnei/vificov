#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:14:53 2018

@author: Marian
"""

import numpy as np
from vificov.load_config import load_config
from vificov.vificov_utils import (cls_set_config, loadNiiDataExt,
                                   rmp_deg_pixel_x_y_s, crt_2D_gauss)

##############################################################################
## debugging
#strCsvCnfg = '/home/marian/Documents/Git/vificov/config_custom.csv'
##############################################################################

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

# %% recreate image
lstViFi = [None] * len(lstPrmAry)

for indRoi, aryPrm in enumerate(lstPrmAry):
    # prepare image
    aryViFi = np.zeros((cfg.tplVslSpcPix))
    # loop over ROIs
    for indVxl, vecVxlPrm in enumerate(aryPrm):
        # extract winner parameters for this voxel
        varPosX, varPosY, varSd = vecVxlPrm[0], vecVxlPrm[1], vecVxlPrm[2]
        # recreate the winner 2D Gaussian
        aryTmpGss = crt_2D_gauss(cfg.tplVslSpcPix[0], cfg.tplVslSpcPix[1],
                                 varPosX, varPosY, varSd)
        # add Gaussians for this region
        aryViFi += aryTmpGss

    # divide by total number of Gaussians that were added
    aryViFi /= aryPrm.shape[0]

    # put way to list
    lstViFi[indRoi] = aryViFi
