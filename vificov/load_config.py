# -*- coding: utf-8 -*-
"""Load py_pRF_mapping config file."""

import os
import csv
import ast

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def load_config(strCsvCnfg):
    """
    Load py_pRF_mapping config file.

    Parameters
    ----------
    strCsvCnfg : string
        Absolute file path of config file.

    Returns
    -------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.
    """
    # Print config parameters?
    lgcPrint = True

    # Dictionary with config information:
    dicCnfg = {}

    # Open file with parameter configuration:
    # fleConfig = open(strCsvCnfg, 'r')
    with open(strCsvCnfg, 'r') as fleConfig:

        # Read file  with ROI information:
        csvIn = csv.reader(fleConfig,
                           delimiter='\n',
                           skipinitialspace=True)

        # Loop through csv object to fill list with csv data:
        for lstTmp in csvIn:

            # Skip comments (i.e. lines starting with '#') and empty lines.
            # Note: Indexing the list (i.e. lstTmp[0][0]) does not work for
            # empty lines. However, if the first condition is no fullfilled
            # (i.e. line is empty and 'if lstTmp' evaluates to false), the
            # second logical test (after the 'and') is not actually carried
            # out.
            if lstTmp and not (lstTmp[0][0] == '#'):

                # Name of current parameter (e.g. 'varTr'):
                strParamKey = lstTmp[0].split(' = ')[0]
                # print(strParamKey)

                # Current parameter value (e.g. '2.94'):
                strParamVlu = lstTmp[0].split(' = ')[1]
                # print(strParamVlu)

                # Put paramter name (key) and value (item) into dictionary:
                dicCnfg[strParamKey] = strParamVlu

    # Extent of visual space from centre of the screen in negative x-direction
    # (i.e. from the fixation point to the left end of the screen) in degrees
    # of visual angle.
    dicCnfg['varXminDeg'] = float(dicCnfg['varXminDeg'])
    if lgcPrint:
        print('---Extent of visual space in negative x-direction: '
              + str(dicCnfg['varXminDeg']))

    # Extent of visual space from centre of the screen in positive x-direction
    # (i.e. from the fixation point to the right end of the screen) in degrees
    # of visual angle.
    dicCnfg['varXmaxDeg'] = float(dicCnfg['varXmaxDeg'])
    if lgcPrint:
        print('---Extent of visual space in positive x-direction: '
              + str(dicCnfg['varXmaxDeg']))

    # Extent of visual space from centre of the screen in negative y-direction
    # (i.e. from the fixation point to the lower end of the screen) in degrees
    # of visual angle.
    dicCnfg['varYminDeg'] = float(dicCnfg['varYminDeg'])
    if lgcPrint:
        print('---Extent of visual space in negative y-direction: '
              + str(dicCnfg['varYminDeg']))

    # Extent of visual space from centre of the screen in positive y-direction
    # (i.e. from the fixation point to the upper end of the screen) in degrees
    # of visual angle.
    dicCnfg['varYmaxDeg'] = float(dicCnfg['varYmaxDeg'])
    if lgcPrint:
        print('---Extent of visual space in positive y-direction: '
              + str(dicCnfg['varYmaxDeg']))

    # Size of visual space model in pixel along
    # x- and y-dimension.
    dicCnfg['tplVslSpcPix'] = tuple([int(dicCnfg['varXextPix']),
                                     int(dicCnfg['varYextPix'])])
    if lgcPrint:
        print('---Size of visual space model (x & y): '
              + str(dicCnfg['tplVslSpcPix']))

    # Path to nii files with parameter output (x-position, y-position, sigma):
    dicCnfg['lstPathNiiPrm'] = ast.literal_eval(dicCnfg['lstPathNiiPrm'])
    if lgcPrint:
        print('---Path(s) to nii file(s) with winner model parameters:')
        for strTmp in dicCnfg['lstPathNiiPrm']:
            print('   ' + str(strTmp))

    # Path to nii files with mask (region of interests like V1, V2, V3):
    dicCnfg['lstPathNiiMask'] = ast.literal_eval(dicCnfg['lstPathNiiMask'])
    if lgcPrint:
        print('---Path(s) to nii file(s) with region of interest masks:')
        for strTmp in dicCnfg['lstPathNiiMask']:
            print('   ' + str(strTmp))

    # Path to nii files with threshold map:
    dicCnfg['strPathNiiThr'] = ast.literal_eval(dicCnfg['strPathNiiThr'])
    if lgcPrint:
        print('---Path to nii files with threshold map:')
        print('   ' + str(dicCnfg['strPathNiiThr']))

    # Threshold value for threshold map:
    dicCnfg['varThr'] = float(dicCnfg['varThr'])
    if lgcPrint:
        print('---Threshold value for threshold map: '
              + str(dicCnfg['varThr']))

    # Number of bootstraps of the FOV:
    dicCnfg['varNumBts'] = int(dicCnfg['varNumBts'])
    if lgcPrint:
        print('---Number of bootstraps of the FOV: '
              + str(dicCnfg['varNumBts']))

    # Output basename:
    dicCnfg['strPathOut'] = ast.literal_eval(dicCnfg['strPathOut'])
    if lgcPrint:
        print('---Output basename:')
        print('   ' + str(dicCnfg['strPathOut']))

    return dicCnfg
