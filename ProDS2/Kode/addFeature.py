#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:13:49 2024

@author: kevinchristian
"""
import pandas as pd

def addNDVI(band4, band8):

    results = pd.Series([(a - b) / (a + b) for a, b in zip(band8, band4)])
    return results

def addEVI(band2, band4, band8):
    return 2.5 * ((band8/10000-band4/10000) / (band8/10000 + 6 * band4/10000 - 7.5 * band2/10000 + 1))

def addNDWI(band3, band8):
    results = pd.Series([(green - nir) / (green + nir) for green, nir in zip(band3, band8)])
    return results