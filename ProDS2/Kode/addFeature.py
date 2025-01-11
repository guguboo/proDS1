#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:13:49 2024

@author: kevinchristian
"""
import pandas as pd

def addNDVI(band4, band8):

    results = pd.Series([(a - b) / (a + b) if (a + b) != 0 else 0 for a, b in zip(band8, band4)])
    return results

def addEVI(band2, band4, band8):
    G = 2.5
    C1 = 6
    C2 = 7.5
    L = 1
    
    results = pd.Series([
        G * (nir - red) / (nir + C1 * red - C2 * blue + L) if (nir + C1 * red - C2 * blue + L) != 0 else 0
        for nir, red, blue in zip(band8, band4, band2)
    ])
    
    return results


def addNDWI(band3, band8):
    results = pd.Series([(green - nir) / (green + nir) if (green + nir) != 0 else 0 for green, nir in zip(band3, band8)])
    return results