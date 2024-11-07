#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:13:49 2024

@author: kevinchristian
"""

def addNDVI(band4, band8):
    return (band8-band4) / (band8+band4)

def addEVI(band2, band4, band8):
    return 2.5 * (band8-band4) / (band8 + (6 * band4) - (7.5 * band2) + 1)