# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:14:44 2016

@author: s1321168
"""

import Levenshtein

string1 = "30分でつくれるAndroidアプリ"
string2 = "3C分でn <fるAnd r0S7ブu N0>UめAd!"

# string1 = string1.decode('utf-8')
# string2 = string2.decode('utf-8')

print(Levenshtein.distance(string1, string2))