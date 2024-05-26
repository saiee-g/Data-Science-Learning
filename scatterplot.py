# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:51:41 2024

@author: snowfox
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x= np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])
print(x,y)

model=np.polyfit(x, y, 1)
print(model)

predict=np.poly1d(model)
print(predict(65))

y_pred=predict(x)
print(y_pred)

r2_score(y,y_pred)
print(r2_score(y,y_pred))

y_line=model[1]+model[0]*x
plt.plot(x, y_line, c='r')
plt.scatter(x,y_pred)
plt.scatter(x,y,c='r')
