import pandas as pd
import numpy as np
import src
import copy
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from io import StringIO
import sys

class BurstDetector:

    def __init__(self):
        self._r = ro.r
        self._rbase = importr('base')
        self._pracma = importr('pracma')
        self._sjemea = importr('sjemea')
        self._e1071 = importr('e1071')
        
        #print(src.__file__)
        #c:\users\demogorgon\documents\college\marcus\boston university phd\ocker lab\correlations_and_bursts\src\__init__.py
        path = copy.deepcopy(src.__file__)
        total_length = len(path)
        trim_length = len("__init__.py")
        path = path[:total_length-trim_length]
        path.replace("\\", "/")
        #path = f"{path}rbursts\\R\\"
        self._r[f"{path}"]("logisi.R")
        #self._r[f"\'{path}\'"]('logisi.R')
        self._r_logisi = ro.globalenv['logisi.pasq.method']
        
        #self._rbursts = importr('rbursts', path)
        #self._logisi = ro.r['logisi.pasq.method']
        return
        
    #function converting output matrix to data frame
    def Matrix2DF(self, mat):   
        r_columns = [c for c in mat.colnames]
        columns = [c.replace('.', '_')  for c in r_columns]
        df = {}
        for i,c in enumerate(columns):
            column =  mat.rx(True, r_columns[i])
            df[c] = [x for x in column]
        return pd.DataFrame(df)
    
    def LogISI(self, sp_times, cutoff=0.1):
        #print('LogISI method')
        #%R -i sp_times
        #%R -i cutoff
        #%R bursts = logisi.pasq.method(sp_times, cutoff)
        #burst_frame = %Rget bursts
        #TODO: Most of the spike times are worked in matrices,
        # eventually this will need to run with sp_times as a matrix
        burst_frame = self._r_logisi(ro.FloatVector(sp_times), cutoff)
        return self.Matrix2DF(burst_frame)