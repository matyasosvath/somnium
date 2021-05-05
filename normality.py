import numpy as np
import pandas as pd

# Distribution
def distribution(col):
    skewness = col.skew()
    kurtosis = col.kurtosis()
    return skewness, kurtosis
