import numpy as np
import pandas as pd
from sklearn import datasets

boston = datasets.load_boston()
data = np.column_stack((boston.data, boston.target)) 
df = pd.DataFrame(data, columns=[f for f in boston.feature_names] + ['target'])

print(df.describe().T)
