import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
#%%
# 데이터 로드
from statsmodels.datasets import co2
data = co2.load_pandas()
df = data.data
#%%
# 인덱스를 날짜 형식으로 변환
df.index = pd.date_range(start='1958-03-29', end='2001-03-29', freq='W-SAT')

# STL 분해
result = seasonal_decompose(df, model='additive')

# 시각화
plt.rcParams.update({'figure.figsize': (10,10)})
fig, axes = plt.subplots(4, 1, sharex=True)
result.observed.plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Observed')
result.trend.plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Trend')
result.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Seasonal')
result.resid.plot(ax=axes[3], legend=False)
axes[3].set_ylabel('Residual')
plt.show()

# %%
