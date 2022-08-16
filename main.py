# %% 0. Libraries
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.utils import scatter_plot

# %% 1. Settings
warnings.filterwarnings('ignore')
option_settings = {
    'display.max_rows': None,
    'display.max_columns': False,
    'display.float_format': '{:,.4f}'.format
}
[pd.set_option(setting, option) for setting, option in option_settings.items()]

IN_PATH = 'data/in/'
OUT_PATH = 'data/out/'

# %% 2. Load data
dtype_dict = {'SKU': 'str'}
filename = 'DataTransaccion.csv'
data_ = pd.read_csv(IN_PATH + filename, sep=';', encoding='latin-1', dtype=dtype_dict)

# %% 3. Format data
data = data_.copy()
data.columns = map(str.lower, data)
data['diacompra_'] = pd.to_datetime(data['diacompra'], format='%m/%d/%Y', errors='coerce')
print(f'Date errors represent: {data[data.diacompra_.isnull()].shape[0] / data.shape[0] * 100} % of total records.') 
# Representan data de un dia, a nivel de serie de tiempo no se considerará pero para la segmentación sí

districts = [
    'San Miguel',
    'San Borja', 
    'San Isidro',
    'San Martin de Porres',
    'Miraflores',
    'Surquillo',
    'Surco',
    'Breña',
    'Callao',
    'Santa Anita'
]
data.loc[data['localcompra'].isin(districts), 'city'] = 'Lima'
data.loc[~data['localcompra'].isin(districts), 'city'] = data.loc[~data['localcompra'].isin(districts), 'localcompra']

data['customer_id_'] = data['customer_id'] + '_' + data['genero'] + '_' + data['nse']

data.groupby(['jerarquiacompra'])['jerarquiacompra2'].nunique().sort_values(ascending=False)

data = pd.get_dummies(data, columns=['nse', 'genero'])
data_users = pd.pivot_table(
    data=data, 
    index='customer_id_',
    values=['monto_venta', 'cuotas', 'edad', 'nse_A', 'nse_B', 'nse_C', 'nse_D', 'genero_F', 'genero_M'],
    aggfunc={
        'monto_venta': np.mean,
        'cuotas': np.mean,
        'edad': np.max,
        'nse_A': np.max,
        'nse_B': np.max,
        'nse_C': np.max,
        'nse_D': np.max,
        'genero_F': np.max,
        'genero_M': np.max
    }
).reset_index()

## EDA
# No hay nulos
# 17 jerarquias de SKU
# 136 jerarquias2 de SKU
# 308,947 usuarios
# 140,939 SKUs


# 1. Se identifico que un customer_id tiene doble genero: Puede que tenga el mismo código pero lo hacen dos personas distintas (tal vez convivientes)
# 2. Se generó un nuevo customer_id => concatenando el customer_id previo con el género (+1k usuarios)
# 3. Luego, se identificó que, de estos usuarios, 65 tienen doble NSE: 
# 80 customer_id tienen doble genero (son dos personas distintas)
# 115 customer_id tienen doble NSE

# %% Model
# Al utilizar métodos basados en distancias, es importante ver la distribución de las variables continuas
isf = IsolationForest(n_estimators=100, random_state=123, contamination=0.02)
preds = isf.fit_predict(data_users[['edad', 'monto_venta']])
data_users['iso_forest_outliers'] = preds 
data_users['iso_forest_outliers'] = data_users['iso_forest_outliers'].astype(str)
data_users['iso_forest_scores'] = isf.decision_function(data_users[['edad', 'monto_venta']])
data_users['iso_forest_outliers'].value_counts()

# %%
lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(data_users[['edad', 'monto_venta']])
data_users['lof_outliers'] = y_pred.astype(str)
data_users['lof_scores'] = lof.negative_outlier_factor_
data_users['lof_outliers'].value_counts()

# %%
data_users['outliers_sum'] = (data_users['iso_forest_outliers'].astype(int) + data_users['lof_outliers'].astype(int))



# %%
