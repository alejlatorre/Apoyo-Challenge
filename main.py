# %% 0. Libraries
import dtale
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.models import CFRecommender

# %% 1. Settings
warnings.filterwarnings('ignore')
option_settings = {
    'display.max_rows': False,
    'display.max_columns': None,
    'display.float_format': '{:,.4f}'.format
}
[pd.set_option(setting, option) for setting, option in option_settings.items()]

%matplotlib inline
plt.rcParams["figure.autolayout"] = True
sns.set_theme(style='darkgrid')

IN_PATH = 'data/in/'
OUT_PATH = 'data/out/'

# %% 2. Load data
dtype_dict = {'SKU': 'str'}
filename = 'DataTransaccion.csv'
data_ = pd.read_csv(IN_PATH + filename, sep=';', encoding='latin-1', dtype=dtype_dict)

# %% 3. EDA
d = dtale.show(data_)
dtale.instances()

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

# %% 4. Format data
data = data_.copy()
data.columns = map(str.lower, data)
data.drop(index=data[data['edad'] == data['edad'].max()].index, inplace=True)
data['diacompra_'] = pd.to_datetime(data['diacompra'], format='%m/%d/%Y', errors='coerce')
data['codmes'] = data['diacompra_'].dt.strftime('%Y%m')
print(f'Date errors represent: {data[data.diacompra_.isnull()].shape[0] / data.shape[0] * 100} % of total records.') 

# Creacion de variable ciudad
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
data['quantity'] = 1

# Dividir por NSE
data_a = data[data['nse']=='A'].copy()
data_b = data[data['nse']=='B'].copy()
data_c = data[data['nse']=='C'].copy()
data_d = data[data['nse']=='D'].copy()

# Top 100 por c/NSE para disminuir dimensionalidad
a_cust_id_top_100 = data_a.groupby(['customer_id_'])['monto_venta'].sum().sort_values(ascending=False)[:100].index
b_cust_id_top_100 = data_b.groupby(['customer_id_'])['monto_venta'].sum().sort_values(ascending=False)[:100].index
c_cust_id_top_100 = data_c.groupby(['customer_id_'])['monto_venta'].sum().sort_values(ascending=False)[:100].index
d_cust_id_top_100 = data_d.groupby(['customer_id_'])['monto_venta'].sum().sort_values(ascending=False)[:100].index

# Filtro sobre base
subset_a = data_a[data_a['customer_id_'].isin(a_cust_id_top_100)].copy()
subset_b = data_b[data_b['customer_id_'].isin(b_cust_id_top_100)].copy()
subset_c = data_c[data_c['customer_id_'].isin(c_cust_id_top_100)].copy()
subset_d = data_d[data_d['customer_id_'].isin(d_cust_id_top_100)].copy()

# Metricas por usuario por grupo de NSE
data_users = pd.pivot_table(
    data=data,
    index=['customer_id_', 'nse'],
    values=['monto_venta', 'cuotas', 'edad'],
    aggfunc={
        'monto_venta': np.sum,
        'cuotas': np.mean,
        'edad': np.max
    }
).reset_index()
data_users_a = pd.pivot_table(
    data=data_a,
    index='customer_id_',
    values=['monto_venta', 'cuotas', 'edad'],
    aggfunc={
        'monto_venta': np.sum,
        'cuotas': np.mean,
        'edad': np.max
    }
).reset_index()
data_users_b = pd.pivot_table(
    data=data_b,
    index='customer_id_',
    values=['monto_venta', 'cuotas', 'edad'],
    aggfunc={
        'monto_venta': np.sum,
        'cuotas': np.mean,
        'edad': np.max
    }
).reset_index()
data_users_c = pd.pivot_table(
    data=data_c,
    index='customer_id_',
    values=['monto_venta', 'cuotas', 'edad'],
    aggfunc={
        'monto_venta': np.sum,
        'cuotas': np.mean,
        'edad': np.max
    }
).reset_index()
data_users_d = pd.pivot_table(
    data=data_d,
    index='customer_id_',
    values=['monto_venta', 'cuotas', 'edad'],
    aggfunc={
        'monto_venta': np.sum,
        'cuotas': np.mean,
        'edad': np.max
    }
).reset_index()

# %% Plots
# Barplot: Sales in Lima
pd.pivot_table(
    data=data[(data['city'] == 'Lima')],
    index='codmes',
    columns='city',
    values='monto_venta',
    aggfunc=np.sum
).plot(kind='bar', figsize=(10, 5))
plt.title('Venta por mes en Lima', fontsize=15)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Venta (S/)', fontsize=12)
plt.legend(loc='upper left')
plt.show()

# Lineplot: Sales per city (w/o Lima)
pd.pivot_table(
    data=data[(data['city'] != 'Lima')],
    index='codmes',
    columns='city',
    values='monto_venta',
    aggfunc=np.sum
).plot(kind='line', figsize=(12, 5))
plt.title('Venta por mes y provincia (excl. Lima)', fontsize=15)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Venta (S/)', fontsize=12)
plt.legend(loc='upper left')
plt.show()

# %% Plots 
# Boxplots: nse X monto venta
plt.figure(figsize=(10, 5))
sns.boxplot(data=data_users[data_users['monto_venta'] < 500], y='nse', x='monto_venta')
plt.xlabel('Monto de venta', fontsize=12)
plt.ylabel('NSE', fontsize=12)
plt.title('Distribución de montos de venta por NSE', fontsize=15)
plt.show()

# Boxplots: nse X edad
plt.figure(figsize=(10, 5))
sns.boxplot(data=data_users[data_users['monto_venta'] < 500], y='nse', x='edad')
plt.xlabel('Edad', fontsize=12)
plt.ylabel('NSE', fontsize=12)
plt.title('Distribución de edad por NSE', fontsize=15)
plt.show()

# %% Customer Segmentation
list_of_user_df = {
  'A': data_users_a,
  'B': data_users_b,
  'C': data_users_c,
  'D': data_users_d
}
results = {}

for nse, df in list_of_user_df.items():
  # Scaling
  scaler = MinMaxScaler()
  norm_cols = ['norm_cuotas', 'norm_edad', 'norm_monto_venta']
  df[norm_cols] = scaler.fit_transform(df[['cuotas', 'edad', 'monto_venta']])

  # Get Elbow Method plot
  WCSS = []
  for i in range(1,11):
      model = KMeans(n_clusters = i, init = 'k-means++')
      model.fit(df[norm_cols])
      WCSS.append(model.inertia_)
  fig = plt.figure(figsize = (6, 3))
  plt.plot(range(1,11), WCSS, linewidth=4, markersize=7, marker='o', color = 'green')
  plt.xticks(np.arange(11))
  plt.xlabel("Number of clusters")
  plt.ylabel("WCSS")
  plt.title(f'Elbow Method - {nse}')
  plt.show()

  results[nse] = pd.DataFrame(list(zip(range(1, 11), WCSS)), columns=['k', 'inertia'])
  results[nse]['pct_change'] = results[nse]['inertia'].pct_change()
  results[nse]['improvement'] = results[nse]['pct_change'] - results[nse]['pct_change'].shift(1) 

model_a = KMeans(n_clusters=3, init='k-means++').fit(data_users_a[norm_cols])
model_b = KMeans(n_clusters=3, init='k-means++').fit(data_users_b[norm_cols])
model_c = KMeans(n_clusters=3, init='k-means++').fit(data_users_c[norm_cols])
model_d = KMeans(n_clusters=3, init='k-means++').fit(data_users_d[norm_cols])

data_users_a['cluster'] = model_a.predict(data_users_a[norm_cols])
data_users_b['cluster'] = model_b.predict(data_users_b[norm_cols])
data_users_c['cluster'] = model_c.predict(data_users_c[norm_cols])
data_users_d['cluster'] = model_d.predict(data_users_d[norm_cols])

# %% KMeans plots
# NSE A
colors = ['red', 'green', 'blue']
assign = []
for row in data_users_a['cluster'].values:
    assign.append(colors[row])

f1 = data_users_a['edad'].values
f2 = data_users_a['cuotas'].values

plt.figure(figsize=(7, 4))
plt.scatter(f1, f2, c=assign, s=60)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Cuotas', fontsize=12)
plt.title(f'Clusters de usuarios - NSE A', fontsize=15)
plt.show()

# NSE B
colors = ['red', 'green', 'blue']
assign = []
for row in data_users_b['cluster'].values:
    assign.append(colors[row])

f1 = data_users_b['edad'].values
f2 = data_users_b['cuotas'].values

plt.figure(figsize=(7, 4))
plt.scatter(f1, f2, c=assign, s=60)
plt.xlabel('Edad')
plt.ylabel('Cuotas')
plt.title(f'Clusters de usuarios - NSE B')
plt.show()

# NSE C
colors = ['red', 'green', 'blue']
assign = []
for row in data_users_c['cluster'].values:
    assign.append(colors[row])

f1 = data_users_c['edad'].values
f2 = data_users_c['cuotas'].values

plt.figure(figsize=(7, 4))
plt.scatter(f1, f2, c=assign, s=60)
plt.xlabel('Edad')
plt.ylabel('Cuotas')
plt.title(f'Clusters de usuarios - NSE C')
plt.show()

# NSE D
colors = ['red', 'green', 'blue']
assign = []
for row in data_users_d['cluster'].values:
    assign.append(colors[row])

f1 = data_users_d['edad'].values
f2 = data_users_d['cuotas'].values

plt.figure(figsize=(7, 4))
plt.scatter(f1, f2, c=assign, s=60)
plt.xlabel('Edad')
plt.ylabel('Cuotas')
plt.title(f'Clusters de usuarios - NSE D')
plt.show()

# %% Recommendation System
CFR_a = CFRecommender(dataframe=subset_a, user_col='customer_id_', item_col='sku', ranking_metric='quantity')
CFR_b = CFRecommender(dataframe=subset_b, user_col='customer_id_', item_col='sku', ranking_metric='quantity')
CFR_c = CFRecommender(dataframe=subset_c, user_col='customer_id_', item_col='sku', ranking_metric='quantity')
CFR_d = CFRecommender(dataframe=subset_d, user_col='customer_id_', item_col='sku', ranking_metric='quantity')

cf_preds_a = CFR_a.get_cf_predictions(number_of_factors_mf=15)
cf_preds_b = CFR_b.get_cf_predictions(number_of_factors_mf=15)
cf_preds_c = CFR_c.get_cf_predictions(number_of_factors_mf=15)
cf_preds_d = CFR_d.get_cf_predictions(number_of_factors_mf=15)

# Top 10 customers
top_10_customers = data.groupby(['customer_id_'])['monto_venta'].sum().sort_values(ascending=False)[:10].index

for idx, customer_id in enumerate(top_10_customers):
    nse = customer_id[-1] 
    if nse == 'A':
        reco = CFR_a.get_recommended_items(cf_predictions_df=cf_preds_a, user_id=customer_id)
    elif nse == 'B':
        reco = CFR_b.get_recommended_items(cf_predictions_df=cf_preds_b, user_id=customer_id)
    elif nse == 'C':
        reco = CFR_c.get_recommended_items(cf_predictions_df=cf_preds_c, user_id=customer_id)
    elif nse == 'D':
        reco = CFR_d.get_recommended_items(cf_predictions_df=cf_preds_d, user_id=customer_id)
    else:
        print('NSE not found.')

    print(f'Customer ranking: {idx+1}')
    print(f'Customer ID: {customer_id[0:-4]}')
    print('Recommended SKUs:')
    print(reco)
