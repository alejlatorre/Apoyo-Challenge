import scipy
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from .utils import normalize

class CFRecommender:   
    def __init__(self, dataframe, user_col, item_col, ranking_metric):
        self.dataframe = dataframe
        self.user_col = user_col
        self.item_col = item_col 
        self.ranking_metric = ranking_metric

    def get_cf_predictions(self, number_of_factors_mf=15):
        users_items_df = self.dataframe.groupby([self.user_col, self.item_col])[self.ranking_metric].sum().unstack().fillna(0)
        items_list = self.dataframe[[self.item_col]].copy()
        users_list = list(users_items_df.index)
        users_items_matrix = users_items_df.values
        users_items_sparse_matrix = csr_matrix(users_items_matrix)
        
        U, sigma, Vt = linalg.svds(users_items_sparse_matrix, k=number_of_factors_mf)
        sigma = np.diag(sigma)

        user_predicted_values = np.dot(np.dot(U, sigma), Vt)
        user_predicted_values_norm = normalize(user_predicted_values)
        
        return pd.DataFrame(user_predicted_values_norm, columns=users_items_df.columns, index=users_list).transpose()

    def get_recommended_items(self, cf_predictions_df, user_id, items_to_ignore=[], attributes_from_index=[], topn=10):
        sorted_user_predictions = cf_predictions_df[user_id].sort_values(ascending=False).reset_index()
        sorted_user_predictions.rename(columns={user_id: 'rec_strength'}, inplace=True)

        recommendations_df = sorted_user_predictions[~sorted_user_predictions[self.item_col].isin(items_to_ignore)]
        recommendations_df.sort_values('rec_strength', ascending = False, inplace=True)

        return recommendations_df.head(topn)