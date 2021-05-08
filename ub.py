import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import math

def Intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))


def predict_rating_UB(userId, train_pt_df, sm_df, K):
    # Check if a completly new movie comes in my testing dataset.
    try:
        users_similarities = sm_df[userId]   # cosine similarities of user 'userId' with all other users.
        # users_ratings = train_pt_df[itemId]  # ratings of all users for item 'itemId'
    except:
        return []   # That movie doesn't exit in my training dataset. Ignore this case.
    
    # Consider only highest K item similarities. (Here, similarity of the same item will also not considered).
    d = dict(users_similarities) # {itemId : item_similarity}
    d_sorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    drop_rest_users = list(d_sorted.keys())
    drop_rest_users = drop_rest_users[K:]
    # users_ratings = users_ratings.drop(drop_rest_users)
    users_similarities = users_similarities.drop(drop_rest_users)
    
    # Not consider users who haven't rated the movie 'itemId'
    # drop_indices = users_ratings[users_ratings == 0].index
    # users_ratings = users_ratings.drop(drop_indices)
    # users_similarities = users_similarities.drop(drop_indices)


    l = set([])
    for sim_user in list(reversed(list(dict(users_similarities).keys()))):
        series_object = train_pt_df.loc[sim_user]
        l = set(list(series_object[series_object == 5].index)) | l

    return list(l)



# This function takes the training & testing dataframes and return the MAE.
def UB_MAE(userId, train_df, train_pt_df, K):
    # pivot the training dataframe.
    # train_pt_df = pd.pivot_table(train_df, values='Rating', index='userId', columns='itemId')
    # Replace the NA values with 0   (Note: I observed -> No user in the whole dataset have rated 0 to any movie)
    # train_pt_df = train_pt_df.fillna(0)
    # Calculate the cosine similarities of user-user.
    sm_df = pd.DataFrame(cosine_similarity(train_pt_df), index=train_pt_df.index, columns=train_pt_df.index)
    # Not consider the similarity of same user while predicting the rating. eg: similarity of user 1 with user 1.
    np.fill_diagonal(sm_df.values, 0)
    
    # d = {}
    # for user in train_df['userId'].unique().tolist():
    #     d[user] = train_df[train_df['userId']==user]['itemId'].tolist()
        
    # UserCount = {user:len(d[user]) for user in list(d.keys())}
    
    # threshold = {}
    # for user in list(d.keys()):
    #     other_users = list(set(list(d.keys())).difference([user]))
    #     l = []
    #     for other_u in other_users:
    #         l.append(len(Intersection(d[user], d[other_u])))
    #     threshold[user] = sum(l)/len(l)

    # for i in tqdm(sm_df.index):
    #     for j in sm_df.columns:
    #         y = len(Intersection(d[i], d[j]))
    #         if y >= threshold[i] and y >= threshold[j]:
    #             num = 2 * len(Intersection(d[i], d[j]))
    #             den = len(d[i]) + len(d[j])
    #             sm_df[i][j] = sm_df[i][j] * (num/den) 
    #         else:
    #             pi = abs(y - UserCount[i])
    #             pj = abs(y - UserCount[j])
    #             sm_df[i][j] = ((pi/UserCount[i]) * sm_df[i][j]) + ((pj/UserCount[j]) * sm_df[i][j])
    
    

    return predict_rating_UB(userId, train_pt_df, sm_df, K)