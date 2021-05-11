import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def uar(userId, train_pt_df):
    l = [x for x in list(train_pt_df.loc[userId]) if x != 0]
    if len(l)==0:
        return 2.5
    else:
        return sum(l)/len(l)


def iar(itemId, train_pt_df):
    l = [x for x in list(train_pt_df[itemId]) if x != 0]
    if len(l)==0:
        return 2.5
    else:
        return sum(l)/len(l)


def func(train_pt_df):
    train_pt_df = train_pt_df.T
    
    user_average_rating = {userId:uar(userId, train_pt_df) for userId in train_pt_df.index}
    item_average_rating = {itemId:iar(itemId, train_pt_df) for itemId in train_pt_df.columns}
    
    git = {}
    for item in (train_pt_df.columns):
        l = train_pt_df[item].tolist()
        M = 3
        rp = [i for i in l if i >= M and i != 0]
        rq = [i for i in l if i < M and i != 0]
        left = 0
        right = 0
        for i in rp:
            left += i-M
        for i in rq:
            right += i-M
        git[item] = left - right
    
    gitd = {}
    for item in (train_pt_df.columns):
        if git[item] > 0:
            l = train_pt_df[item].tolist()
            M = 3
            rp = [i for i in l if i >= M and i != 0]
            if len(rp) == 0:
                gitd[item] = 0
            else:
                gitd[item] = sum(rp)/len(rp)
        elif git[item] == 0:
            gitd[item] = 3
        else:
            l = train_pt_df[item].tolist()
            M = 3
            rq = [i for i in l if i < M and i != 0]
            if len(rq) == 0:
                gitd[item] = 0
            else:
                gitd[item] = sum(rq)/len(rq)
        
    new_train_pt_df = train_pt_df
    
    for item in tqdm(train_pt_df.columns):
        for user in train_pt_df.index:
            if train_pt_df[item][user] == 0:
    #             r = [x for x in list(train_pt_df[item]) if x != 0]
    #             ri_bar = item_average_rating[item]
    #             r = [i-ri_bar for i in r]
    #             ru = user_average_rating[user] + (sum(r)/len(r))

    #             u = [x for x in list(train_pt_df.iloc[user-1]) if x != 0]
    #             ui_bar = user_average_rating[user]
    #             u = [i-ui_bar for i in u]
    #             ri = item_average_rating[item] + (sum(u)/len(u))
                one = user_average_rating[user]
                two = item_average_rating[item]
                three = gitd[item]
                new_train_pt_df[item][user] = three #math.sqrt(three)
    return new_train_pt_df.T

def predict_rating_IB(userId, itemId, train_pt_df, sm_df, K):
    try:
        # users_ratings = train_pt_df[userId]  # ratings of all items for user 'userId'
        items_similarities = sm_df[itemId]  # cosine similarities of item 'itemId' with all other items.
    except:
        return []
    
    # Consider only highest K item similarities. (Here, similarity of the same item will also not considered).
    d = dict(items_similarities) # {itemId : item_similarity}
    d_sorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    drop_rest_users = list(d_sorted.keys())
    drop_rest_users = drop_rest_users[K:]
    # users_ratings = users_ratings.drop(drop_rest_users)
    items_similarities = items_similarities.drop(drop_rest_users)
    
    # Not consider items whose ratings are NA.
    # drop_empty_ratings = users_ratings[users_ratings == 0].index
    # users_ratings = users_ratings.drop(drop_empty_ratings)
    # items_similarities = items_similarities.drop(drop_empty_ratings)
    
    return list(dict(items_similarities).keys())

# This function takes the training & testing dataframes and return the MAE.
def IB_MAE(userId, itemId, train_df, K):
    # pivot the training dataframe and Transpose it.
    train_pt_df = pd.pivot_table(train_df, values='Rating', index='userId', columns='itemId', aggfunc=np.sum).T
    # Replace the NA values with 0   (Note: I observed -> No user in the whole dataset have rated 0 to any movie)
    train_pt_df = train_pt_df.fillna(0)
    
    # train_pt_df = func(train_pt_df)
    
    # Calculate the cosine similarities of item-item.
    sm_df = pd.DataFrame(cosine_similarity(train_pt_df), index=train_pt_df.index, columns=train_pt_df.index)
    # Not consider the similarity of same item while predicting the rating. eg: similarity of item 1 with item 1.
    np.fill_diagonal(sm_df.values, 0)
    
    return predict_rating_IB(userId, itemId, train_pt_df, sm_df, K)

