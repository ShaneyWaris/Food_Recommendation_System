from item_based import *
from user_based import *
from flask import Flask, render_template, request, url_for, redirect, flash
import os, joblib
import numpy as np
import pandas as pd
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


df = joblib.load('./Dataset/df.pkl')
ingred_df = joblib.load('./Dataset/ingred_df.pkl')
recipe_imageURL = {}
userId = -1


@app.route('/most_popular_foods', methods=['POST', 'GET'])
def most_popular_foods():
    mpl = list(dict(df['recipe_name'].value_counts()).keys())[:50]
    d = {}
    for mp in mpl:
        d[mp] = df[df['recipe_name']==mp]['image_url'].iloc[0]
    return render_template('most_popular.html', available=True, recipe_imageURL=d)

@app.route('/recommended_items', methods=['POST', 'GET'])
def recommended_items():
    return render_template('index3.html', available=True, recipe_imageURL=recipe_imageURL)


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    available = False
    if request.method == 'POST':
        d = {key: 0 for key in range(0, 1398)}
        for i in range(0, 1398):
            d[i] = request.form.get(str(i))
        vector = []
        for i in list(d.values()):
            if i == None:
                vector.append(0)
            else:
                vector.append(5)

        any_favourite = False
        for i in vector:
            if i != 0:
                any_favourite = True
                break
        global userId
        if any_favourite == False:
            flash("Pleas SELECT atleast ONE FOOD from the below list, else we can't give you better recommendations!")
            return render_template('index2.html')

        global df
        pt_df = pd.pivot_table(df, values='Rating',index='userId', columns='itemId', aggfunc=np.sum)
        pt_df = pt_df.fillna(0)

        df2 = pd.DataFrame([vector], columns=list(pt_df.columns))

        pt_df = pt_df.append(df2, ignore_index=True)

        
        userId = list(pt_df.index)[-1]

        l = [[userId, itemId, 5, "", ""]
             for itemId, rating in d.items() if rating != None]

        df = df.append(pd.DataFrame(
            l, columns=list(df.columns)), ignore_index=True)

        items_list = UB_MAE(userId, df, pt_df, 1)  # 1816, not added in df, 


        global recipe_imageURL
        for itemId in items_list:
            recipe_name = df[df['itemId'] == itemId]['recipe_name'].iloc[0]
            image_url = df[df['itemId'] == itemId]['image_url'].iloc[0]

            recipe_imageURL[recipe_name] = image_url

        available = True

        return redirect(url_for('recommended_items'))

    return render_template('index2.html')


@app.route('/about/<recipe_name>', methods=['POST', 'GET'])
def about(recipe_name):
    menu_df = df.drop_duplicates(subset=['recipe_name']).reset_index(drop=True)
    recipeName_recipeId = {rn: ri for rn, ri in zip(menu_df['recipe_name'].tolist(), menu_df['itemId'].tolist())}
    itemId = recipeName_recipeId[recipe_name]
    similar_items = IB_MAE(userId, itemId, df, 15)

    global ingred_df
    ingred_list = ingred_df[ingred_df['recipe_name']==recipe_name]['ingredients'].tolist()[0].split("^")

    recipe_url = df[df['recipe_name']==recipe_name]['image_url'].iloc[0]
    return render_template('index4.html',
                           available=True,
                           recipe_imageURL={df[df['itemId'] == itemId]['recipe_name'].iloc[0]: df[df['itemId'] == itemId]['image_url'].iloc[0] for itemId in similar_items},
                           recipe_name=recipe_name,
                           recipe_url=recipe_url,
                           ingred_list=ingred_list
                           )


port = int(os.getenv('PORT', 8000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
