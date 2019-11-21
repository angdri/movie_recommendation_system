import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pickle
import re
import dash_bootstrap_components as dbc
from nltk.stem.snowball import SnowballStemmer

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


log_model = pickle.load(open('legendary_predictor.sav', 'rb'))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#function for make set of unique values of a columns
def make_set(df,col):
    set_keywords = set()
    for keyword in df[col].str.split(',').values:
        if isinstance(keyword, float): continue
        set_keywords = set_keywords.union(keyword)
    return set_keywords

#function for make a dictionary and list of tuples from a columns, sort of value_counts()
def count_word(df, col):
    list_word = make_set(df,col)
    #make dict for count words
    keyword_count = dict()
    
    #fill dict key with words set default value to 0
    for i in list_word: keyword_count[i] = 0
    
    #loop all data from column, split all the word, set it as a list
    for list_key in df[col].str.split(','):  
        #ignore NaN
        if type(list_key) == float and pd.isnull(list_key): continue  
        #loop all tha data, remove word that's not in our list_word
        for key in [s for s in list_key if s in list_word]:
            #add 1 count for every word that found
            keyword_count[key] += 1
    #______________________________________________________________________
    
    # convert sort the dictionary return as list of tupples
    keyword_occurences = sorted(keyword_count.items(), key=lambda x:x[1], reverse=True)

    #return sorted list and unsorted dict
    return keyword_occurences, keyword_count

def sum_col_by_gb_col(df, col, gb_col):
    list_word = make_set(df,gb_col)
    #make dict for count category
    keyword_count = dict()
    keyword_occurences = []
    #fill dict key with words set default value to 0
    for i in list_word: keyword_count[i] = [0,0,99999999999999999999,0]
    
    #loop all data from col and gb_col
    for col_val, gb_col_val in zip(df[col],df[gb_col]): 
        #loop all the category in one row
        #ignore NaN
        if type(gb_col_val) == float and pd.isnull(gb_col_val): continue
        for gb_key in gb_col_val.split(','):
            #add col values to gb_col category
            if gb_key in list_word:
                if col_val > keyword_count[gb_key][1]:
                    max_val = col_val
                else:
                    max_val = keyword_count[gb_key][1]
                if col_val < keyword_count[gb_key][2]:
                    min_val = col_val
                else:
                    min_val = keyword_count[gb_key][2]
                sum_val = keyword_count[gb_key][0] + col_val
                count_val = keyword_count[gb_key][3] + 1
                keyword_count[gb_key] = [sum_val, max_val, min_val,count_val]
                

    for key,val in zip(keyword_count.keys(),keyword_count.values()):
        keyword_occurences.append([key, val[0], val[1], val[2], val[3]])

    #return dataframe count, sum, max, min, count
    return pd.DataFrame(keyword_occurences, columns=[gb_col, 'sum'+col, 'max'+col, 'min'+col, 'count'])

def generate_table(dataframe, page_size = 10):
     return dash_table.DataTable(
                    # id = 'dataTable',
                    columns = [{"name": i.capitalize(), "id": i} for i in dataframe.columns],
                    data=dataframe.to_dict('records'),
                    page_action="native",
                    page_current= 0,
                    page_size= page_size,
                    sort_action='native',
                    filter_action='native',
                    # fixed_columns={ 'headers': True, 'data': 2 },
                    style_table={
                        # all three widths are needed
                        # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'overflowX': 'scroll',
                    },
                    # style_cell_conditional=[
                    #     { 'if': {'column_id': 'soup'}, 'textAlign': 'left', 'height' : 'auto'},
                    #     { 'if': {'column_id': 'overview'}, 'textAlign': 'left', 'height' : 'auto'},
                    # ]
                )

def create_cosine_sim(df, col, vector, model):
    # cosine_sim = 0
    if vector in ['tfidf']:
        #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(df[col])

        if model in ['cosine_similarity']:
            cosine_sim = cosine_similarity( tfidf_matrix,  tfidf_matrix)
        elif model in ['linear_kernel']:
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    elif vector in ['count']:
        countVector = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

        #Construct the required TF-IDF matrix by fitting and transforming the data
        count_matrix = countVector.fit_transform(df[col])

        if model in ['cosine_similarity']:
            cosine_sim = cosine_similarity(count_matrix, count_matrix)
        elif model in ['linear_kernel']:
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(count_matrix, count_matrix)
    
    return cosine_sim

def get_recommendations(df, title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie (pair of the index and the similarity scores)
    # by taking 1 row that represent the movie in cosine_sim matrix
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the similarity scores, so we can get the most similar movie with the title that was given
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies, exclude index 0 because it is the title that was given
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
movie_df = pd.read_csv('./data/clean_movie_plus_credit.csv', encoding='raw_unicode_escape')

#Fill all NaN with empty string
movie_copy = movie_df.fillna('')
movie_df['cast_name'] = movie_copy['cast_name'].apply(lambda x: re.sub(r'\'|\[|\]','',str(x.split(',')[:3])) if len(x.split(',')) > 3 else str(x))
list_genres = sorted(make_set(movie_df,'genres'))
list_directors = sorted(make_set(movie_df,'director'))
list_actors = sorted(make_set(movie_df,'cast_name'))
used_col = ['id', 'title', 'genres', 'runtime', 'production_companies','release_date', 'vote_average', 'cast_name', 'director', 'budget', 'revenue']
unused_col = ['keywords', 'original_language', 'overview', 'popularity', 'production_countries', 
              'spoken_languages', 'tagline', 'vote_count', 'production_countries_iso', 'spoken_languages_iso',
              'release_year', 'release_month', 'producer']


C= movie_df['vote_average'].mean()

#minimum vote required, only 10% of the highest vote_count 
m= movie_df['vote_count'].quantile(0.9)

#function for count weight rating
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

#get 10% of top vote_count movies
q_movies = movie_df.copy().loc[movie_df['vote_count'] >= m]

#insert weight rating into new columns ['score']
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#sort movies base on weigth rating
q_movies = q_movies.sort_values('score', ascending=False)

# movie_copy['director_soup'] = movie_copy['director'].apply(lambda x: x.replace(',',' ').lower())
# movie_copy = movie_df.fillna('')
movie_copy['director_soup'] = movie_copy['director'].apply(lambda x: x.replace(',',' ').lower())
movie_copy['cast_name_soup'] = movie_copy['cast_name'].apply(lambda x: re.sub(r' |\'|\[|\]','',x.lower()).replace(',',' '))
movie_copy['genres_soup'] = movie_copy['genres'].apply(lambda x: x.replace(',',' ').lower())
movie_copy['soup'] = movie_copy[['keywords','cast_name_soup','director_soup','genres_soup']].apply(lambda x: ''.join(x['keywords']) + ' '+ ''.join(x['cast_name_soup'])+ ' ' + ''.join(x['director_soup']) +' '+ ''.join(x['genres_soup']), axis=1)
movie_copy.drop(['director_soup','cast_name_soup','genres_soup'], axis=1, inplace=True)

#Construct a reverse map of indices and movie titles, so we can get the index base on movie's title
indices = pd.Series(movie_df.index, index=movie_df['title']).drop_duplicates()


# print(children)

navbar = dbc.NavbarSimple(
    children=[],
    brand="Movie Recommendation",
    brand_href="#",
    sticky="top",
)

dataframe = dbc.Card(
                dbc.CardBody([
                    html.Div([ html.Center(html.H1('Movies Dataframe')),
                        # dbc.Row([
                        #     dbc.Col(html.Div(children =[
                        #         html.P('Filter by:'),
                        #         dcc.Dropdown(value = 'genres', id='filter-by', style = {'max-width': '160px'},
                        #                     options = [{f'label': i, 'value': v} for i,v in zip(['Genres','Actors','Directors', 'Year'],['genres','cast_name','director', 'release_year'])]
                        #         )
                        #     ]), width=3),

                        #     dbc.Col(children = [
                        #         html.Div(children = [
                        #             html.P('Genre:', id='filter-choice-p'),
                        #             dcc.Dropdown(value = '', id='filter-choice', options = [{f'label': i, 'value': i} for i in list_genres], style = {'max-width': '160px'})
                        #         ], id='filter-choice-div')], width=3),
                            
                        #     dbc.Col(html.Div(children = [
                        #         html.P('Max Rows : '),
                        #         dcc.Input(
                        #             id='filter-row',
                        #             type='number',
                        #             value=10,)
                        #     ]), width=3),
                        #     dbc.Col([dbc.Button("Search", color="secondary", className="mr-1",id = 'filter')], width=3),
                        # ], align="center",),
                        html.Br(),
                        html.Div(id = 'div-table', children =[generate_table(movie_df[used_col])])
                    ])
                ]), className="mt-3",)

barchart = dbc.Card(
                dbc.CardBody([
                    html.Div(children=[
                        dbc.Row(children = [
                            dbc.Col(html.Div(children = [html.P('X-axis'), dcc.Dropdown(
                                    id = 'x-axis',
                                    options = [{'label' : i, 'value':i} for i in ['release_year', 'release_month','cast_name', 'director','producer','production_companies','genres']],
                                    value = 'release_year')
                                ]), width=3),
                            dbc.Col(children = [
                                    html.Div(
                                        children = [
                                            html.P('Y-axis'), dcc.Dropdown(
                                            id = 'y-axis',
                                            options = [{'label' : i, 'value':i} for i in ['Count','Max Revenue','Sum Revenue','Min Revenue','Max Budget','Sum Budget','Min Budget',]],
                                            value = 'Count')
                                        ], id='y-axis-div'),
                            ], width=3),
                        ]),

                        dbc.Row(children = [
                                dbc.Col([
                                    html.Div(children = [], id='bar-chart-div')
                                ], width=9),
                        ]),
                    ])
                ])
            )

recommendation = dbc.Card(
                    dbc.CardBody([
                        html.Div([ html.Center(html.H1('Movies Recommendation')),html.Br(),
                            dbc.Row([
                                dbc.Col(children = [
                                    html.Div(children = [
                                        html.P('Recommendation by:'),
                                        dcc.Dropdown(value = 'top', id='recommendation-by', options = [{f'label': k, 'value': v} for k,v in zip(['Top Movies', 'Overview', 'Keywords', 'Overview, Cast, Director'],['top', 'overview', 'keywords', 'all'])], style = {'max-width': '160px'})
                                    ], id='recommendation-by-div')], width=3),

                                dbc.Col(html.Div(children =[html.P('Movie Title:'),
                                                            dcc.Dropdown(value = '', id='movie-title',
                                                                        options = [{f'label': i, 'value': v} for i,v in zip(indices.index, indices.values)]
                                                            )], id='movie-title-div'), width=3,),

                                dbc.Col(html.Div(children =[html.P('Vector:'),
                                                            dcc.Dropdown(value = '', id='recommendation-vector',
                                                                    options = [{f'label': i, 'value': v} for i,v in zip(['TfidfVectorizer','CountVectorizer'],['tfidf','count'])]
                                                            )], id='vector-recommendation-div'), width=3,),

                                dbc.Col(html.Div(children =[html.P('Model:'),
                                                            dcc.Dropdown(value = '', id='recommendation-model',
                                                                    options = [{f'label': i, 'value': i} for i in ['cosine_similarity','linear_kernel']]
                                                            )], id='model-recommendation-div'), width=3,),
                                
                            ], align="center",),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([dbc.Button("Recommendation Result", color="secondary", className="mr-1",id = 'result_button')], width=6),
                            ]),
                            html.Br(),
                            html.Div(id = 'recommendation-table', children =[])
                        ])
                    ]), className="mt-3",)

tabs = dbc.Tabs(children = [
            dbc.Tab(children =[dataframe], label="DataFrame Table"),

            dbc.Tab(children=[barchart], label="Bar Chart"),

            dbc.Tab(children=[recommendation], label="Movie Recommendation"),
])

app.layout = html.Div(children = [ navbar, tabs,],
                    style ={
                    'maxWidth': '1200px',
                    'margin': '0 auto'
                    }
                )


# @app.callback(
#     Output(component_id = 'div-table', component_property = 'children'),
#     [Input(component_id = 'filter', component_property = 'n_clicks')],
#     [State(component_id = 'filter-choice', component_property = 'value'),
#     State(component_id = 'filter-by', component_property = 'value'),
#     State(component_id = 'filter-row', component_property = 'value')]
# )
# def update_table(n_clicks, choice, by, row):
#     if choice == '':
#         children = [generate_table(movie_df[used_col], page_size = row)]
#     else:
#         if by == 'release_year':
#             children = [generate_table(movie_df[movie_df[by] == choice][used_col], page_size = row)]
#         else:
#             children = [generate_table(movie_df[movie_df[by].str.contains(choice)][used_col], page_size = row)]            
#     return children


# @app.callback(
#     Output(component_id = 'filter-choice-div', component_property = 'children'),
#     [Input(component_id = 'filter-by', component_property = 'value')],
# )

# def update_filter_choice(filter_by):
#     if filter_by == 'genres' :
#         children = [html.P('Genre:', id='filter-choice-p'),
#                     dcc.Dropdown(value = '', id='filter-choice', options = [{f'label': i, 'value': i} for i in list_genres])]
#     elif filter_by == 'director':
#         children = [html.P('Directors:', id='filter-choice-p'),
#                     dcc.Dropdown(value = '', id='filter-choice', options = [{f'label': i, 'value': i} for i in list_directors])]
#     elif filter_by == 'actors':
#         children = [html.P('Actors :', id='filter-choice-p'),
#                      dcc.Dropdown(value = '', id='filter-choice',options = [{f'label': i, 'value': i} for i in list_actors])]
#     elif filter_by == 'release_year':
#         children = [html.P('Release Year :', id='filter-choice-p'),
#                     dcc.Dropdown(value = '', id='filter-choice', options = [{f'label': i, 'value': i} for i in movie_df['release_year'].value_counts().sort_index().index])]   
#     return children

# @app.callback(
#     Output(component_id = 'y-axis-div', component_property = 'children'),
#     [Input(component_id = 'x-axis', component_property = 'value')],)

# def update_y_axis(x_axis):
#     if x_axis in ['release_year'] :
#         children = [html.P('Y-axis'), dcc.Dropdown(
#                         id = 'y-axis',
#                         options = [{'label' : i, 'value':i} for i in ['Count','Max Revenue','Sum Revenue','Min Revenue','Max Budget','Sum Budget','Min Budget',]],
#                         value = 'Count')
#                     ]
#     elif x_axis in ['release_month'] :
#         children = [html.P('Y-axis'), dcc.Dropdown(
#                         id = 'y-axis',
#                         options = [{'label' : i, 'value':i} for i in ['Count','Max Revenue','Sum Revenue','Max Budget','Sum Budget',]],
#                         value = 'Count')
#                     ]
#     elif x_axis in ['cast_name','director','producer','production_companies', 'genres']:
#         children = [html.P('Y-axis'), dcc.Dropdown(
#                         id = 'y-axis',
#                         options = [{'label' : i, 'value':i} for i in ['Count','Max Revenue','Sum Revenue','Max Budget','Sum Budget',]],
#                         value = 'Count')
#                     ]
#     return children

@app.callback(
    Output(component_id = 'bar-chart-div', component_property = 'children'),
    # Output(component_id = 'bar-chart-div', component_property = 'style'),],
    [Input(component_id = 'x-axis', component_property = 'value'),
    Input(component_id = 'y-axis', component_property = 'value')]
)

def create_graph_bar(x_axis, y_axis):
    if x_axis in ['release_year','release_month']:
        if y_axis == 'Count':
            x_data = movie_df[x_axis].value_counts().sort_index(ascending=False).index
            y_data = movie_df[x_axis].value_counts().sort_index(ascending=False).values
        elif y_axis == 'Max Revenue':
            x_data = movie_df.groupby(x_axis)['revenue'].max().sort_values().index
            y_data = movie_df.groupby(x_axis)['revenue'].max().sort_values().values
        elif y_axis == 'Sum Revenue':
            x_data = movie_df.groupby(x_axis)['revenue'].sum().sort_values().index
            y_data = movie_df.groupby(x_axis)['revenue'].sum().sort_values().values
        elif y_axis == 'Min Revenue':
            x_data = movie_df.groupby(x_axis)['revenue'].min().sort_values().index
            y_data = movie_df.groupby(x_axis)['revenue'].min().sort_values().values
        elif y_axis == 'Max Budget':
            x_data = movie_df.groupby(x_axis)['budget'].max().sort_values().index
            y_data = movie_df.groupby(x_axis)['budget'].max().sort_values().values
        elif y_axis == 'Sum Budget':
            x_data = movie_df.groupby(x_axis)['budget'].sum().sort_values().index
            y_data = movie_df.groupby(x_axis)['budget'].sum().sort_values().values
        elif y_axis == 'Min Budget':
            x_data = movie_df.groupby(x_axis)['budget'].min().sort_values().index
            y_data = movie_df.groupby(x_axis)['budget'].min().sort_values().values

    elif x_axis in ['cast_name','director','producer','production_companies', 'genres']:
        if y_axis == 'Count':
            if x_axis == 'cast_name':
                keyword_occurences = pd.DataFrame(count_word(movie_df, x_axis)[0], columns=[x_axis,'count']).loc[1:21]
            else:
                keyword_occurences = pd.DataFrame(count_word(movie_df, x_axis)[0], columns=[x_axis,'count']).head(20)
            x_data = keyword_occurences[x_axis].values
            y_data = keyword_occurences['count'].values
        elif y_axis == 'Max Revenue':
            keyword_occurences = sum_col_by_gb_col(movie_copy, 'revenue', x_axis).sort_values(by='maxrevenue', ascending=False).head(20)
            x_data = keyword_occurences[x_axis].values
            y_data = keyword_occurences['maxrevenue'].values
        elif y_axis == 'Sum Revenue':
            keyword_occurences = sum_col_by_gb_col(movie_copy, 'revenue', x_axis).sort_values(by='sumrevenue', ascending=False).head(20)
            x_data = keyword_occurences[x_axis]
            y_data = keyword_occurences['sumrevenue'].values
        elif y_axis == 'Max Budget':
            keyword_occurences = sum_col_by_gb_col(movie_copy, 'budget', x_axis).sort_values(by='maxbudget', ascending=False).head(20)
            x_data = keyword_occurences[x_axis].values
            y_data = keyword_occurences['maxbudget'].values
        elif y_axis == 'Sum Budget':
            keyword_occurences = sum_col_by_gb_col(movie_copy, 'budget', x_axis).sort_values(by='sumbudget', ascending=False).head(20)
            x_data = keyword_occurences[x_axis].values
            y_data = keyword_occurences['sumbudget'].values
        

    children = [ dcc.Graph( id='bar-chart-graph',
                            figure = {
                                'data' : [
                                    {'x': x_data, 'y': y_data, 'type': 'bar', 'name' :'revenue'},
                                    # {'x': dfPokemonplot['Generation'], 'y': dfPokemonplot[x2], 'type': 'bar', 'name': x2}
                                ],
                                'layout': {'title': 'Bar Chart'}
                            }
                )]
    return children                


@app.callback(
    [Output(component_id = 'movie-title-div', component_property = 'children'),
    Output(component_id = 'vector-recommendation-div', component_property = 'children'),
    Output(component_id = 'model-recommendation-div', component_property = 'children'),],
    [Input(component_id = 'recommendation-by', component_property = 'value')],)

def get_movie_title(recommendation_by):
    if recommendation_by in ['top']:
        children_title = children_vector = children_model = ''
    elif recommendation_by in ['overview', 'keywords', 'all',]:
        children_title = [html.P('Movie Title :'),
                    dcc.Dropdown(value = '', id='movie-title',
                                options = [{f'label': i, 'value': v} for i,v in zip(indices.index, indices.values)]
                    )]
        children_vector = [html.P('Vector :'),
                    dcc.Dropdown(value = '', id='recommendation-vector',
                            options = [{f'label': i, 'value': v} for i,v in zip(['TfidfVectorizer','CountVectorizer'],['tfidf','count'])]
                    )]
        children_model = [html.P('Model :'),
                    dcc.Dropdown(value = '', id='recommendation-model',
                            options = [{f'label': i, 'value': i} for i in ['cosine_similarity','linear_kernel']]
                    )]
    return children_title, children_vector, children_model

# @app.callback(
#     Output(component_id = 'model-recommendation-div', component_property = 'style'),
#     [Input(component_id = 'movie_title', component_property = 'value'),]
#     #  Input(component_id = 'recommendation-by', component_property = 'value')],
#     )

# def get_model_recommendation(movie_title):
#     if movie_title in ['']:
#         children = ''
#     else:
#         children = [html.P('Model:'),
#                     dcc.Dropdown(value = '', id='recommendation-model',
#                             options = [{f'label': i, 'value': v} for i,v in zip(['TfidfVectorizer','CountVectorizer'],['tfidf','count'])]
#                     )]
#     return children

@app.callback(
    Output(component_id ='recommendation-table', component_property = 'children'),
    [Input(component_id ='result_button', component_property = 'n_clicks'),],
    [State(component_id ='recommendation-by', component_property = 'value'),
    State(component_id ='movie-title', component_property = 'value'),
    State(component_id ='recommendation-vector', component_property = 'value'),
    State(component_id ='recommendation-model', component_property = 'value'),]
)
def get_movie_recommendation(n_clicks, recommendation_by, movie_title, vector, model):
    # ['top', 'overview', 'keywords', 'all']
    if recommendation_by in ['top']:
        #get top 10 weigthed rating
        children = [generate_table(q_movies.head(10)[['id', 'title', 'genres', 'production_companies',
       'spoken_languages_iso', 'tagline',  'vote_average','vote_count', 'release_date', 'cast_name', 'producer', 'director',
       'score']])]
    elif recommendation_by in ['overview']:
        cosine_sim = create_cosine_sim(movie_copy, 'overview', vector, model)
        result = get_recommendations(movie_df, movie_title, cosine_sim).values
        children = [generate_table(movie_copy[movie_copy['title'].isin(result)].head(20)[['id', 'title', 'genres', 'overview']])]
    elif recommendation_by in ['keywords']:
        cosine_sim = create_cosine_sim(movie_copy, 'keywords', vector, model)
        result = get_recommendations(movie_df, movie_title, cosine_sim).values
        children = [generate_table(movie_copy[movie_copy['title'].isin(result)].head(20)[['id', 'title', 'genres', 'keywords']])]
    elif recommendation_by in ['all']:
        cosine_sim = create_cosine_sim(movie_copy, 'soup', vector, model)
        result = get_recommendations(movie_df, movie_title, cosine_sim).values
        children = [generate_table(movie_copy[movie_copy['title'].isin(result)].head(20)[['id', 'title', 'genres', 'soup']])]

    return children

if __name__ == '__main__':
    app.run_server(debug=True)
