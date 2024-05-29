"""
Estudiante : David Steven Villamizar López
Academia : Henry
Carrera : Data Science 
Direccion Web : https://www.soyhenry.com/
email : davidvilla042211@gmail.com
Año 2024
"""
# Importamos las variables que vamos a usar
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Leemos los archivos que vamos a usar para las consultas
df_games = pd.read_parquet(r'Datasets/games.parquet', engine='auto')
df_reviews = pd.read_parquet(r'Datasets/reviews.parquet', engine='auto')
df_items = pd.read_parquet(r'Datasets/user_items.parquet', engine='auto')
piv_table_norm = pd.read_parquet(r"Datasets/piv_table_norm.parquet", engine='auto')
df_user_simil =pd.read_parquet(r"Datasets/df_user_simil.parquet", engine='auto')
cosine_sim_df = pd.read_parquet(r"Datasets/cosine_sim_df.parquet", engine='auto')


app = FastAPI( 
    title = 'Machine Learning Operations (MLOps)',
    description='API para realizar consultas',
)

@app.get("/")
async def index():
    return {"Hola! Bienvenido a la API de consulta y recomendación. Por favor dirígete a /docs"}

@app.get("/about/")
async def about():
    return {"PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)"}

 
@app.get("/developer/{desarrollador}")
async def developer(desarrollador : str):
    '''
    Devuelve un diccionario con tres diccionarios en su interior que contienen: 
    Cantidad de items publicados por año, cantidad de items gratis y porceentaje de contenido gratis por año
             
    Parametro
    ---------  
            desarrollador : Nombre de la empresa desarrolladora
    
    Retorna
    -------
            - Cantidad de items por año
            - Cantidad gratis pos año
            - Porcentaje gratis por año
    '''
    
    mask1 = df_games["developer"] == desarrollador
    mask2 = df_games["price"] == 0.0
    developer = df_games[mask1]
    
    gratis = developer[mask2]
    cantidad_items= developer.groupby("release_year")["id"].count().to_dict()
    cantidad_gratis = gratis.groupby("release_year")["id"].count().to_dict()
    porcentaje_gratis = {year: f"{(cantidad_gratis.get(year, 0) / cantidad_items.get(year, 1)) * 100:.1f}%" for year in cantidad_items}
    total = {
            "cantidad de items por año":cantidad_items,
             "Cantidad gratis pos año":cantidad_gratis,
             "porcentaje gratis por año":porcentaje_gratis
            }
    
    return total

@app.get("/userdata/{user_id}")
async def userdata( user_id : str ):
   
    usermask = df_items['user_id'] == user_id
    usermask2 = df_reviews['user_id'] == user_id
    df_user_items = df_items[usermask]
    df_user_reco = df_reviews[usermask2]
    dinero = df_user_items['price'].sum()
    items = df_user_items['item_id'].count()
    recomendaciones = df_user_reco['recommend'].count()

    if recomendaciones < 1:
        porcentaje = 0
    else:
        if items < 1:
            porcentaje = 0
        else:    
            porcentaje = (recomendaciones*100)//items

    resultado = print(f'El usuario {user_id} ha gastado ${dinero}, tiene un porcentaje de recomendacion del {porcentaje}% y tiene {items} items')
    
    return resultado

@app.get("/UserForGenre/{genre}")
async def UserForGenre(genre : str):
    
    generos = df_games[['item_id','genres']]
    generos['genres'] = generos['genres'].fillna('Unassigned gender')
    df_items2 = pd.merge(df_items, generos, how='left', on=['item_id'], indicator=False)
    mask1 = df_items2['genres'] == genre
    df = df_items2[mask1]
    df = df[['user_id','playtime_forever']]
    id = df.groupby('user_id')['playtime_forever'].sum().idxmax()
    horas = df.groupby('user_id')['playtime_forever'].sum().max()

    return print(f'El usuario con mas horas jugadas para el genero {genre} es {id} con {horas} horas jugadas')

@app.get("/best_developer_year/{año}")
async def best_developer_year (año):
    
    top = df_reviews[['Year','recommend','developer']]
    mask1 = top['Year'] == año
    top = top[mask1]
    top.drop(columns='Year', inplace=True)
    agrupados = top.groupby('developer')['recommend'].sum().sort_values(ascending=False).head(3).to_dict()
    
    return agrupados

@app.get("/develreviewsanalys/({desarrolladorsent})")
def devel_reviews_analys(desarrollador: str):

    mask1 = df_reviews['developer'] == desarrollador
    df = df_reviews[mask1]
    comentario = df['Sentiment_analysis'].value_counts().to_dict()
    comentario['Positivo'] = comentario.pop(2)
    comentario['Neutro'] = comentario.pop(1)
    comentario['Negativo'] = comentario.pop(0)
    diccionario = {desarrollador: comentario}
    
    return diccionario


# ML: RECOMENDACIÓN USER-ITEM:

@app.get("/similaruserrecs/({user})")
def similar_user_recs(user):

    '''Los 5 juegos más recomendados similares recomendados por usuario...'''
    # Se verifica si el usuario está presente en las columnas de piv_table_norm
    if user not in df_user_simil.columns:
        return {'message': 'El Usuario no tiene datos disponibles {}'.format(user)}

    # Se obtienen los usuarios más similares 
    sim_users = df_user_simil.sort_values(by=user, ascending=False).index[1:11]

    best = []  
    most_common = {}  

    # Por cada usuario similar, encuentra el juego mejor calificado y lo agrega a la lista 'best'
    for i in sim_users:
        max_score = piv_table_norm.loc[:, i].max()
        best.append(piv_table_norm[piv_table_norm.loc[:, i] == max_score].index.tolist())

    # Se cuenta cuántas veces se recomienda cada juego
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1

    # Se ordenan los juegos de mayor recomendación
    sorted_list = sorted(most_common.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_list[:5])



# RECOMENDACIÓN ITEM-ITEM:

@app.get("/getsimilaritems/({item_id})")
def get_similar_items(item_id, top_n=5):
    ''' La función para obtener el top N=5 de items similares al introducido por id de juego'''

    similar_items = cosine_sim_df[item_id].sort_values(ascending=False).head(top_n + 1).iloc[1:]
    return similar_items