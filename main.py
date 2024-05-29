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

    resultado = f'El usuario {user_id} ha gastado {dinero}, tiene un porcentaje de recomendacion del {porcentaje}% y tiene {items} items'
    
    return resultado