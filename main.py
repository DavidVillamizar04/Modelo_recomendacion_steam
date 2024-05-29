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
#df_reviews = pd.read_parquet(r'Datasets/reviews.parquet', engine='auto')
#df_items = pd.read_parquet(r'Datasets/user_items.parquet', engine='auto')


app = FastAPI( 
    title = 'Machine Learning Operations (MLOps)',
    description='API para realizar consultas',
)


@app.get('/', tags=['inicio'])
async def inicio():
    cuerpo = '<center><h1 style="background-color:#daecfe;">Proyecto Individual Numero 1:<br>Machine Learning Operations (MLOps)</h1></center>'
    return HTMLResponse(cuerpo)

 
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
    
    mask1 = df_games['developer'] == desarrollador
    mask2 = df_games['price'] == 0.0
    developer = df_games[mask1]
    
    gratis = developer[mask2]
    cantidad_items= developer.groupby('release_year')['id'].count().to_dict()
    cantidad_gratis = gratis.groupby('release_year')['id'].count().to_dict()
    porcentaje_gratis = {year: f"{(cantidad_gratis.get(year, 0) / cantidad_items.get(year, 1)) * 100:.1f}%" for year in cantidad_items}
    total = {'cantidad de items por año':cantidad_items,
             'Cantidad gratis pos año':cantidad_gratis,
             'porcentaje gratis por año':porcentaje_gratis}
    return total