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

# Se muestra en la siguiente ruta un sencillo titulo http://127.0.0.1:8000/
@app.get('/', tags=['inicio'])
async def inicio():
    cuerpo = '<center><h1 style="background-color:#daecfe;">Proyecto Individual Numero 1:<br>Machine Learning Operations (MLOps)</h1></center>'
    return HTMLResponse(cuerpo)

# Endpoint http://127.0.0.1:8000/developer/{desarrollador} 
@app.get("/developer/{desarrollador}",  tags=['developer'])
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
    
    data_filtrada = df_games[df_games["developer"].str.lower() == desarrollador]
    # Cantidad de items por año
    cantidad_items = data_filtrada.groupby("year")["items_count"].count().to_dict()
    # Calculamos la cantidad de contenido free por año
    cantidad_free = data_filtrada[data_filtrada["price"] == 0.0].groupby("year")["items_count"].count().fillna(0).to_dict()
    # Se calcula el porcentaje free, redondeado a un decimal
    porcentaje_free = {year: f"{(cantidad_free.get(year, 0) / cantidad_items.get(year, 1)) * 100:.1f}%" for year in cantidad_items}
    
    # Formato de salida en JSON
    output = {
            "Agnos": cantidad_items,
            "Cantidad de items free": cantidad_free,
            "Porcentaje free por agno": porcentaje_free
            }
    return output