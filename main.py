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
    version='1.0 / Jesus Parra (2024)'
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
    Devuelve un diccionario año, cantidad de items y porcentaje de contenido libre por empresa desarrolladora
             
    Parametro
    ---------  
            desarrollador : Nombre de la empresa desarrolladora
    
    Retorna
    -------
            Anio                         : Año
            Cantidad Items               : Videos juegos desarrollados
            Porcentaje de contenido Free : Porcetnaje de contenidos gratuito
    '''
    
    lista_diccioanario = {"Anio" : list(),"Cantidad de items" : list(),"Porcentaje de contenido Free" : list()}

    # Filtramos por desarrollador
    el_desarrollador = df_games[df_games['developer'] == desarrollador]
    # Calcula el total de items por año
    cantidad_items = el_desarrollador.groupby('year')['item_id'].count()
    # Calcula el total de contenido gratis por año
    cantidad_gratis = el_desarrollador[el_desarrollador['price'] == 0.0].groupby('year')['item_id'].count()
    # Calcula el porcentaje de contenido gratis por año
    porcentaje_gratis = (cantidad_gratis / cantidad_items * 100).fillna(0).astype(int)
    # Damos formato para el retorno de la informacion
    for year, item_id_counts in cantidad_items.items():
        lista_diccioanario["Anio"].append(year)
        lista_diccioanario["Cantidad de items"].append(item_id_counts)
    for year, item_porc in porcentaje_gratis.items():
        lista_diccioanario["Porcentaje de contenido Free"].append(item_porc)
    
    diccionario = pd.DataFrame(lista_diccioanario).to_dict(orient='records')
    
    return  "No existen registros" if len(el_desarrollador) == 0 else diccionario
