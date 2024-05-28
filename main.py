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
    
    mask1 = df_games['developer'] == desarrollador
    mask2 = df_games['price'] == 0
    developer = df_games[mask1]
    
    gratis = developer[mask2]
    cantidad_items= developer.groupby('release_year')['id'].count()
    cantidad_gratis = gratis.groupby('release_year')['id'].count()
    cantidad_items = cantidad_items.to_dict()
    cantidad_items = cantidad_items.items()
    cantidad_gratis = cantidad_gratis.to_dict()
    cantidad_gratis = cantidad_gratis.items()
    cantidad_items = pd.DataFrame(cantidad_items, columns=['Año','Cantidad de items'])
    cantidad_gratis= pd.DataFrame(cantidad_gratis, columns= ['Año','Cantidad gratis'])
    total = pd.concat([cantidad_items,cantidad_gratis['Cantidad gratis']], axis=1)
    total['Cantidad gratis'] = total['Cantidad gratis'].fillna(0).astype(int)
    total['Cantidad gratis'] = ((total['Cantidad gratis']/total['Cantidad de items'])*100).astype(int)
