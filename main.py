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

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_reviews["review"])
    
feactures =  np.column_stack([tfidf_matrix.toarray(), df_reviews["recommend"], df_reviews["sentiment_analysis"]])

similarity_matrix = cosine_similarity(feactures)
df_sentiment_analysis = df_reviews.reset_index(drop=True)

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

@app.get("/recomendacion_juego/{item_id}",  tags=['recomendacion'])
async def recomendacion_juego (item_id : int):
    '''
    <strong>Devuelve una cantidad de 5 juegos recomendado a partir del identifcador de un juego</strong>
             
    Parametro
    ---------
             item_id : Identificador unico del juego.
    
    Retorna
    -------   
             Diccionario con una lista de 5 juegos similiares recomendados a partir del ingresado
    '''

    producto = df_sentiment_analysis[df_sentiment_analysis['item_id'] == item_id]
    if not producto.empty:
        product_index = producto.index[0]
        product_similarities = similarity_matrix[product_index]
        most_similar_products_indices = np.argsort(-product_similarities)
        most_similar_products = df_sentiment_analysis.loc[most_similar_products_indices, 'item_name']
    else:
        return "Producto no encontrado"
    
    diccionario = {"Juegos recomendados" : list()}
    similares = most_similar_products[:5]
    diccionario["Juegos recomendados"] = [similar for similar in similares]
    diccionario

    return diccionario

# Endpoint http://127.0.0.1:8000/recomendacion_usuario/{user_id}
@app.get("/recomendacion_usuario/{user_id}",  tags=['recomendacion'])
async def recomendacion_usuario (user_id):
    '''
    <strong>Devuelve una cantidad de 5 juegos recomendado a partir del identifcador unico de un usuario</strong>
             
    Parametro
    ---------
             user_id : Identificador unico del juego.
    
    Retorna
    -------   
             Diccionario con una lista de 5 juegos similares recomendados por un usuario
    '''

    producto = df_sentiment_analysis[df_sentiment_analysis['user_id'] == user_id]
    if not producto.empty:
        product_index = producto.index[0]
        product_similarities = similarity_matrix[product_index]
        most_similar_products_indices = np.argsort(-product_similarities)
        most_similar_products = df_sentiment_analysis.loc[most_similar_products_indices, 'item_name']
    else:
        return "Producto no encontrado"
    
    diccionario = {"Juegos recomendados" : list()}
    similares = most_similar_products[:5]
    diccionario["Juegos recomendados"] = [similar for similar in similares]
    diccionario

    return diccionario