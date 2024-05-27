# Importa librerías necesarias
from fastapi import FastAPI
import pandas as pd
import importlib
import numpy as np
import pyarrow
import fastparquet
import parquet
import operator
from sklearn.metrics.pairwise import cosine_similarity

# Usar datasets desde parquet (consultas)
genres_year_playtime = pd.read_parquet('genres_year_playtime.parquet')
max_playtime_by_genre_year= pd.read_parquet('max_playtime_by_genre_year.parquet')
user_year_playtime = pd.read_parquet('user_year_playtime.parquet')
year_recommend_game = pd.read_parquet('year_recommend_game.parquet')
top3_recomendados = pd.read_parquet('top3_recomendados.parquet')
top3_no_recomendados = pd.read_parquet('top3_no_recomendados.parquet')
developer_sentiment = pd.read_parquet('developer_sentiment.parquet')

# Usar datasets desde parquet (recomendación)
unique_item_ids = pd.read_parquet('unique_item_ids.parquet')
user_name_time = pd.read_parquet('user_name_time.parquet')
piv_table_norm = pd.read_parquet('piv_table_norm.parquet')
df_item_simil = pd.read_parquet('df_item_simil.parquet')
df_user_simil = pd.read_parquet('df_user_simil.parquet')

# Se instancia la aplicación
app = FastAPI(title="PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Amelia Herrera Briceño - melinnicri@gmail.com",
            description="API de datos y recomendaciones de juegos de video online STEAM")


# Función para reconocer el servidor local
@app.get("/")
async def index():
    return {"Hola! Bienvenido a la API de consulta y recomedación. Por favor dirígete a /docs"}

@app.get("/about/")
async def about():
    return {"PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)"}


# Primera consulta:
@app.get("/playtimegenre/({genero})")
def PlayTimeGenre(genero: str):
    '''Se ingresa el Género y la función retorna el Año con más horas jugadas'''

    # Convertir el género a minúsculas para hacer la comparación sin ser sensible a mayúsculas
    genero = genero.lower()

    # Verificar si el género está presente en el DataFrame
    if genero not in max_playtime_by_genre_year["genres"].str.lower().unique():
        return {"Error": f"El género '{genero}' no se encuentra en el conjunto de datos."}

    # Filtrar el DataFrame por el género proporcionado
    genre_data = max_playtime_by_genre_year[max_playtime_by_genre_year["genres"].str.lower() == genero]

    # Obtener el año con más horas jugadas para el género dado
    max_playtime_year = genre_data["release_year"].values[0] if not genre_data.empty else None


    # Crear el diccionario para el formato JSON
    result = {f"Año de lanzamiento con más horas jugadas para {genero.capitalize()}": str(max_playtime_year)}

    return result

# Segunda consulta:
@app.get("/userforgenre/({genero})")
def UserForGenre(genero: str): # Este funciona con GENRE, NO CON USER!!!
    # Agrupar por usuario y año, sumar las horas jugadas y encontrar el usuario con más horas jugadas
    result_df = user_year_playtime.groupby(["user_id", "release_year"])["playtime_forever"].sum().reset_index()
    max_user = result_df.loc[result_df["playtime_forever"].idxmax()]

    # Convertir las horas jugadas de minutos a horas en el DataFrame resultante
    result_df["playtime_forever"] = result_df["playtime_forever"] / 60
    result_df["playtime_forever"] = result_df["playtime_forever"].round(0)

    # Crear una lista de acumulación de horas jugadas por año
    accumulation = result_df.groupby("release_year")["playtime_forever"].sum().reset_index()
    accumulation = accumulation.rename(columns={"release_year": "Año", "playtime_forever": "Horas"})
    accumulation_list = accumulation.to_dict(orient="records")

    return {"Usuario con más horas jugadas para el género " + genero: max_user["user_id"], "Horas jugadas": accumulation_list}

# Tercera consulta:
@app.get("/usersrecommend/({anio})")
def UsersRecommend(anio: int): 
    '''Se ingresa el Año y la función retorna el Top 3 de los juegos más recomendados por los usuarios'''
    # Verificar si el año proporcionado tiene datos en el DataFrame top3_recomendados
    if anio not in top3_recomendados["release_year"].unique():
        return f"No hay datos disponibles para el año {anio}"

    # Filtrar el DataFrame top3_recomendados para el año proporcionado
    top_games_year = top3_recomendados[top3_recomendados["release_year"] == anio]

    # Verificar si hay menos de 3 juegos para el año consultado
    if len(top_games_year) < 1:
        return "No hay suficientes juegos para mostrar el top 3"

    # Reiniciar el índice del DataFrame filtrado por año
    top_games_year = top_games_year.reset_index(drop=True)

    # Obtener el top 3 de juegos más recomendados para el año dado
    top_3_games = top_games_year.head(3)

    # Crear una lista con el formato especificado
    top_3_list = []
    for index, row in top_3_games.iterrows():
        game_rank = index + 1  # Usar el índice + 1 como número de puesto
        game_dict = {"Puesto " + str(game_rank): row["item_name"]}
        top_3_list.append(game_dict)

    return top_3_list

# Cuarta consulta:
@app.get("/usersworstdeveloper/({anio})")
def UsersWorstDeveloper(anio: int):
    '''Se ingresa el Año y la función retorna el top 3 de Desarrolladores con MENOS recomendaciones por Usuarios'''
    # Verificar si el año proporcionado tiene datos en el DataFrame top3_no_recomendados
    if anio not in top3_no_recomendados["release_year"].unique():
        return f"No hay datos disponibles para el año {anio}"

    # Filtrar el DataFrame top3_no_recomendados para el año proporcionado
    top_games_year = top3_no_recomendados[top3_no_recomendados["release_year"] == anio]

    # Verificar si hay menos de 3 juegos para el año consultado
    if len(top_games_year) < 1:
        return "No hay suficientes juegos para mostrar el top 3 menos recomendados"

    # Reiniciar el índice del DataFrame filtrado por año
    top_games_year = top_games_year.reset_index(drop=True)

    # Obtener el top 3 de juegos más recomendados para el año dado
    top_3_games = top_games_year.head(3)

    # Crear una lista con el formato especificado
    top_3_list = []
    for index, row in top_3_games.iterrows():
        game_rank = index + 1  # Usar el índice + 1 como número de puesto
        game_dict = {"Puesto " + str(game_rank): row["developer"]}
        top_3_list.append(game_dict)

    return top_3_list

# Quinta consulta:
@app.get("/sentimentanalysis/({desarrollador})")
def SentimentAnalysis(desarrollador: str): 
    '''Se ingresa el Desarrollador y la función retorna con la lista de registros totales de reseñas Negativa, Positiva, o Neutral de los Usuarios'''
    # Filtrar las reseñas por el desarrollador dado
    reseñas_desarrolladora = developer_sentiment[developer_sentiment["developer"] == desarrollador]

    # Contar las reseñas con sentimiento negativo, positivo o neutro
    sentimiento_negativo = reseñas_desarrolladora[reseñas_desarrolladora["sentiment_analysis"] == 0].shape[0]
    sentimiento_neutral = reseñas_desarrolladora[reseñas_desarrolladora["sentiment_analysis"] == 1].shape[0]
    sentimiento_positivo = reseñas_desarrolladora[reseñas_desarrolladora["sentiment_analysis"] == 2].shape[0]

    # Crear el diccionario de resultados
    resultado = {desarrollador: {"Negative": sentimiento_negativo, "Neutral": sentimiento_neutral, "Positive": sentimiento_positivo}}

    return resultado

# Consulta recomendación 1:
#@app.get("/recomendacionjuego/({game}{df_item_simil})")
# Los 5 juegos más recomendados por juego similar...
#def recommended_games_item(game, df_item_simil):
#    similar_games = {}
#    count = 1
#    for item in df_item_simil.sort_values(by=game, ascending=False).index[1:6]:
#        similar_games[f"Recomendación {count}"] = item
#        count += 1
#    return similar_games

# Consulta recomendación 2:
@app.get("/recomendacionjuegousuario/({user})")
# Los 5 juegos más recomendados similares por usuario...
def similar_user_recs(user):
    
    # Se verifica si el usuario está presente en las columnas de piv_table_norm
    if user not in piv_table_norm.columns:
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

    # Se ordenan los juegos de mayor recomendacion
    sorted_list = sorted(most_common.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_list[:5])
