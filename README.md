# <h1 align=center> **MLOP (Machine Learning Operations)** </h1>

![alt text](Imagenes/Steam.jpg)

# Contenido

* [Introducción](#introducción)
* [Descripcion del problema](#descripcion-del-problema)
* [Propuesta](#propuesta)
* [ETL](#etl)
* [Modelo de recomendacion](#modelo-de-recomendacion)
* [API](#api)
* [Fuente de datos](#fuente-de-datos)


# Introducción 

El aprendizaje automático, o machine learning en inglés, es una rama de la inteligencia artificial que se enfoca en el desarrollo de algoritmos y modelos que permiten a las computadoras aprender y mejorar su desempeño a partir de datos sin una programación explícita. Los algoritmos de machine learning son capaces de identificar patrones y tomar decisiones basadas en estos patrones, lo que los hace útiles en una amplia variedad de aplicaciones, desde reconocimiento de voz y visión por computadora hasta pronósticos de mercado y recomendaciones personalizadas.

Por otro lado, los MVP (Minimum Viable Products) son versiones simplificadas de un producto o servicio que incluyen solo las características más básicas necesarias para satisfacer las necesidades del cliente y validar su viabilidad en el mercado. Los MVP son utilizados comúnmente en el desarrollo de productos y aplicaciones para obtener retroalimentación temprana de los usuarios, minimizando la inversión de tiempo y recursos. Esto permite a los equipos iterar y mejorar gradualmente el producto basándose en la retroalimentación recibida, lo que aumenta las posibilidades de éxito a largo plazo. En el contexto del machine learning, los MVP pueden ser utilizados para probar y validar modelos de aprendizaje automático antes de su implementación completa, permitiendo ajustes y mejoras iterativas.


# <h1 align=center> **PROBLEMATICA** </h1>

<p align=center><img src=Imagenes/Problematica.png><p>

# Descripcion del problema

Se nos pide desarrollar un modelo de recomendacion para la empresa multinacional de videojuegos Steam, para esto se nos entregan tres conjuntos de datos, pero en ellos podemos notar que los datos no estan de una manera limpia, por lo cual se dificulta el trabajo para realizar el modelo, para ello tenemos que hacer la limpieza que consideremos necesaria, teniendo en cuenta tambien que tendremos que desarrollar una API por medio de la plataforma de Render la cual tiene limitaciones de uso de memoria RAM por lo cual debemos realizar un MVP.

# Propuesta 

No se pedian transformaciones para el MVP, por ende me enfoque en leer los archivos de forma correcta, despues se propone ralizar una limpieza de datos nulos y algunos duplicados, eliminar columnas que no vamos a usar optimizando todo al maximo, despues teniendo los datos listos se diseñara una API por medio de Render para poder realizar las consultas pedidas y incluir tambien el modelo de recomendacion. 

# <h1 align=center> **SOLUCION** </h1>

<p align=center><img src=Imagenes/Solucion.png><p>

# ETL

Teniendo como punto de partida que los archivos proporsionados se encuentran comprimidos en un formato GZ, como primer objetivo se tiene tratar de leer los datasets de una forma correcta. Despues de esto se logra observar que los Datasets tienen muchos campos vacios, en alguno hasta mas de la mitad, por esto se decide eliminar todos los valores nulos, permitiendo reducir el volumen de datos y mejorando la calidad de los datos que se tomaran despues. Dentro de este proceso teniendo los datos mas limpios decido eliminar y agregar algunas columnas que permitiran que las consultas en la API sean mas eficientes, por ultimo guarde los archivos listos para las consultas en la carpeta Datasets, en formaro parquet, el cual considere que es el mas eficiente.

# Modelo de recomendacion

El modelo de recomendación de coseno es una técnica de aprendizaje automático utilizada en sistemas de recomendación. Se basa en calcular la similitud entre elementos, como productos o usuarios, mediante el cálculo del coseno del ángulo entre sus vectores de características. Cuanto más cercano sea el ángulo entre los vectores, mayor será la similitud coseno y más probable será la recomendación. Es una herramienta efectiva y simple que se utiliza en plataformas de comercio electrónico y servicios de streaming para mejorar la relevancia de las recomendaciones.

# API 

<p align=center><img src=Imagenes/API.png><p>

Dentro del desarrollo de la `API` se solicitaban las siguientes consultas ademas de los sistemas de recomendacion:

+ def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora. 
    
+ def **userdata( *`User_id` : str* )**:
    Debe devolver `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

+ def **best_developer_year( *`año` : int* )**:
   Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)

+ def **developer_reviews_analysis( *`desarrolladora` : str* )**:
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. 

## Fuente de datos

+ [Dataset](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)