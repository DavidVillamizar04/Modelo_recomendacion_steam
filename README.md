# <h1 align=center> **MLOP (Machine Learning Operations)** </h1>

![alt text](Imagenes/Steam.jpg)

# Contenido

* [Introducción](#introducción)
* [Descripcion del problema](#descripcion-del-problema)
* [Propuesta](#propuesta)
* [ETL](#etl)
* [Modelo de recomendacion](#modelo-de-recomendacion)
* [API](#api)


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

...

# API 

..