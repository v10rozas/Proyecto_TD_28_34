# Proyecto_TD_28_34
Proyecto de la asignatura Tratamiento de Datos realizado por Calin Cristian Dinga Pastae (100451528) y Víctor Díez Rozas (100451534).

## Descripción del problema y objetivos del proyecto
El objetivo del proyecto es desarrollar un modelo de regresión que, a partir de las características de una receta de cocina (e.g., sus instrucciones), prediga la valoración dada por un usuario.

Las características disponibles son: instrucciones, categorías, descripción, título, cantidad de grasa, cantidad de proteínas, cantidad de calorías, cantidad de sodio, ingredientes y fecha de publicación. Para determinar qué características se deben emplear en el modelo de regresión, se analiza la relación entre la característica ```categories``` y ```rating```.

Los pasos para realizar el análisis son:
1) Eliminar recetas que contengan valores nulos en ```categories``` o ```rating```.
2) Asociar a cada categoría su valoración.
3) Contar el número de veces que aparece cada categoría.
4) Filtrar las 400 categorías más frecuentes.
5) Calcular la valoración media de las 400 categorías más frecuentes.
6) Mostrar las 25 mejores y las 25 peores categorías valoradas.

<img width="671" alt="valoracion_de_categorias" src="https://github.com/user-attachments/assets/a4ccea0f-1e5b-4a85-a72d-65385acf31ca" />

La imagen anterior revela que las categorías con mejor ```rating``` están relacionadas con productos cárnicos (lamb chop, ground lamb, brisket, etc.) y con eventos/celebraciones (game, Kentucky derby, Rosh Hashanah/Yom Kippur, New Year's Day, etc.), mientras que las categorías con peor ```rating``` están relacionadas con bebidas alcohólicas (gin, bitters, spirit, alcoholic, etc.) y con otro tipo de bebidas (lime juice, tea, etc.). Entonces, se puede concluir que el modelo de regresión hay que entrenarlo con datos que contengan el tipo de alimento o de bebida, y las celebraciones.

En el enunciado se indica que se debe trabajar, como mínimo, con ```directions``` y/o ```desc```. En este caso se va a utilizar ```directions```. El motivo de omitir ```desc``` es que ```directions``` ya incluye toda la información sobre la receta, por lo que ```desc``` es contenido redundante.

También se encuentran las características ```ingredients``` y ```title```. La primera tendría sentido utilizarla ya que, como se ha visto en la imagen, los ingredientes son el factor más influyente en el ```rating``` (e.g., la carne está mucho mejor calificada que las bebidas). Sin embargo, se ha descartado para evitar superar el umbral máximo de tokens proporcionado por BERT<sup>[1]</sup>. La característica ```title``` no se utiliza ya que es contenido redundante; la información brindada en el título ya se incluye en las instrucciones.

La característica ```categories``` sí que se utiliza porque resulta útil para predecir el ```rating```, tal y como se ha visto antes.

La característica ```date``` también se utiliza ya que, como se ha visto en la imagen, las celebraciones influyen en el ```rating```. En concreto, se van a utilizar el día y el mes. Las horas, minutos, segundos, año y zona horaria no se consideran relevantes porque las festividades que se han visto se realizan todos los años, durante todo el día y en todas las regiones del planeta.

Las características ```fat```, ```protein```, ```calories``` y ```sodium``` no parecen influir de manera significativa. Apenas se observan entre los valores máximos y mínimos de ```rating```. Por tanto, no se van a emplear para entrenar el modelo.

<sup>[1]</sup> Entre ```ingredients``` y ```directions``` se ha decidido utilizar ```directions``` porque en el enunciado se indica que esta última es de obligado uso. Además, cabe mencionar que al descartar ```ingredients``` no se suele perder excesiva cantidad de información ya que en ```directions``` suelen aparecer los ingredientes más relevantes.

## Metodología aplicada
Antes de seguir, es importante mencionar que se ha reducido el número de recetas empleadas a 4000 porque las 20130 recetas suponían un elevado coste computacional.

### Preprocesado de los datos
El primer paso es realizar un preprocesado de los datos. Para ello, se va a emplear la librería SpaCy. Como se ha comentado en el apartado anterior, las características a las que se va a realizar el preprocesado son ```directions```, ```categories``` y ```date```. Sin embargo, cada una de ellas cuenta con un preprocesado diferente.

- La característica ```directions``` es una lista de strings. Entonces, para cada string se va a realizar la tokenización, la homogeneización y la limpieza.
- La característica ```categories``` es una lista de strings. Sin embargo, en lugar de realizar el mismo preprocesado que en ```directions```, solamente se realiza la conversión a minúsculas. El motivo de no realizar un preprocesado mayor es que las palabras que forman cada string ya constituyen una entidad, por lo que es conveniente mantenerlas juntas y que no pierdan información. Además, se convierten a minúsculas con el objetivo de reducir el tamaño del diccionario que se creará en el siguiente apartado.
- La característica ```date``` proporciona el año, mes, día, hora, minutos, segundos y zona horaria. Como se ha comentado anteriormente, el preprocesado que se realiza consiste en extraer el mes y el día en que se publicó la receta.
- El preprocesado finaliza tras juntar las tres características preprocesadas en una única lista.

```
  # Date: 
  ** Contenido: 2004-08-20 04:00:00+00:00
  ** Contenido preprocesado: ['8', '20']
  
  # Categories: 
  ** Contenido: ['Soup/Stew', 'Beef', 'Tomato', 'Celery', 'Fall', 'Simmer', 'Gourmet']
  ** Contenido preprocesado: ['soup/stew', 'beef', 'tomato', 'celery', 'fall', 'simmer', 'gourmet']
  
  # Directions: 
  ** Contenido: ['Whisk egg whites in a large bowl until foamy and add eggshells.' ...]
  ** Contenido preprocesado: ['whisk', 'egg', 'white', 'large', 'bowl', 'foamy', 'add', 'eggshell', ...]

  # Receta preprocesada:
  ['8', '20', 'soup/stew', 'beef', 'tomato', 'celery', 'fall', 'simmer', 'gourmet', 'whisk', 'egg', 'white',
  'large', 'bowl', 'foamy', 'add', 'eggshell', ...]
```
**Nota:** Ejemplo (recortado) de receta preprocesada.

### Vectorización de los datos
El segundo paso es vectorizar los datos preprocesados. En concreto, se van a realizar tres tipos de vectorización: TF-IDF, Word2Vec y embeddings contextuales de BERT.

- Los pasos para realizar la vectorización TF-IDF son:
  1) Generar el diccionario y filtrar aquellos tokens que aparezcan en menos de 4 recetas o en más del 80% de las recetas.
  2) Aplicar la vectorización BoW a cada una de las recetas.
  3) Generar el modelo TF-IDF y aplicar la vectorización TF-IDF a cada una de las recetas preprocesadas.
  4) Resultado: matriz ```4000x2354```, donde 4000 es el número de recetas y 2354 es el número de tokens del diccionario.

```
Los primeros cinco tokens de la primera receta sin vectorizar:
  ['8', '20', 'soup/stew', 'beef', 'tomato']
Los primeros cinco tokens de la primera receta en formato BoW:
  8->1.0, 20->1.0, soup/stew->1.0, beef->3.0, tomato->2.0
Los primeros cinco tokens de la primera receta en formato TF-IDF:
  8->0.0155, 20->0.0167, soup/stew->0.0844, beef->0.2815, tomato->0.1160
```
**Nota:** Ejemplo (recortado) de vectorización TF-IDF.

- Los pasos para realizar la vectorización Word2Vec son:
  1) Entrenar el modelo Word2Vec aplicando el algoritmo CBoW.
  2) Aplicar la vectorización Word2Vec a cada una de las recetas preprocesadas.
  3) Resultado: matriz ```4000x100```, donde 4000 es el número de recetas y 100 es el tamaño del vector Word2Vec.

```
La primera receta sin vectorizar:
['8', '20', 'soup/stew', 'beef', 'tomato', ..., 'season', 'salt', 'ladle', 'bowl', 'profiterole', 'garnish']

La primera receta vectorizada en formato Word2Vec:
[ 0.43711016  0.4392863  -0.04940761 -0.23191251  0.6653966   0.03018906 0.4440535   0.12933357  0.02830963
-0.5292608   0.05647407 -0.02353401 ...  0.16899839  0.4170615  -0.02847773  0.04467263 -0.35252956]
```
**Nota:** Ejemplo (recortado) de vectorización Word2Vec.

- Los pasos para realizar la vectorización de embeddings contextuales de BERT son:
  1) Preparar el corpus de datos. A BERT hay que pasarle las recetas sin el preprocesado realizado anteriormente. El único preprocesado que se ha realizado ha sido juntar las características ```date```, ```categories``` y ```directions``` en un único string, añadiendo delante de cada una el nombre de la característica.
  2) Cargar el modelo preentrenado.
  3) Extraer los embeddings contextuales como el promedio del tensor ```last_hidden_state```.
  4) Resultado: matriz ```4000x768```, donde 4000 es el número de recetas y 768 es el tamaño del vector embeddings.

```
La primera receta sin preprocesar:
"Date: month 8 and day 20. Categories: Soup/Stew, Beef, Tomato, Celery, Fall, Simmer, Gourmet. Directions:
Whisk egg whites in a large bowl until foamy and add eggshells..."

Los tokens asignados por BERT a la primera receta:
['date', ':', 'month', '8', 'and', 'day', '20', '.', 'categories', ':', 'soup', '/', 'stew', ',', 'beef', ',',
'tomato', ',', 'ce', '##ler', '##y', ',', 'fall', ',', 'sim', '##mer', ',', 'go', '##ur', '##met', '.',
'directions', ':', 'w', '##his', '##k', 'egg', 'whites', 'in', 'a', 'large', 'bowl', 'until', 'foam', '##y',
'and', 'add', 'eggs', '##hell', ...]

La primera receta vectorizada en formato embeddings:
tensor([-2.7185e-01,  5.5473e-02,  2.1420e-01,  1.5966e-01,  3.8665e-01, -1.1112e-01, -3.5711e-02,  3.8871e-01,
2.1933e-01, -2.1575e-01, -3.6420e-02, ..., -6.0454e-03, -1.0466e-01,  2.6690e-01])
```
**Nota:** Ejemplo (recortado) de vectorización embeddings.

### Redes neuronales
#### Hoja de ruta
1) Obtener los conjuntos de entrenamiento, validación y pruebas. Convertir los conjuntos a tensores de PyTorch. Generar lotes de entrenamiento (```batch_size=64```).
2) Entrenar, validar y evaluar la red neuronal más sencilla: el perceptrón simple.
3) Añadir nuevas capas (lineales y no lineales) al perceptrón para dar mayor expresividad al modelo. Entrenar, validar y evaluar el perceptrón multicapa.
4) Probar diferentes valores de tasa de aprendizaje para determinar cuál sería el valor óptimo en el perceptrón multicapa.
5) Probar técnicas vistas en clase para optimizar el perceptrón multicapa: ```dropout``` y ```early stopping```.
6) Trabajo futuro: aplicar otras técnicas para mejorar las prestaciones, como la inicialización de los pesos (Xavier y He), otros optimizadores (e.g., Adam), el uso de ```schedulers``` para la tasa de aprendizaje, ```lasso regularization```, ```ridge regularization```, ```batch normalisation```, ```data augmentation```, etc.

Sobre las redes neuronales:
- El número de características a la entrada se corresponde con el número de características de cada vectorización. El número de características a la salida es 1 porque se trata de un problema de regresión.
- En el perceptrón simple no hay capas ocultas. En el perceptrón multicapa hay una capa oculta. El número de neuronas de dicha capa es, aproximadamente, 2/3 el número total de características, tal y como se recomienda [aquí](https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3).
- En el perceptrón simple no hay funciones de activación no lineales. En el perceptrón multicapa hay una ReLU. De acuerdo con lo leído [aquí](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/), se recomienda que en los perceptrones siempre se utilice ReLU, por lo que no se prueban otras.
- En todos los casos, la función de activación de la capa de salida es la lineal. De acuerdo con lo leído [aquí](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/), no se puede emplear ninguna otra cuando se trata de un problema de regresión.
- Se ha establecido que la función de pérdidas durante el entrenamiento sea el MSE (Mean Squared Error). El motivo de su elección es que es el error que se suele emplear en problemas de regresión, tal y como se puede leer [aquí](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/).
- Se ha establecido que el optimizador durante el entrenamiento sea el SGD (Stochastic Gradient Descent). El motivo de su elección es que es el optimizador que se ha utilizado en la asignatura.
- Se ha establecido que, para validar los parámetros y evaluar la red neuronal, la métrica empleada sea el MAE (Mean Absolute Error) porque proporciona una medida más interpretable de los errores del modelo.
- Todos los entrenamientos se han llevado a cabo durante 400 épocas. Sin embargo, los regresores utilizados para evaluar sobre el conjunto de pruebas tienen los parámetros para los que el MAE del conjunto de validación es mínimo.

#### Análisis de los resultados
##### Perceptrón simple y multicapa. Validación de la tasa de aprendizaje.
Primero se entrenó, validó y evaluó el perceptrón simple. Los resultados obtenidos sobre el conjunto de pruebas se muestran en la siguiente tabla.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | **0.85**          | 0.88              |

De esta tabla se puede concluir que la mejor vectorización es TF-IDF. No obstante, el perceptrón simple es un regresor que, debido a su naturaleza lineal, no capta de manera adecuada las relaciones no lineales que pueden existir entre los datos. Por este motivo, se decidió añadir nuevas capas (lineales y no lineales) al regresor. En concreto, se añadió una capa oculta y una función de activación ReLU. Los nuevos resultados obtenidos sobre el conjunto de pruebas se muestran en la siguiente tabla.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | 0.85              | 0.88              |
| Perceptrón multicapa + lr=0.01    | 1.03              | 1.11              | **0.92**          |

Como se puede observar, la mejor vectorización para el perceptrón multicapa es Word2Vec. Sin embargo, se observa que las prestaciones al añadir nuevas capas son peores que las obtenidas en el perceptrón simple. Por este motivo, se decidió probar diferentes valores de tasa de aprendizaje (0'001, 0'0055, 0'055 y 0'1) para intentar mejorar las métricas, obteniendo los resultados que se muestran en la siguiente tabla.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | 0.85              | 0.88              |
|**Perceptrón multicapa + lr=0.001**| **0.89**          | **0.86**          | **0.88**          |
| Perceptrón multicapa + lr=0.0055  | 0.98              | 0.98              | 0.89              |
| Perceptrón multicapa + lr=0.01    | 1.03              | 1.11              | 0.92              |
| Perceptrón multicapa + lr=0.055   | 0.93              | 0.97              | 1.13              |
| Perceptrón multicapa + lr=0.1     | 1.06              | 0.87              | 1.04              |

De esta tabla se puede concluir que la mejor tasa de aprendizaje para los tres tipos de vectorización en el perceptrón multicapa es 0'001, con la vectorización TF-IDF mostrando el mejor rendimiento general en este caso. No obstante, se decidió continuar analizando si con las técnicas de ```dropout``` y ```early stopping``` se podían mejorar las prestaciones.

##### Dropout
Esta técnica se aplicó para TF-IDF con tasa de aprendizaje 0'1, Word2Vec con tasa de aprendizaje 0'0055 y embeddings con tasa de aprendizaje 0'001. Lo lógico hubiera sido aplicar ```dropout``` para las tasas de aprendizaje que mejores valores ofrecían en el punto anterior (0'001 en los tres casos). Sin embargo, en TF-IDF y Word2Vec con tasa de aprendizaje 0'001 no había sobreentrenamiento, por lo que aplicar esta técnica no tenía sentido. En su lugar, se utilizaron las segundas mejores tasas de aprendizaje para ver si se podían mejorar sus prestaciones.

Además, el valor de la probabilidad de ```dropout``` se estableció inicialmente en 0'5 para las tres técnicas de vectorización. Este valor fue suficiente para evitar el sobreentrenamiento en Word2Vec y embeddings. Sin embargo, no ocurrió lo mismo en TF-IDF, lo que llevó a aumentar su tasa de ```dropout``` a 0'8 en un intento de mitigar este problema. Aunque este incremento ayudó a reducir el sobreentrenamiento, los resultados apenas mejoraron. Por ello, se considera que una posible mejora futura sería aplicar una técnica de regularización adicional para reducir aún más el sobreentrenamiento en TF-IDF.

Los resultados obtenidos tras aplicar ```dropout``` se muestran en la siguiente tabla. De ellos, se concluye que aplicar ```dropout``` mejora ligeramente las prestaciones de Word2Vec y embeddings. Sin embargo, no ocurre lo mismo con TF-IDF.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | 0.85              | 0.88              |
| Perceptrón multicapa + lr=0.001   | **0.89**          | **0.86**          | **0.88**          |
| Perceptrón multicapa + lr=0.0055  | 0.98              | 0.98              | 0.89              |
| Perceptrón multicapa + lr=0.01    | 1.03              | 1.11              | 0.92              |
| Perceptrón multicapa + lr=0.055   | 0.93              | 0.97              | 1.13              |
| Perceptrón multicapa + lr=0.1     | 1.06              | 0.87              | 1.04              |
| MLP + lr=0.1 + dropout            | -                 | **0.91**          | -                 |
| MLP + lr=0.0055 + dropout         | -                 | -                 | **0.85**          |
| MLP + lr=0.001 + dropout          | **0.89**          | -                 | -                 |

##### Early stopping
Esta técnica se aplicó para el mejor modelo de cada vectorización: TF-IDF con tasa de aprendizaje 0'001, Word2Vec con tasa de aprendizaje 0'0055 y ```dropout```, y embeddings con tasa de aprendizaje 0'001 y ```dropout```. Los resultados obtenidos se muestran en la siguiente tabla. De ellos, se concluye que aplicar ```early stopping``` no mejora las prestaciones de ninguna vectorización. La razón de este empeoramiento de los resultados es que ```early stopping``` hace que el entrenamiento se detenga exceisvamente pronto.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | 0.85              | 0.88              |
| Perceptrón multicapa + lr=0.001   | 0.89              | **0.86**          | 0.88              |
| Perceptrón multicapa + lr=0.0055  | 0.98              | 0.98              | 0.89              |
| Perceptrón multicapa + lr=0.01    | 1.03              | 1.11              | 0.92              |
| Perceptrón multicapa + lr=0.055   | 0.93              | 0.97              | 1.13              |
| Perceptrón multicapa + lr=0.1     | 1.06              | 0.87              | 1.04              |
| MLP + lr=0.1 + dropout            | -                 | 0.91              | -                 |
| MLP + lr=0.0055 + dropout         | -                 | -                 | **0.85**          |
| MLP + lr=0.001 + dropout          | **0.89**          | -                 | -                 |
| MLP + lr=0.001 + es               | -                 | **0.91**          | -                 |
| MLP + lr=0.0055 + dropout + es    | -                 | -                 | **0.92**          |
| MLP + lr=0.001 + dropout + es     | **0.89**          | -                 | -                 |

##### Elección de la red neuronal
A la vista de los resultados, la red neuronal que mejores prestaciones tiene es el perceptrón multicapa con vectorización Word2Vec, tasa de aprendizaje 0'0055 y ```dropout``` ya que es el que mejor valor de MAE de pruebas tiene.

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | 0.95              | 0.85              | 0.88              |
| Perceptrón multicapa + lr=0.001   | 0.89              | 0.86              | 0.88              |
| Perceptrón multicapa + lr=0.0055  | 0.98              | 0.98              | 0.89              |
| Perceptrón multicapa + lr=0.01    | 1.03              | 1.11              | 0.92              |
| Perceptrón multicapa + lr=0.055   | 0.93              | 0.97              | 1.13              |
| Perceptrón multicapa + lr=0.1     | 1.06              | 0.87              | 1.04              |
| MLP + lr=0.1 + dropout            | -                 | 0.91              | -                 |
| **MLP + lr=0.0055 + dropout**     | -                 | -                 | **0.85**          |
| MLP + lr=0.001 + dropout          | 0.89              | -                 | -                 |
| MLP + lr=0.001 + es               | -                 | 0.91              | -                 |
| MLP + lr=0.0055 + dropout + es    | -                 | -                 | 0.92              |
| MLP + lr=0.001 + dropout + es     | 0.89              | -                 | -                 |

### Regresor k-NN
#### Hoja de ruta
1) Obtener los conjuntos de entrenamiento y de pruebas. Es importante mencionar que el conjunto de entrenamiento incluye el conjunto de validación porque en este apartado se utilizará validación cruzada. Además, las muestras asignadas a cada conjunto son las mismas que en redes neuronales ya que se les ha asignado el mismo ```random_state```.
2) El regresor k-NN cuenta con [varios hiperparámetros](https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsRegressor.html). En este proyecto se analiza el valor óptimo del número de vecinos (```n_neighbors```) en un rango de 1 a 100. El motivo de elegir un rango amplio de vecinos se debe a que el número de muestras utilizadas para entrenar el regresor k-NN es elevado. El resto de los hiperparámetros mantienen sus valores por defecto.
3) El valor óptimo de ```n_neighbors``` se va a determinar mediante validación cruzada utilizando el conjunto de entrenamiento. En concreto, se aplica ```10-Fold cv```, tal y como se realizó en la práctica de regresión de la asignatura. Además, la métrica para determinar el valor óptimo de ```n_neighbors``` es el MAE (Mean Absolute Error). El motivo de su elección es que es una métrica que se suele emplear en problemas de regresión, además de ser la métrica que se ha ido analizando en la red neuronal.
4) Determinar las prestaciones de cada regresor k-NN hallando el MAE sobre el conjunto de pruebas.

#### Análisis de los resultados
Los resultados mostrados en la siguiente tabla indican, para cada regresor k-NN, su número óptimo de vecinos y el MAE obtenido sobre el conjunto de pruebas. Se observa que la vectorización que mejores prestaciones ofrece es embeddings, seguido de TF-IDF y Word2Vec. Además, en cuanto al número de vecinos, las vectorizaciones embeddings y Word2Vec necesitan un número menor de vecinos respecto a TF-IDF.

| **Vectorización**  | **Número de vecinos óptimo** | **MAE (pruebas)**     |
|--------------------|------------------------------|-----------------------|
| **Embeddings**     | **37**                       | **0.81**              |
| TF-IDF             | 42                           | 0.85                  |
| Word2Vec           | 37                           | 0.86                  | 

#### Elección del regresor k-NN
A la vista de los resultados, el regresor k-NN que mejores prestaciones tiene es el que utiliza la vectorización embeddings. Por un lado, es el que mejor valor de MAE de pruebas tiene. Por otro lado, es el que menor número de vecinos necesita.

### Fine-tuning de BERT
El modelo preentrenado que se va a utilizar para realizar fine-tuning es BERT. Se ha elegido porque es el transformers principal que se ha estudiado en la asignatura. Un trabajo futuro sería analizar otros modelos y diferentes arquitecturas (```encoder-only```, ```decoder-only``` y ```encoder-decoder```), teniendo en cuenta las limitaciones de hardware que tiene Google Colab.

Los pasos para realizar el fine-tuning son:
1) Tokenizar el conjunto de entrenamiento y de pruebas.
2) Adaptar los conjuntos de datos a objetos de la clase ```Dataset```, necesaria para trabajar con Hugging Face.
3) Cargar el modelo preentrenado BERT y configurarlo para un problema de regresión.
4) Entrenar el modelo con el conjunto de entrenamiento.
5) Evaluar el modelo con el conjunto de pruebas y la métrica MAE. El motivo de su elección es que es una métrica que se suele emplear en problemas de regresión, además de ser la métrica que se ha ido analizando en la red neuronal y el regresor k-NN.

El resultado obtenido es un modelo que, al ser evaluado sobre el conjunto de pruebas, obtiene un MAE de 0'85. Además, cabe destacar el buen rendimiento del fine-tuning; BERT es un modelo que no ha sido preentrenado para predecir valoraciones, pero es capaz de ofrecer buenas prestaciones.

### Red Neuronal vs k-NN vs fine-tuning
El mejor regresor k-NN destaca sobre la mejor red neuronal y el modelo con fine-tuning. Los motivos son:

1) Simplicidad: El regresor k-NN solamente necesita validar un parámetro (número de vecinos), mientras que los otros dos dependen de muchos parámetros (número de capas, número de neuronas en cada capa, tasa de aprendizaje, número de épocas, optimizador, etc.).

2) Prestaciones: El regresor k-NN junto con la vectorización embeddings ofrece los mejores resultados.

| **Regresor** | **MAE (pruebas)** |
|--------------|-------------------|
| Red Neuronal | 0.85              |
| **k-NN**     | **0.81**          |
| fine-tuning  | 0.85              | 

# Extensión
La extensión que se va a implementar es: ```uso de un summarizer preentrenado (utilizando pipelines de Hugging Face) para proporcionar un resumen de la característica 'directions'```.

- Se va a probar la capacidad de resumir de tres modelos: [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)<sup>[1]</sup>, [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) y [DISLab/SummLlama3.2-3B](https://huggingface.co/DISLab/SummLlama3.2-3B)<sup>[2]</sup>.
- Para determinar cómo de buenos son los resúmenes obtenidos, lo ideal sería compararlos con los realizados por un ser humano. Sin embargo, para optimizar el tiempo de trabajo, los resúmenes obtenidos de los tres modelos se van a comparar con los resúmenes generados por ChatGPT-4o. 
- Para obtener un valor cuantitativo de la calidad de los resúmenes obtenidos, se van a aplicar las [métricas ROUGE y BLEU](https://neptune.ai/blog/llm-evaluation-text-summarization). El motivo de su elección es el bajo coste computacional que conllevan. Además, se utilizará la métrica BERT score que, aunque suponga un coste computacional mayor, determinará mejor la calidad de los resúmenes obtenidos.

<sup>[1]</sup> Se trata de un modelo con arquitectura encoder-decoder. En concreto, es el fine-tuning de [facebook/bart-large](https://huggingface.co/facebook/bart-large).

<sup>[2]</sup> Se trata de un modelo con arquitectura decoder-only. En concreto, es el fine-tuning de [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), que a su vez es el fine-tunning de [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).

## Ejemplo

```
INSTRUCCIONES:
Combine dried peaches and water in heavy medium saucepan. Let stand 30 minutes.
Bring to boil. Reduce heat and simmer gently until peaches are very soft, about
20 minutes. Meanwhile, place 1/4 cup Pêcher Mignon in small bowl; sprinkle
gelatin over. Set aside to soften. Beat 3 egg yolks, 1/2 cup sugar and remaining
1/4 cup Pêcher Mignon in top of double boiler until light in color. Set over
simmering water and stir until thick enough to coat back of spoon when finger is
drawn across (do not boil), about 2 minutes. Transfer to small bowl. Add gelatin
to hot peach mixture and stir until dissolved. Transfer to processor and puree
until smooth. Pour into large bowl. Cool to room temperature, stirring occasionally
(do not let peach mixture set up.) Add custard to peach mixture and whisk to
combine. Using electric mixer, beat cream in medium bowl to soft peaks. Mix 1/3
of whipped cream into peach mixture to lighten. Gently fold in remaining cream in
2 batches. Using clean dry beaters, beat egg whites and cream of tartar in another
medium bowl until frothy. Gradually add remaining 3 1/2 tablespoons sugar and beat
until smooth, glossy and almost stiff but not dry. Fold 1/3 of whites into peach
mixture to lighten. Gently fold in remaining whites in 2 batches. Cover bowl with
plastic and refrigerate mousse 8 hours or overnight. Spoon mousse into pastry bag
fitted with large star tip. Pipe mousse into goblets or wine glasses. Top mousse
with fresh mint sprigs and serve.


RESUMEN GENERADO POR ChatGPT-4o:
This recipe makes a peach mousse. First, dried peaches are softened in water,
boiled, and pureed. Gelatin is dissolved in peach liquid, then combined with
a custard made from egg yolks, sugar, and Pêcher Mignon. Whipped cream and
beaten egg whites are gently folded into the peach mixture to create a light
and airy mousse. The mousse is refrigerated for 8 hours or overnight, then
piped into glasses and topped with fresh mint before serving.
    

RESUMEN GENERADO POR meta-llama/Llama-3.2-3B-Instruct:
The document is a recipe for a peach mousse dessert. Here's a summary of the
steps involved:

1. Make a peach mixture by combining dried peaches and water, letting it sit,
then boiling and reducing heat to simmer until the peaches are soft. Add gelatin
and any remaining ingredients to the mixture and puree until smooth.

2. Prepare a custard mixture by beating egg yolks, sugar, and Pêcher Mignon in
a double boiler until thickened.

3. Combine the peach mixture with the custard mixture and add whipped cream to
lighten the mixture.

4. Whip egg whites and cream of tartar, then gently fold into the peach mixture.

5. Cover the bowl with plastic and refrigerate the mousse for at least 8 hours
or overnight.

6. Pipe the mousse into goblets or wine glasses and top with fresh mint sprigs
before serving.


RESUMEN GENERADO POR DISLab/SummLlama3.2-3B:
The recipe described is for a peach mousse, involving the preparation of a peach
puree, the creation of a custard base, and the folding of whipped cream and egg
whites to combine, all of which are then piped into pastry bags and set to chill
in the refrigerator before serving, garnished with fresh mint sprigs.


RESUMEN GENERADO POR facebook/bart-large-cnn:
Dried peaches and water in heavy medium saucepan. Bring to boil. Reduce heat and
simmer gently until peaches are very soft, about 20 minutes. Meanwhile, place 1/4
cup Pêcher Mignon in small bowl; sprinkle gelatin over. Set aside to soften.
```

## ROUGE, BLEU y BERT score

- ROUGE indica las coincidencias de tokens (o N-gramas) entre dos textos: el de referencia (resúmenes generados por ChatGPT-4o) y el de hipótesis (resúmenes generados por los otros tres modelos).

- Existen múltiples variantes de ROUGE. En este proyecto se ha utilizado ROUGE-1. Los pasos para calcular la métrica son: 1) tokenizar los resúmenes, 2) calcular cuántos tokens tienen en común la referencia y la hipótesis, 3) calcular ```precision```.

<br>

$$\text{precision} = \frac{\text{Número de tokens solapados}}{\text{Número total de tokens en el resumen de hipótesis}}$$

<br>

- Los pasos para calcular la métrica BLEU son: 1) tokenizar los resúmenes, 2) calcular los tokens, bigramas y trigramas que tienen en común la referencia y la hipótesis, 3) calcular el valor de ```precision``` para los tokens, bigramas y trigramas, 4) calcular una penalización basada en la longitud de los resúmenes de referencia y de hipótesis, 4) calcular BLEU como el producto de los tres valores de ```precision``` y la penalización.

- BERT score hace uso del modelo RoBERTa (355M de parámetros) para obtener los embeddings del texto de referencia y de hipótesis. Después, calcula la similitud coseno entre ambos.

- Las tres métricas tienen un rango de valores de 0 a 1, donde 0 significa que el resumen de la hipótesis es de baja calidad y 1 indica que es de alta calidad.

## Análisis de los resultados
Los resultados obtenidos se resumen en la siguiente tabla:

| Modelo                           | Recetas resumidas            | ROUGE-1 precision  | BLEU   | BERT score |
|----------------------------------|------------------------------|--------------------|--------|------------|
| meta-llama/Llama-3.2-3B-Instruct | 46                           | 0.3948             | 0.1428 | 0.8775     |
| DISLab/SummLlama3.2-3B           | 46                           | 0.4437             | 0.1341 | 0.9020     |
| facebook/bart-large-cnn          | 46                           | 0.3772             | 0.0585 | 0.8742     |

<br>

En un principio, solo se utilizaron las métricas ROUGE y BLEU para evaluar los resúmenes. Con estas métricas, se observó que la calidad de los tres modelos era bastante baja, especialmente en el caso de ```facebook/bart-large-cnn```. No obstante, al emplear BERT score, las conclusiones son completamente diferentes. Esta gran discrepancia entre los resultados de las métricas se debe a las limitaciones de ROUGE y BLEU. Ambas métricas se basan únicamente en la coincidencia de n-gramas, lo que las hace incapaces de capturar el significado del texto generado.

Si se analizan los resultados de BERT score:
- Se observa que los modelos ```meta-llama/Llama-3.2-3B-Instruct``` y ```DISLab/SummLlama3.2-3B``` son mejores que ```facebook/bart-large-cnn```. Esto puede deberse a que los dos primeros modelos cuentan con 3.21B de parámetros respecto a los 406M del último. Aunque el número de parámetros no asegura automáticamente mejores resultados, generalmente sí contribuye a un mejor desempeño.
- Se observa que ```DISLab/SummLlama3.2-3B``` es mejor que ```meta-llama/Llama-3.2-3B-Instruct```. Esto tiene sentido ya que el primer modelo ha sido especializado en resumir, mientras que el segundo no.
