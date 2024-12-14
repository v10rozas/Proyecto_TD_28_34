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
El segundo paso es vectorizar los datos preprocesados. En concreto, se van a realizar tres tipos de vectorización: TF-IDF, word2vec y embeddings contextuales de BERT.

- Los pasos para realizar la vectorización de TF-IDF son:
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
['date', ':', 'month', '8', 'and', 'day', '20', '.', 'categories', ':', 'soup', '/', 'stew', ',', 'beef', ',', 'tomato',
',', 'ce', '##ler', '##y', ',', 'fall', ',', 'sim', '##mer', ',', 'go', '##ur', '##met', '.', 'directions', ':', 'w',
'##his', '##k', 'egg', 'whites', 'in', 'a', 'large', 'bowl', 'until', 'foam', '##y', 'and', 'add', 'eggs', '##hell', ...]

La primera receta vectorizada en formato embeddings:
tensor([-2.7185e-01,  5.5473e-02,  2.1420e-01,  1.5966e-01,  3.8665e-01, -1.1112e-01, -3.5711e-02,  3.8871e-01,
2.1933e-01, -2.1575e-01, -3.6420e-02, ..., -6.0454e-03, -1.0466e-01,  2.6690e-01])
```
**Nota:** Ejemplo (recortado) de vectorización embeddings.

### Preparación de los datos para los modelos

El tercer paso es entrenar los modelos de regresión. Sin embargo, antes hay que preparar los datos:
1) Obtener los conjuntos de entrenamiento y de pruebas.
2) En el caso de las redes neuronales, convertir las recetas vectorizadas a tensores de PyTorch y, posteriormente, generar los lotes de entrenamiento y de pruebas. Cabe mencionar que el tamaño de los lotes (```batch_size```) es de 64 y que, para este proyecto, no se han probado otros valores.

### Redes neuronales
#### Hoja de ruta
1) Obtener los conjuntos de entrenamiento, validación y pruebas. Convertir los conjuntos a tensores de PyTorch. Generar lotes de entrenamiento (```batch_size=64```).
2) Entrenar, validar y evaluar la red neuronal más sencilla: el perceptrón simple.
3) Añadir nuevas capas (lineales y no lineales) al perceptrón para dar mayor expresividad al modelo. Entrenar, validar y evaluar el perceptrón multicapa.
4) Probar diferentes valores de tasa de aprendizaje y épocas para determinar cuáles serían los valores óptimos en el perceptrón multicapa.
5) Probar técnicas vistas en clase para optimizar el perceptrón multicapa: ```dropout``` y ```early stopping```.
6) Trabajo futuro: aplicar otras técnicas para mejorar las prestaciones, como la inicialización de los pesos (Xavier y He), otros optimizadores (e.g., Adam), el uso de ```schedulers``` para la tasa de aprendizaje, ```lasso regularization```, ```ridge regularization```, ```batch normalisation```, ```data augmentation```, etc.

Sobre el entrenamiento:
- Se ha establecido que la función de pérdidas sea el MSE (Mean Squared Error). El motivo de esta elección es que es el error que se suele emplear en problemas de regresión, tal y como se puede leer [aquí](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/).
- Las funciones de activación no lineales de las capas ocultas son ReLU. De acuerdo con lo leído [aquí](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/), se recomienda que en los perceptrones siempre se utilice ReLU, por lo que no se prueban otras.
- La función de activación de la capa de salida es la lineal. De acuerdo con lo leído [aquí](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/), no se puede emplear ninguna otra cuando se trata de un modelo de regresión.

Sobre la validación y la evaluación:
- Se ha establecido que para validar los parámetros y evaluar la red neuronal la métrica empleada sea el MAE (Mean Absolute Error) porque proporciona una medida más interpretable de los errores del modelo.

#### Análisis de los resultados

|                                   | **Embeddings**    | **TF-IDF**        | **Word2Vec**      |
|-----------------------------------|-------------------|-------------------|-------------------|
| **Red Neuronal**                  | **MAE (pruebas)** | **MAE (pruebas)** | **MAE (pruebas)** |
| Perceptrón simple + lr=0.01       | ~0.83             | ~0.86             | ~0.86             |
| Perceptrón multicapa + lr=0.01    | ~0.79             | ~0.86             | ~0.85             |
| MLP + lr=0.1                      | ~0.9              | ~0.9              | ~0.9              | 
| MLP + lr=0.01                     | ~0.79             | ~0.86             | ~0.85             |
| MLP + lr=0.001                    | ~0.84             | ~0.88             | ~0.88             |
| MLP + lr=0.01 + dropout           | ~0.89             | ~0.87             | ~0.89             |
| MLP + lr=0.01 + early stopping    | ~0.83             | ~0.85             | ~0.83             |

### Regresor k-NN
#### Hoja de ruta
1) Obtener los conjuntos de entrenamiento y de pruebas. Es importante mencionar que el conjunto de entrenamiento incluye el conjunto de validación porque en este apartado se utilizará validación cruzada. Además, las muestras asignadas a cada conjunto son las mismas que en redes neuronales ya que se les ha asignado el mismo ```random_state```.
2) El regresor k-NN cuenta con [varios hiperparámetros](https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsRegressor.html). En este proyecto se analiza el valor óptimo del número de vecinos (```n_neighbors```) en un rango de 1 a 100. El motivo de elegir un rango amplio de vecinos se debe a que el número de muestras utilizadas para entrenar el regresor k-NN es elevado. El resto de los hiperparámetros mantienen sus valores por defecto.
3) El valor óptimo de ```n_neighbors``` se va a determinar mediante validación cruzada utilizando el conjunto de entrenamiento. En concreto, se aplica ```10-Fold cv```, tal y como se realizó en la práctica de regresión de la asignatura. Además, la métrica para determinar el valor óptimo de ```n_neighbors``` es el MAE (Mean Absolute Error). El motivo de su elección es que es una métrica que se suele emplear en problemas de regresión, además de ser la métrica que se ha ido analizando en la red neuronal.
4) Determinar las prestaciones de cada regresor k-NN hallando el MAE sobre el conjunto de pruebas.

#### Análisis de los resultados
Los resultados mostrados en la siguiente tabla indican, para regresor k-NN, su número óptimo de vecinos y el MAE obtenido sobre el conjunto de pruebas. Se observa que la vectorización que mejores prestaciones ofrece es embeddings, seguido de TF-IDF y Word2Vec. Además, en cuanto al número de vecinos, las vectorizaciones embeddings y Word2Vec necesitan un número menor de vecinos respecto a TF-IDF.

| **Vectorización**  | **Número de vecinos óptimo** | **MAE (pruebas)**     |
|--------------------|------------------------------|-----------------------|
| **Embeddings**     | **37**                       | **0.81**              |
| TF-IDF             | 42                           | 0.85                  |
| Word2Vec           | 37                           | 0.86                  | 

#### Elección del regresor k-NN
A la vista de los resultados, el modelo k-NN que mejores prestaciones tiene es el que utiliza la vectorización embeddings. Por un lado, es el que mejor valor de MAE de pruebas tiene. Por otro lado, es el que menor número de vecinos necesita.

### Fine-tuning de BERT
Completar...

#### Análisis de los resultados
Completar...

### Red Neuronal vs k-NN vs fine-tuning
Completar...

# Extensión
La extensión que se va a implementar es: ```uso de un summarizer preentrenado (utilizando pipelines de Hugging Face) para proporcionar un resumen de la característica 'directions'```.

En concreto, se va a probar la capacidad de resumir de tres modelos: [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)<sup>[1]</sup>, [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) y [DISLab/SummLlama3.2-3B](https://huggingface.co/DISLab/SummLlama3.2-3B)<sup>[2]</sup>. Para determinar cómo de buenos son los resúmenes obtenidos, lo ideal sería compararlos con los realizados por un ser humano. Sin embargo, para optimizar el tiempo de trabajo, los resúmenes obtenidos de los tres modelos se van a comparar con los resúmenes generados por ChatGPT-4o. Además, para obtener un valor cuantitativo de la calidad de los resúmenes obtenidos, se van a aplicar las [métricas ROUGE y BLEU](https://neptune.ai/blog/llm-evaluation-text-summarization). El motivo de su elección es el bajo coste computacional que conllevan.

<sup>[1]</sup> Se trata de un modelo con arquitectura encoder-decoder. En concreto, es el fine-tuning de [facebook/bart-large](https://huggingface.co/facebook/bart-large).

<sup>[2]</sup> Se trata de un modelo con arquitectura decoder-only. En concreto, es el fine-tuning de [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), que a su vez es el fine-tunning de [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).

## Ejemplo

```
INSTRUCCIONES:
Bring 6 quarts water to a boil in pot, then plunge 2 lobsters headfirst into water and cook, covered, 8 minutes
from time they enter water. Transfer with tongs to a shallow baking pan to cool. Return water to a boil and cook
remaining 2 lobsters in same manner. When lobsters are cool enough to handle, remove meat from tail and claws
and set aside. Cut tail shells and lobster bodies (not including claws) into 1-inch pieces with kitchen shears,
then rinse well, discarding gills, eye sacs, tomalley, any roe, and claw shells. Transfer to a 6- to 8-quart
heavy pot, then add wine, carrots, celery, fennel, onion, garlic, large tarragon sprigs, salt, fennel seeds,
red-pepper flakes, and remaining 2 quarts water and bring to a boil. Reduce heat and simmer, uncovered, until
liquid is reduced to about 6 cups, about 1 1/2 hours. While stock reduces, scrape any coagulated white albumin
from lobster meat with a knife and cut meat into 1/2-inch pieces, then chill, covered. Pour stock through a
dampened cheesecloth-lined large sieve into a large bowl, pressing on and then discarding solids. Transfer 2 3/4
cups stock to a bowl. (Cool remaining stock completely, uncovered, then freeze in an airtight container for
another use.) Sprinkle gelatin evenly over 1/4 cup stock in a 1-quart saucepan, then let stand 1 minute to soften.
Heat over moderately low heat, stirring, just until gelatin is dissolved, then stir in vinegar and remaining 2 1/2
cups stock. Put molds in a baking pan. Add 2 teaspoons gelatin mixture to each mold and freeze until set, about 10
minutes. Put 1 small sprig of tarragon and a tip of claw meat in bottom of each mold, then divide lobster meat
among molds. Fill with remaining gelatin mixture and chill, covered with plastic wrap, until set, at least 2 hours.
To unmold, dip 1 mold in a pan of hot water 3 to 5 seconds to loosen. Run a thin knife around edge of mold and
invert gelée out onto a plate. Repeat with remaining molds. Drizzle plates with fresh tarragon oil .


RESUMEN GENERADO POR ChatGPT-4o:
Boil 6 quarts of water in a pot and cook 2 lobsters for 8 minutes, then transfer to a pan to cool. Repeat with
the remaining lobsters. Once cooled, remove the lobster meat from the tails and claws, set aside, and discard
shells and unwanted parts. Cut the remaining shells into pieces and rinse well. In a large pot, combine the shell
pieces with wine, vegetables, herbs, and seasonings, and simmer for about 1.5 hours until the liquid reduces.
Strain the stock through cheesecloth, reserving 2 3/4 cups for the gelée. Sprinkle gelatin over 1/4 cup of stock
and dissolve it over low heat, then add vinegar and the remaining stock. Pour the gelatin mixture into molds and
freeze for 10 minutes. Add a sprig of tarragon and claw meat to each mold, then divide the lobster meat and fill
with the gelatin mixture. Chill until set, about 2 hours. To unmold, dip molds in hot water briefly, run a knife
around the edge, and invert onto plates. Drizzle with fresh tarragon oil.
    

RESUMEN GENERADO POR meta-llama/Llama-3.2-3B-Instruct:
This document provides a recipe for tempato lobster gelée, a French-inspired dish made from lobster, cream, and
vegetables. Here's a step-by-step summary:

1. Bring 6 quarts of water to a boil, then cook 2 lobsters for 8 minutes. Let them cool, then remove the meat.
2. Cut the lobster meat into small pieces and rinse them clean.
3. In a large pot, combine the lobster meat, wine, vegetables, and stock, and simmer until the liquid is reduced
to 6 cups.
4. Strain the stock and chill it. Refrigerate or freeze it for later use.
5. Mix together 1/4 cup of stock with gelatin to soften it. Heat it until the gelatin is dissolved, then stir in
vinegar and the remaining stock.
6. Divide the gelatin mixture among individual molds and pour it into each mold, followed by lobster meat and a
sprig of tarragon.
7. Chill the mixture in the mold until set, at least 2 hours.
8. To unmold, dip the mold in hot water, then run a knife around the edge and invert the gelée onto a plate.
9. Serve the gelée with fresh tarragon oil.

This dish is designed to be elegant and flavorful, with a light and creamy texture from the gelatin and a delicate
flavor from the lobster and tarragon.


RESUMEN GENERADO POR DISLab/SummLlama3.2-3B:
This recipe involves cooking lobsters and using the resulting broth to make a lobster gelée, which is then set in
molds and served with a garnish of fresh tarragon oil. The instructions outline the process of boiling lobsters,
cooling and setting them, and then using the lobster meat to make a stock that is reduced and clarified with gelatin,
vinegar, and other ingredients. The stock is then used to set the gelée in molds, which are garnished with a sprig
of tarragon and a tip of claw meat. The dish is finished with a drizzle of fresh tarragon oil.


RESUMEN GENERADO POR facebook/bart-large-cnn:
Lobster meat can be frozen for up to two months. gelatin, vinegar and tarragon can be added to the lobster meat. The
gelatin mixture can be used to make a glaze for the lobster. The glaze can also be used as a base for other lobster
dishes.
```

## ROUGE y BLEU

- ROUGE indica las coincidencias de tokens (o N-gramas) entre dos textos: el de referencia (resúmenes generados por ChatGPT-4o) y el de hipótesis (resúmenes generados por los otros tres modelos).

- Existen múltiples variantes de ROUGE. En este proyecto se ha utilizado ROUGE-1. Los pasos para calcular la métrica son: 1) tokenizar los resúmenes, 2) calcular cuántos tokens tienen en común la referencia y la hipótesis, 3) calcular ```precision```, ```recall``` y ```F1```.

<br>

$$\text{precision} = \frac{\text{Número de tokens solapados}}{\text{Número total de tokens en el resumen de hipótesis}}$$

<br>

$$\text{recall} = \frac{\text{Número de tokens solapados}}{\text{Número total de tokens en el resumen de referencia}}$$

<br>

$$\text{F1 score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

<br>

- En cuanto a BLEU, los pasos para calcular la métrica son: 1) tokenizar los resúmenes, 2) calcular los tokens, bigramas y trigramas que tienen en común la referencia y la hipótesis, 3) calcular el valor de ```precision``` para los tokens, bigramas y trigramas, 4) calcular una penalización basada en la longitud de los resúmenes de referencia y de hipótesis, 4) calcular BLEU como el producto de los tres valores de ```precision``` y la penalización.

- Tanto ROUGE como BLEU tienen un rango de valores de 0 a 1, donde 0 significa que el resumen de la hipótesis es de baja calidad y 1 indica que es de alta calidad.

## Análisis de los resultados
Los resultados obtenidos se resumen en la siguiente tabla:

| Modelo                           | Recetas resumidas            | ROUGE-1 precision  | ROUGE-1 recall | ROUGE-1 F1   | BLEU    |
|----------------------------------|------------------------------|--------------------|----------------|--------------|---------|
| meta-llama/Llama-3.2-3B-Instruct | 46                           | 0.4646             | 0.4556         | 0.4540       | 0.1874  |
| DISLab/SummLlama3.2-3B           | 46                           | 0.4395             | 0.3181         | 0.3616       | 0.1067  |
| facebook/bart-large-cnn          | 46                           | 0.6048             | 0.3161         | 0.4077       | 0.0944  |

<br>

Estos valores muestran que ```meta-llama/Llama-3.2-3B-Instruct``` es el modelo que mejores prestaciones tiene, seguido de ```DISLab/SummLlama3.2-3B``` y, finalmente, ```facebook/bart-large-cnn```. Las razones por las que esto ocurre son:

- Los dos primeros modelos cuenta con 3.21B de parámetros respecto a los 406M del último. Aunque el número de parámetros no asegura automáticamente mejores resultados, generalmente sí contribuye a un mejor desempeño.

- Entre los dos primeros modelos, el segundo debería obtener mejores prestaciones ya que ha sido especializado en resumir textos. Sin embargo, los valores de las métricas dicen lo contrario.

- Que los resultados no sean los esperados puede deberse a:

  1.   Bajo número de pruebas. Solamente se han probado 46 recetas. Se deberían probar más para poder afirmar qué modelo es mejor.
  2.   Las métricas utilizadas. Debido a su bajo coste computacional, ROUGE y BLEU tienen debilidades. Ambas métricas se basan en la comparación de n-gramas y, por lo tanto, no capturan el significado del texto generado. Además, no son sensibles a la fluidez, la gramática o la creatividad del lenguaje, lo que puede llevar a puntuaciones que no coinciden con las evaluaciones humanas.

- Si se confía en las pruebas realizadas por los autores de la versión con fine-tuning, esta mejora las prestaciones del modelo base, tal y como se puede ver en su [ficha](https://huggingface.co/DISLab/SummLlama3.2-3B#human-evaluation) de Hugging Face.

- Trabajo futuro: probar más muestras y probar otras métricas que solucionan los problemas de ROUGE y BLEU a cambio de tener un mayor coste computacional (e.g., G-Eval).
