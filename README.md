# Proyecto_TD_28_34
Proyecto de la asignatura Tratamiento de Datos realizado por 100451528 y 100451534.

## Descripción del problema y objetivos del proyecto

## Metodología aplicada
### Preprocesado de los datos
El primer paso es realizar un preprocesado de los datos. Para ello, se va a emplear la librería SpaCy. Además, como se ha comentado en el apartado anterior, se van a utilizar las características 'date', 'categories' y 'directions'. Sin embargo, cada una de ellas cuenta con un preprocesado diferente.

- La característica 'directions' es una lista de strings. Entonces, para cada string se va a realizar la tokenización, la homogeneización y la limpieza.
- La característica 'categories' es una lista de strings. Sin embargo, en lugar de realizar el mismo preprocesado que en 'directions', solamente se realiza la conversión a minúsculas. El motivo de no realizar un preprocesado mayor es que las palabras que forman cada string ya constituyen una entidad, por lo que es conveniente mantenerlas juntas y que no pierdan información.
- El preprocesado de la característica 'date' consiste en extraer el mes y el día en que se publicó la receta.

<div style="border: 2px solid #000; padding: 10px; background-color: #f9f9f9;">
  # Date: 
  ** Contenido: 2004-08-20 04:00:00+00:00
  ** Contenido preprocesado: ['8', '20']
  
  # Categories: 
  ** Contenido: ['Soup/Stew', 'Beef', 'Tomato', 'Celery', 'Fall', 'Simmer', 'Gourmet']
  ** Contenido preprocesado: ['soup/stew', 'beef', 'tomato', 'celery', 'fall', 'simmer', 'gourmet']
  
  # Directions: 
  ** Contenido: ['Whisk egg whites in a large bowl until foamy and add eggshells. Separately pulse tomatoes and celery in a food processor until coarsely chopped, then add to egg whites. Whisk in beef, salt, and peppercorns.', ...']
  ** Contenido preprocesado: ['whisk', 'egg', 'white', 'large', 'bowl', 'foamy', 'add', 'eggshell', 'separately', 'pulse', 'tomato', 'celery', 'food', 'processor', 'coarsely', 'chop', 'add', 'egg', 'white', 'whisk', 'beef', 'salt', 'peppercorn', ...]
</div>


### Vectorización de los datos

### Preparación de los datos para los modelos

### Redes neuronales

### Regresor k-NN

### Fine-tuning de BERT

## Análisis de los resultados

# Extensión

## Análisis de los resultados
