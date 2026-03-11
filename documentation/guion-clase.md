# Guion clase — Proyecto H (Túnel de Flores)

## 0. Quién soy y por qué estoy aquí (5’)

- Quién soy ahora  
  - Responsable de IA / datos en VerdNatura, trabajando en visión por computador aplicada a negocio real.
- De dónde vengo  
  - Mi formación, cómo acabé en IA, primeros proyectos “juguete” vs proyectos que tocan negocio.
- Qué vengo a contar hoy  
  - No vengo a hacer un tutorial de librerías, sino a enseñar un proyecto real de visión de principio a fin, con sus aciertos y sus chapuzas iniciales.

---

## 1. El problema de negocio y el ciclo del dato (10’)

- Qué es el túnel de flores  
  - Carros de plantas pasando por un túnel con cámaras frontal y trasera.
  - Necesidad: saber cuántas unidades reales hay por ticket y por balda, sin que una persona tenga que contarlas.
- Qué datos tengo realmente  
  - Imagen frontal: tickets, baldas, masas de flores y plantas.
  - Imagen trasera: baldas y grupos de tallos.
  - Condiciones reales: variación de iluminación, carros desalineados, baldas con huecos, etc.
- Cómo pienso esto como ciclo del dato  
  - Captura → etiquetado → limpieza/unificación → entrenamiento de modelos → reglas de negocio → JSON final → comparación con el pedido.
  - Desde el principio tengo claro que no es solo entrenar una red, es diseñar todo el flujo.

---

## 2. Primeros intentos: YOLO y por qué no fue suficiente (8–10’)

- Por qué empecé con YOLO  
  - Es lo más conocido: rápido, fácil de montar, buen “primer martillo” para detección.
  - Primer objetivo: detectar cosas en el carro (flores, plantas, baldas, tickets) con bounding boxes.
- Qué funcionaba bien  
  - Detección de objetos grandes y relativamente separados.
  - Experimentos rápidos, feedback visual rápido.
- Dónde se rompía el enfoque  
  - YOLO solo da cajas, no máscaras.
  - Necesitaba saber exactamente dónde acaba una balda y dónde empieza otra, y qué píxeles son tallos, no solo “un rectángulo por ahí”.
  - Era muy difícil escribir reglas de negocio finas (asignar ticket a balda, separar masas solapadas) solo con cajas.
- Decisión  
  - YOLO me sirvió para orientarme, pero para este problema concreto necesitaba segmentación de instancias → salto a Mask R‑CNN.

---

## 3. Cómo capturo los datos en el túnel (5’)

- Setup físico  
  - Dos cámaras: frontal y trasera, disparan sincronizadas cuando el carro entra en el túnel.
  - Disparo con trigger (sensor/laser) para que las dos imágenes correspondan al mismo instante.
- Qué asumo y qué no  
  - Los carros no siempre están perfectos, hay pequeños desplazamientos.
  - Siempre hay la misma estructura general: baldas apiladas verticalmente, tickets en zona frontal.
- Decisiones aquí  
  - Prefiero asegurar sincronía y buena calidad de imagen antes de complicar el modelo.
  - Lo que no resuelva aquí me lo voy a comer luego en el pipeline.

---

## 4. Etiquetado y Roboflow: donde empieza lo serio (10’)

- Por qué uso Roboflow  
  - Me resuelve gran parte de la gestión del dataset: etiquetado, versiones, splits train/val/test, export a COCO/ImageFolder.
- Cómo etiqueto  
  - Defino 5 clases claras: `Flores`, `Planta`, `Balda`, `ticket`, `tallo_grupo`.
  - Decisión importante: unificar cosas tipo `Balda1/Balda2/Balda3` → una sola clase `Balda`, y luego manejo la posición con coordenadas, no con nombres.
  - Criterios de etiquetado: qué entra como “Flores” y qué entra como “Planta”, cómo dibujo los polígonos, cómo trato los casos dudosos.
- Qué hace Roboflow por mí  
  - Preprocesado básico: resize, formato, algún augmentation ligero.
  - Mantener versiones de dataset según voy cambiando criterios de etiquetado.
- Dónde pongo el límite Roboflow vs código  
  - Roboflow: cosas estáticas, “bakeadas” en los ficheros.
  - Código: augmentations de iluminación más agresivas, rotaciones específicas, lógica de rotar 90º para trabajar siempre en retrato, unificar categorías, etc.

---

## 5. Limpieza y unificación del dataset (fix_coco, crops…) (8–10’)

- Problema típico: exportar no es el final  
  - Exporto COCO desde Roboflow, pero no viene perfecto para mi caso: nombres de clases, orientación, etc.
- Script de unificación (`01_fix_coco.py`)  
  - Roto todas las imágenes 90º si hace falta para que frontal/trasera sigan el mismo criterio.
  - Unifico nombres de categorías (`Balda1/Balda2/Balda3` → `Balda`).
  - Genero tres ficheros COCO limpios: `train.json`, `valid.json`, `test.json` en `data/coco_unified/`.
- Preparar el dataset de clasificación  
  - A partir de las máscaras de Mask R‑CNN, recorto crops de `Flores` y `Planta` (`00_cropping.py`).
  - Eso se convierte en un `ImageFolder`: carpetas = `producto_id`.
  - Aquí se ve bien que un modelo alimenta al otro: Mask R‑CNN genera los cortes con los que entreno ConvNeXt.

---

## 6. Mask R‑CNN en mi proyecto: qué hace y qué entra/sale (10’)

- Qué versión uso  
  - `mask_rcnn_R_50_FPN_3x` preentrenado en COCO, con ResNet‑50 + FPN.
- Qué le doy de entrada  
  - Las imágenes frontales y traseras en la resolución que he decidido (vertical, hasta ~1080×2000).
  - Configuración de anchors adaptada a mi caso:
    - Tamaños pequeños para tickets y tallos, grandes para baldas.
    - Ratios muy horizontales (baldas) y muy verticales (tallos).
- Qué me devuelve  
  - Para cada imagen: lista de detecciones con clase, `bbox`, máscara y `score`.
  - Este es mi “lenguaje común”: una detección es `{class, bbox, mask, score}`.
- Cómo hago el fine‑tuning aquí  
  - No entreno desde cero: heredo pesos COCO.
  - Cambio `NUM_CLASSES` a mis 5 clases.
  - Ajusto resolución, anchors, sampler para clases raras (tickets, tallos).
  - Entreno con LR bajo, warmup y bastante data augmentation de iluminación para simular el túnel.
- Lo importante para la charla  
  - Mask R‑CNN es mi traductor de píxeles a objetos: convierte la imagen cruda en “aquí hay una balda”, “aquí hay un ticket”, etc.
  - A partir de ahí todo es lógica geométrica y reglas.

---

## 7. ConvNeXt Tiny: clasificar especies a partir de los crops (8–10’)

- Por qué un segundo modelo  
  - Mask R‑CNN solo distingue `Flores` y `Planta` como grandes grupos.
  - El negocio necesita `producto_id`: especie concreta.
- Qué hago con los crops  
  - Uso los recortes generados antes para montar un `ImageFolder` de entrenamiento (`train/valid/test`).
  - Cada carpeta es una especie concreta.
- ConvNeXt Tiny y fine‑tuning  
  - Cargo `convnext_tiny.in12k_ft_in1k` preentrenado.
  - Reemplazo la cabeza final por una capa con mis clases actuales (nº de especies).
  - Configuro:
    - `PRETRAINED = True`.
    - `FREEZE_BACKBONE = False` → fine‑tuning completo.
    - LR bajo, AdamW, cosine annealing con warmup, label smoothing.
- Qué entra y qué sale  
  - Entra: crop 224×224, ya normalizado como espera ImageNet.
  - Sale: vector de probabilidades por especie.
  - Me quedo con el `top‑1` y, si hace falta, `top‑5` para análisis.
- Mensaje que quiero darles  
  - Aquí el patrón es muy estándar: modelo preentrenado + dataset propio + fine‑tuning cuidadoso.
  - Lo importante es que el dataset de especies lo genero yo mismo a partir del detector: está totalmente integrado en el ciclo.

---

## 8. Lógica de negocio: de detecciones a conteo por ticket/balda (10–12’)

- Asignar tickets a baldas en la imagen frontal  
  - Ordeno baldas de arriba a abajo por coordenada Y.
  - Cada ticket se asigna a la balda donde cae su centro.
  - Regla de dominancia: un ticket controla su balda y todas las de debajo hasta que aparece otro ticket.
- Proyectar esa dominancia a la imagen trasera  
  - En la trasera detecto baldas y tallos.
  - Para cada ticket sé qué rango de Y ocupa en frontal → calculo el rango equivalente en trasera y defino “zona de captura” de tallos para ese ticket.
- Mapeo en espejo (flip X) por balda  
  - Dentro de cada balda, hago el espejo: izquierda frontal ↔ derecha trasera.
  - Con eso proyecto los centros de los tallos sobre la balda frontal correspondiente.
- Asignar tallos a masas de flores/plantas  
  - Fase 1: garantizo mínimo un tallo por masa (para evitar flores con 0 tallos).
  - Fase 2: reparto los tallos sobrantes a la masa más cercana (distancia euclídea).
- Regla final de unidades  
  - `unidades_finales = max(volumen_frontal, tallos_traseros)`.
  - Ejemplos:
    - 1 bulto frontal + 3 tallos → cuento 3.
    - 2 bultos + 5 tallos → reparto según proximidad.
- JSON de salida  
  - Estructura `Ticket → Balda → producto_id` con `detecciones`, `tallos` y `unidades_finales`.
  - Este JSON es lo que comparo contra el pedido real.

---

## 9. Cosas que todavía quiero mejorar (5’)

- Limitaciones actuales  
  - Tallos pegados que se detectan como un solo grupo.
  - Múltiples tickets en la misma balda.
  - Baldas que Mask R‑CNN no detecta bien.
- Ideas de mejora  
  - Mejor dataset de tallos (más anotaciones, quizá separar tallos individuales).
  - Algoritmos más robustos para baldas (fallback con detección clásica de bordes).
  - Tracking temporal si algún día hay más de un frame por carro.
- Mensaje para ellos  
  - Esto no es “proyecto perfecto”, es un proyecto vivo.
  - Lo interesante es ver cómo tomo decisiones, qué trade‑offs acepto y dónde pongo mi tiempo.

---

## 10. Parte práctica (30’) – ideas de dinámica

- Mini‑demostración Roboflow  
  - Crear un proyecto rápido.
  - Enseñar cómo etiquetar bien 3–4 imágenes y qué decisiones tomar al etiquetar.
  - Configurar un par de augmentations y enseñar el export COCO.
- Breve vistazo a Hugging Face  
  - Buscar un modelo de clasificación de imágenes.
  - Probar 1–2 imágenes del túnel para que vean qué devuelve un modelo genérico.
  - Comentar cómo leer una model card para tomar decisiones de arquitectura/preentrenamiento.
