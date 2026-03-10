# 📖 Libro de Reglas de Negocio — Sistema de Conteo PoC.H

---

## Bloque A · `asignar_tickets_a_baldas()`

---

### REGLA-A01 · Ordenación de baldas
- **Cuándo:** siempre, al inicio de la función
- **Condición:** hay baldas detectadas en la imagen frontal
- **Decisión:** ordenar baldas por coordenada Y ascendente → Balda 1 = más baja físicamente, Balda 3 = más alta. El número de baldas debe ser exactamente 3; si se detectan más o menos, lanzar error por terminal y abortar
- **Razón:** garantiza que la herencia de dominancia baje siempre en orden físico coherente

---

### REGLA-A02 · Asignación de ticket a balda por centro Y
- **Cuándo:** por cada ticket detectado
- **Condición:** el centro Y del ticket cae dentro del rango Y de una balda
- **Decisión:** asignar ese ticket a esa balda
- **Razón:** el centro es el punto más representativo de la posición del ticket

---

### REGLA-A03 · Fallback por solapamiento
- **Cuándo:** el centro Y de un ticket no cae dentro de ninguna balda
- **Condición:** existe solapamiento parcial entre el ticket y alguna balda
- **Decisión:** asignar el ticket a la balda con mayor solapamiento Y
- **Razón:** el modelo puede detectar el ticket ligeramente desplazado; el overlap es mejor criterio que descartar

---

### REGLA-A04 · Herencia de dominancia hacia abajo
- **Cuándo:** al recorrer todas las baldas de arriba a abajo
- **Condición:** una balda no tiene ticket propio asignado
- **Decisión:** la balda hereda el ticket dominante más reciente (el del ticket más cercano por encima)
- **Razón:** un ticket físicamente representa las baldas que tiene debajo hasta el siguiente ticket

---

### REGLA-A05 · Ticket duplicado en misma balda
- **Cuándo:** dos tickets tienen su centro Y en la misma balda
- **Condición:** `mejor_balda` ya existe en `balda_a_ticket`
- **Decisión:** conservar el primero encontrado; notificar por terminal de fallo de Mask R-CNN
- **Razón:** esto no debería ocurrir nunca; se mantiene como mecanismo preventivo

---

### REGLA-A06 · Sin tickets detectados
- **Cuándo:** la imagen frontal no contiene ninguna detección de clase `ticket`
- **Condición:** lista de tickets vacía tras separar detecciones
- **Decisión:** devolver `{}`, abortar el pipeline y notificar por consola de fallo de Mask R-CNN
- **Razón:** esto no debería ocurrir nunca; se mantiene como mecanismo preventivo

---

### REGLA-A07 · Sin baldas detectadas
- **Cuándo:** la imagen frontal no contiene ninguna detección de clase `balda`
- **Condición:** lista de baldas vacía
- **Decisión:** devolver `{}`, abortar el pipeline y notificar por consola de fallo de Mask R-CNN
- **Razón:** sin baldas no hay estructura espacial posible; esto no debería ocurrir nunca, se mantiene como mecanismo preventivo

---

## Bloque B · `contar_articulos()`

---

### REGLA-B01 · Ubicación de items en balda (umbral 60%, fallback 40%)
- **Cuándo:** por cada detección de clase `Flores`, `Planta` o `tallo_grupo`, **tanto en frontal como en trasera**
- **Condición:**
  - ≥ 60% de la bbox del item dentro de una balda → asignar normalmente
  - ≥ 40% y < 60% → asignar con warning (item sobresale, pero mayoritariamente está dentro)
  - < 40% → no asignar, notificar por terminal
- **Decisión:** asignar el item a la balda con mayor ratio de solapamiento si cumple el umbral mínimo
- **Razón:** items como flores colgantes pueden sobresalir de la balda; el fallback del 40% evita pérdidas silenciosas manteniendo un filtro contra falsos positivos

---

### REGLA-B01b · Recolección bidireccional de masas
- **Cuándo:** antes del reparto de tallos
- **Condición:** existen detecciones de `Flores` o `Planta` **en la imagen trasera**
- **Decisión:** ubicar las masas traseras en baldas traseras, aplicar flip X para proyectar su centro al espacio frontal, e incorporarlas al pool de masas de la misma balda. Para la clasificación ConvNeXt, el crop se toma de la imagen trasera (no la frontal)
- **Razón:** las cámaras frontal y trasera capturan ángulos complementarios; flores visibles solo desde atrás (ej. Gerberas colgantes) no deben perderse

---

### REGLA-B02 · Mapeo en espejo (flip X)
- **Cuándo:** al cruzar tallos con masas de la misma balda (traseros↔frontales y frontales↔traseros)
- **Condición:** existe balda en la vista contraria con el mismo índice (debe ocurrir siempre)
- **Decisión:** invertir la posición X del tallo dentro de su balda y proyectarla al espacio X de la balda en la vista contraria
- **Razón:** las cámaras frontal y trasera ven el carro en espejo horizontal entre sí

---

### REGLA-B02b · Tallos frontales
- **Cuándo:** al recolectar tallos para el reparto
- **Condición:** existen detecciones de `tallo_grupo` **en la imagen frontal**
- **Decisión:** ubicarlos en baldas frontales; como ya están en espacio frontal, no necesitan flip X. Se combinan con los tallos traseros proyectados antes del reparto
- **Razón:** los tallos pueden ser visibles desde ambas vistas; no ignorar los detectados en la frontal

---

### REGLA-B03 · Balda sin espejo en vista contraria
- **Cuándo:** el índice de una balda no existe en la lista de baldas de la vista contraria
- **Condición:** `b_idx >= len(baldas_contraria)`
- **Decisión:** abortar y devolver fallo por consola
- **Razón:** esto no debería ocurrir nunca; se mantiene como mecanismo preventivo

---

### REGLA-B04 · Reparto de tallos
- **Cuándo:** antes de repartir, comprobar el estado de cada balda
- **Condición — 4 casos posibles:**
  - ✅ Masas + ✅ Tallos → asignar cada tallo a la masa más cercana dentro de la misma balda (distancia euclidiana en espacio frontal); todos los tallos deben quedar asignados
  - ✅ Masas + ❌ Tallos → aplicar REGLA-B05
  - ❌ Masas + ✅ Tallos → notificar por terminal: tallos sin masa asignable en esta balda
  - ❌ Masas + ❌ Tallos → balda vacía, no hacer nada
- **Nota:** los tallos combinados incluyen tanto los traseros (proyectados via flip X) como los frontales (ya en espacio frontal)
- **Razón:** comprobar el estado antes de repartir evita lógica redundante y garantiza que ningún tallo quede sin asignar silenciosamente

---

### REGLA-B05 · Masa sin tallos
- **Cuándo:** al finalizar el reparto de REGLA-B04
- **Condición:** `tallos_asociados == 0` — no había tallos en la balda desde ninguna vista
- **Decisión:** asignar `unidades_finales = 1` automáticamente
- **Razón:** especialmente para flores, si no se detecta tallo desde ninguna vista se asume falta de visibilidad; 1 unidad es el mínimo conservador

---

### REGLA-B06 · Clasificación por especie con ConvNeXt
- **Cuándo:** hay clasificador disponible y la masa tiene un crop válido
- **Condición:** imagen fuente disponible y `clasificador is not None` y `crop.size > 0`
- **Decisión:** rotar crop 90° CCW, inferir con ConvNeXt, usar `producto_id` = clase predicha. **La imagen fuente depende de la vista de la masa**: si `vista='frontal'` se cropea de `img_frontal`, si `vista='trasera'` se cropea de `img_trasera`
- **Razón:** la imagen fue capturada en orientación rotada; la corrección es necesaria para que el clasificador funcione correctamente. Usar la imagen correcta evita crops vacíos o desalineados

---

### REGLA-B07 · Sin clasificador disponible
- **Cuándo:** ConvNeXt no está cargado o la imagen fuente no se proporcionó
- **Condición:** `clasificador is None` o imagen fuente `is None`
- **Decisión:** usar `producto_id = "{Clase}_{contador}"` (ej. `Flores_1`, `Planta_2`)
- **Razón:** el conteo sigue siendo válido aunque sin identificación de especie

---

### REGLA-B08 · Agrupación por producto_id
- **Cuándo:** al construir el JSON final de una balda
- **Condición:** dos o más masas en la misma balda tienen el mismo `producto_id`
- **Decisión:** sumar sus `detecciones`, `tallos_totales` y `unidades_totales`; promediar sus confianzas
- **Razón:** el mismo producto puede aparecer en múltiples detecciones separadas dentro de la misma balda

---

### REGLA-B09 · Marca de confianza baja
- **Cuándo:** al calcular la confianza media de un producto agrupado
- **Condición:** `confianza_media < 0.5` (umbral configurable en `UMBRAL_CONFIANZA_BAJA`)
- **Decisión:** añadir campo `"confianza_baja": true` en el JSON del producto
- **Razón:** flores envueltas en plástico, crops parcialmente ocluidos o masas ambiguas pueden generar clasificaciones poco fiables; este flag permite al operario identificar y revisar manualmente esos casos
