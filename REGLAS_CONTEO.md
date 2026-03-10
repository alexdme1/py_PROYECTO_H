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

### REGLA-B01 · Ubicación de items en balda
- **Cuándo:** por cada detección de clase `Flores`, `Planta` o `tallo_grupo`, tanto en frontal como en trasera
- **Condición:** al menos el 60% de la bbox del item debe estar dentro del área detectada de una balda
- **Decisión:** asignar el item a esa balda; si no cumple el 60% en ninguna balda, notificar por terminal que el item queda sin asignar
- **Razón:** asignar absolutamente todos los items detectados a su balda correspondiente; no descartar ninguno silenciosamente

---

### REGLA-B02 · Mapeo en espejo (flip X)
- **Cuándo:** al cruzar tallos con masas de la misma balda (traseros↔frontales y frontales↔traseros)
- **Condición:** existe balda en la vista contraria con el mismo índice (debe ocurrir siempre)
- **Decisión:** invertir la posición X del tallo dentro de su balda y proyectarla al espacio X de la balda en la vista contraria
- **Razón:** las cámaras frontal y trasera ven el carro en espejo horizontal entre sí

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
  - ✅ Masas + ✅ Tallos → asignar cada tallo a la masa más cercana dentro de la misma balda (distancia euclidiana); todos los tallos deben quedar asignados
  - ✅ Masas + ❌ Tallos → aplicar REGLA-B05
  - ❌ Masas + ✅ Tallos → notificar por terminal: tallos sin masa asignable en esta balda
  - ❌ Masas + ❌ Tallos → balda vacía, no hacer nada
- **Razón:** comprobar el estado antes de repartir evita lógica redundante y garantiza que ningún tallo quede sin asignar silenciosamente

---

### REGLA-B05 · Masa sin tallos
- **Cuándo:** al finalizar el reparto de REGLA-B04
- **Condición:** `tallos_asociados == 0` — no había tallos en la balda desde la vista contraria
- **Decisión:** asignar `unidades_finales = 1` automáticamente
- **Razón:** especialmente para flores, si no se detecta tallo desde la vista contraria se asume falta de visibilidad; 1 unidad es el mínimo conservador

---

### REGLA-B06 · Clasificación por especie con ConvNeXt
- **Cuándo:** hay clasificador disponible y la masa tiene un crop válido
- **Condición:** `img_frontal is not None` y `clasificador is not None` y `crop.size > 0`
- **Decisión:** rotar crop 90° CCW, inferir con ConvNeXt, usar `producto_id` = clase predicha
- **Razón:** la imagen fue capturada en orientación rotada; la corrección es necesaria para que el clasificador funcione correctamente

---

### REGLA-B07 · Sin clasificador disponible
- **Cuándo:** ConvNeXt no está cargado o la imagen frontal no se proporcionó
- **Condición:** `clasificador is None` o `img_frontal is None`
- **Decisión:** usar `producto_id = "{Clase}_{contador}"` (ej. `Flores_1`, `Planta_2`)
- **Razón:** el conteo sigue siendo válido aunque sin identificación de especie

---

### REGLA-B08 · Agrupación por producto_id
- **Cuándo:** al construir el JSON final de una balda
- **Condición:** dos o más masas en la misma balda tienen el mismo `producto_id`
- **Decisión:** sumar sus `detecciones`, `tallos_totales` y `unidades_totales`; promediar sus confianzas
- **Razón:** el mismo producto puede aparecer en múltiples detecciones separadas dentro de la misma balda
