"""
conteo_module.py — Funciones de conteo reutilizables.
Wrapper limpio para importar desde la app sin ejecutar el __main__ de 05_conteo.py
"""
import os
import sys


def asignar_tickets_a_baldas(detecciones_frontales):
    """Importa y ejecuta desde 05_conteo."""
    from importlib import import_module
    mod = _get_conteo_module()
    return mod.asignar_tickets_a_baldas(detecciones_frontales)


def procesar_pareja_imagenes(det_frontal, det_trasera):
    mod = _get_conteo_module()
    return mod.procesar_pareja_imagenes(det_frontal, det_trasera)


def contar_articulos(det_frontal, det_trasera, asignacion_base,
                     img_frontal=None, img_trasera=None, clasificador=None):
    mod = _get_conteo_module()
    return mod.contar_articulos(det_frontal, det_trasera, asignacion_base,
                                img_frontal, img_trasera, clasificador)


_cached_module = None

def _get_conteo_module():
    global _cached_module
    if _cached_module is not None:
        return _cached_module
    
    import importlib.util
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_conteo.py")
    
    # Leer solo las funciones (sin ejecutar __main__)
    with open(script_path, 'r') as f:
        source = f.read()
    
    # Extraer solo hasta if __name__ == "__main__":
    marker = '\nif __name__ == "__main__":'
    idx = source.find(marker)
    if idx != -1:
        source = source[:idx]
    
    # Compilar y ejecutar solo las funciones
    import types
    mod = types.ModuleType("conteo_funcs")
    exec(compile(source, script_path, "exec"), mod.__dict__)
    
    _cached_module = mod
    return mod
