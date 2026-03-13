"""
conteo_module.py — Wrapper para importar funciones de conteo.
Delega a conteo_lib.py con import directo.
"""
import os
import sys

# Asegurar que el directorio actual está en el path para import directo
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from conteo_lib import (
    ubicar_en_balda,
    asignar_tickets_a_baldas,
    procesar_pareja_imagenes,
    contar_articulos,
)

__all__ = [
    "ubicar_en_balda",
    "asignar_tickets_a_baldas",
    "procesar_pareja_imagenes",
    "contar_articulos",
]
