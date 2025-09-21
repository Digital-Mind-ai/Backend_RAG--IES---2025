# run.py 
import os
import sys
# Importaciones necesarias (añade las que uses en run.py)
from uvicorn.main import main 
from uvicorn import run as uvicorn_run 

# Si tienes otras importaciones globales o llamadas a funciones, ponlas aquí

if __name__ == '__main__':
    # 1. Fuerza el cambio de directorio temporal al directorio actual (Lógica de TEMP)
    # Se ejecuta solo una vez en el proceso padre.
    new_temp_dir = os.path.join(os.path.dirname(__file__), "TEMP_RAG")
    if not os.path.exists(new_temp_dir):
        os.makedirs(new_temp_dir)
        
    os.environ['TEMP'] = new_temp_dir
    os.environ['TMP'] = new_temp_dir
    os.environ['TMPDIR'] = new_temp_dir
    
    # 2. Llama a Uvicorn
    # Usamos la función run directamente para evitar conflictos con sys.argv
    uvicorn_run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Habilita el reloader
    )