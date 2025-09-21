# services/file_serv.py (ADAPTADO para Indexación RAG)
import os
from typing import List, Dict, Any
from fastapi import UploadFile
import mimetypes
from datetime import datetime 
import uuid
# Importar la función de indexación que creamos en services/agent.py
# (Asegúrate de que services/agent.py tiene la función ingest_file_to_es)
from services.agent import ingest_file_to_es 


class FileProcessor:
    """Servicio para procesar archivos subidos e indexarlos para RAG"""
    
    ALLOWED_EXTENSIONS = {
        '.txt', '.pdf', '.doc', '.docx', '.md' 
        # NOTA: Eliminé .jpg, .jpeg, .png, .gif porque LlamaIndex SimpleDirectoryReader 
        # estándar no los procesa fácilmente sin herramientas adicionales.
    }
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def validate_file(file: UploadFile) -> Dict[str, Any]:
        """Valida si el archivo es permitido"""
        result = {
            "valid": True,
            "errors": []
        }
        
        # Verificar extensión
        if file.filename:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in FileProcessor.ALLOWED_EXTENSIONS:
                result["valid"] = False
                result["errors"].append(f"Extensión no permitida: {file_ext}")
        
        # Verificar tamaño (si está disponible)
        if hasattr(file, 'size') and file.size and file.size > FileProcessor.MAX_FILE_SIZE:
            result["valid"] = False
            result["errors"].append(f"Archivo muy grande: {file.size} bytes (máximo: {FileProcessor.MAX_FILE_SIZE})")
        
        return result
    
    @staticmethod
    async def process_file(file: UploadFile, conv_id: str) -> Dict[str, Any]:
        """Procesa un archivo individual y lo indexa en Elasticsearch."""
        # Validar archivo
        validation = FileProcessor.validate_file(file)
        if not validation["valid"]:
            return {
                "success": False,
                "filename": file.filename,
                "errors": validation["errors"]
            }
        
        try:
            # 1. Leer contenido del archivo
            # Usamos read() una sola vez, ya que después el archivo se cierra.
            content = await file.read() 
            if not content or len(content) == 0:
                return {
                    "success": False,
                    "filename": file.filename or "N/A",
                    "errors": ["El archivo está vacío o no se adjuntó correctamente en la petición."]
                }
            file_name = file.filename or f"unknown_{uuid.uuid4().hex[:8]}" 
            indexing_success = ingest_file_to_es(content, file_name)
            # 2. INGESTAR EN ELASTICSEARCH (Llamada a la lógica RAG)
            indexing_success = ingest_file_to_es(content, file.filename)
            
            if not indexing_success:
                 return {
                    "success": False,
                    "filename": file.filename,
                    "errors": ["Error al indexar el documento en Elasticsearch."]
                 }

            # 3. Información del archivo procesado
            file_info = {
                "success": True,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "conv_id": conv_id,
                "processed_at": datetime.now().isoformat()
            }
            
            # TODO: Considera guardar metadatos del archivo en la DB Peewee aquí.
            
            return file_info
            
        except Exception as e:
            # Capturar errores generales (ej. disco lleno, problema de conexión)
            return {
                "success": False,
                "filename": file.filename,
                "errors": [f"Error general procesando/indexando archivo: {str(e)}"]
            }
    
    @staticmethod
    async def process_multiple_files(files: List[UploadFile], conv_id: str) -> Dict[str, Any]:
        """Procesa múltiples archivos"""
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "files": []
        }
        
        # NOTA: En un entorno de producción, es mejor usar asyncio.gather 
        # para procesar archivos en paralelo si el entorno lo soporta.
        for file in files:
            file_result = await FileProcessor.process_file(file, conv_id)
            results["files"].append(file_result)
            
            if file_result["success"]:
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        return results


# Funciones de conveniencia para usar en los controladores
async def upload_files_serv(files: List[UploadFile], conv_id: str) -> Dict[str, Any]:
    """Servicio principal para subir archivos al RAG"""
    return await FileProcessor.process_multiple_files(files, conv_id)


def get_allowed_file_types() -> List[str]:
    """Retorna los tipos de archivo permitidos"""
    return list(FileProcessor.ALLOWED_EXTENSIONS)


def get_max_file_size() -> int:
    """Retorna el tamaño máximo permitido en bytes"""
    return FileProcessor.MAX_FILE_SIZE