import os
from typing import List, Dict, Any
from fastapi import UploadFile
import mimetypes


class FileProcessor:
    """Servicio para procesar archivos subidos"""
    
    ALLOWED_EXTENSIONS = {
        '.txt', '.pdf', '.doc', '.docx', 
        '.jpg', '.jpeg', '.png', '.gif'
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
        """Procesa un archivo individual"""
        # Validar archivo
        validation = FileProcessor.validate_file(file)
        if not validation["valid"]:
            return {
                "success": False,
                "filename": file.filename,
                "errors": validation["errors"]
            }
        
        try:
            # Leer contenido del archivo
            content = await file.read()
            
            # Información del archivo procesado
            file_info = {
                "success": True,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "conv_id": conv_id,
                "processed_at": "now"  # Puedes usar datetime.now() aquí
            }
            
            # Aquí puedes agregar lógica específica según el tipo de archivo
            if file.content_type and file.content_type.startswith('text/'):
                # Para archivos de texto, puedes extraer y procesar el contenido
                try:
                    text_content = content.decode('utf-8')
                    file_info["text_preview"] = text_content[:200] + "..." if len(text_content) > 200 else text_content
                except UnicodeDecodeError:
                    file_info["text_preview"] = "No se pudo decodificar el contenido como texto"
            
            elif file.content_type and file.content_type.startswith('image/'):
                # Para imágenes, puedes agregar metadatos
                file_info["is_image"] = True
            
            # TODO: Aquí puedes agregar lógica para:
            # - Guardar el archivo en disco o cloud storage
            # - Procesar con IA si es necesario
            # - Guardar información en base de datos
            
            return file_info
            
        except Exception as e:
            return {
                "success": False,
                "filename": file.filename,
                "errors": [f"Error procesando archivo: {str(e)}"]
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
    """Servicio principal para subir archivos"""
    
    ## LOGICA DE SUBIR ARCHIVOS al RAG - VECTORIAL
    return await FileProcessor.process_multiple_files(files, conv_id)


def get_allowed_file_types() -> List[str]:
    """Retorna los tipos de archivo permitidos"""
    return list(FileProcessor.ALLOWED_EXTENSIONS)


def get_max_file_size() -> int:
    """Retorna el tamaño máximo permitido en bytes"""
    return FileProcessor.MAX_FILE_SIZE