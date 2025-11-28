from flask import Flask, request
import json
import base64
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

def run_command(command):
    """Ejecuta un comando del sistema y retorna su output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Comando excedió el tiempo límite"
    except Exception as e:
        return f"Error ejecutando comando: {str(e)}"

@app.route('/', methods=['POST'])
def handle_event():
    # Obtener el evento de Eventarc
    envelope = request.get_json()
    
    if not envelope:
        return 'No se recibió ningún evento', 400
    
    # Debug: Log raw payload
    logging.info(f"Raw Payload: {json.dumps(envelope)}")
    
    # Extraer información del evento
    event_type = request.headers.get('ce-type', 'Desconocido')
    event_id = request.headers.get('ce-id', 'Desconocido')
    source = request.headers.get('ce-source', 'Desconocido')
    
    payload = {}
    
    # Helper to find key recursively
    def find_key(data, target_key):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    return value
                if isinstance(value, (dict, list)):
                    found = find_key(value, target_key)
                    if found: return found
        elif isinstance(data, list):
            for item in data:
                found = find_key(item, target_key)
                if found: return found
        return None

    # Parsing Strategies
    
    # Case 1: Pub/Sub Push Subscription
    if 'message' in envelope and 'subscription' in envelope:
        try:
            # Check attributes first (common for GCS notifications)
            if 'attributes' in envelope['message']:
                payload.update(envelope['message']['attributes'])
            
            # Try to decode data
            if 'data' in envelope['message']:
                b64_data = envelope['message']['data']
                decoded_data = base64.b64decode(b64_data).decode('utf-8')
                try:
                    json_data = json.loads(decoded_data)
                    if isinstance(json_data, dict):
                        payload.update(json_data)
                except json.JSONDecodeError:
                    logging.warning("Pub/Sub data is not JSON")
                    pass
            
            logging.info("Processed Pub/Sub message")
        except Exception as e:
            logging.error(f"Failed to process Pub/Sub message: {e}")

    # Case 2: CloudEvent (Eventarc) with 'data' wrapper
    elif 'data' in envelope:
        payload = envelope['data']
    
    # Case 3: Flat payload
    else:
        payload = envelope

    # Extract File Info
    bucket_name = 'Desconocido'
    file_name = 'Desconocido'
    
    # Strategy A: Cloud Audit Log (specific structure)
    # Payload type: type.googleapis.com/google.events.cloud.audit.v1.LogEntryData
    if 'protoPayload' in envelope and 'resourceName' in envelope['protoPayload']:
        resource_name = envelope['protoPayload']['resourceName']
        # Format: projects/_/buckets/BUCKET_NAME/objects/OBJECT_NAME
        if 'objects/' in resource_name:
            try:
                # Split by 'objects/' and get the second part (the object name)
                file_name = resource_name.split('objects/')[-1]
            except Exception:
                pass
        
        if 'resource' in envelope and 'labels' in envelope['resource']:
             bucket_name = envelope['resource']['labels'].get('bucket_name', 'Desconocido')

    # Strategy B: Common fields (GCS/PubSub)
    if bucket_name == 'Desconocido':
        bucket_name = payload.get('bucket') or payload.get('bucketId') or 'Desconocido'
    
    if file_name == 'Desconocido':
        file_name = payload.get('name') or payload.get('objectId') or 'Desconocido'

    # Strategy C: Recursive Fallback
    if bucket_name == 'Desconocido':
        bucket_name = find_key(payload, 'bucket') or find_key(payload, 'bucketId') or find_key(payload, 'bucket_name') or \
                      find_key(envelope, 'bucket') or find_key(envelope, 'bucketId') or find_key(envelope, 'bucket_name') or 'Desconocido'
    
    if file_name == 'Desconocido':
        # Try to find 'name', 'objectId'
        file_name = find_key(payload, 'name') or find_key(payload, 'objectId') or \
                    find_key(envelope, 'name') or find_key(envelope, 'objectId')
        
        # If not found, look for 'resourceName' and parse it
        if not file_name:
            res_name = find_key(payload, 'resourceName') or find_key(envelope, 'resourceName')
            if res_name and 'objects/' in res_name:
                file_name = res_name.split('objects/')[-1]
            elif res_name:
                file_name = res_name
        
        if not file_name:
            file_name = 'Desconocido'

    content_type = find_key(payload, 'contentType') or find_key(payload, 'content-type') or 'Desconocido'
    size = find_key(payload, 'size') or find_key(payload, 'objectGeneration') or 0
    
    # Mostrar información
    logging.info('=' * 60)
    logging.info(f'📁 ARCHIVO CARGADO AL BUCKET')
    logging.info('=' * 60)
    logging.info(f'Bucket: {bucket_name}')
    logging.info(f'Archivo: {file_name}')
    logging.info(f'Tipo: {content_type}')
    logging.info(f'Tamaño: {size} bytes')
    logging.info(f'Event ID: {event_id}')
    logging.info(f'Event Type: {event_type}')
    logging.info('=' * 60)
    
    # Ejecutar comandos NVIDIA
    logging.info('\n🔧 INFORMACIÓN DEL SISTEMA')
    logging.info('=' * 60)
    
    logging.info('\n📌 NVCC Version:')
    nvcc_output = run_command('nvcc -V')
    logging.info(nvcc_output)
    
    logging.info('\n📌 NVIDIA-SMI:')
    nvidia_smi_output = run_command('nvidia-smi')
    logging.info(nvidia_smi_output)
    
    logging.info('=' * 60)
    
    # Debug: Log raw payload again at the end
    logging.info(f"DEBUG: Raw Payload at end: {json.dumps(envelope)}")
    
    # Respuesta
    response = {
        'message': 'Evento procesado exitosamente',
        'bucket': bucket_name,
        'file': file_name,
        'size': size,
        'event_id': event_id
    }
    
    return json.dumps(response), 200

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)