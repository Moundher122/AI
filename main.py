import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import face_recognition
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import numpy as np
app = FastAPI()
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZpXkVOq3lk7TZM5IKng3"
)
@app.post("/face_recognition")
async def compare_faces(file: UploadFile = File(...)):
    contents = await file.read()
    unknown_image = face_recognition.load_image_file(BytesIO(contents))
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    if results[0]==np.True_:
        return {"match": True}
    else:
        return {"match": False}
@app.post("/emotion_recognition")
async def emotion_recognition(file: UploadFile = File(...)):
    contents = await file.read()
    image = face_recognition.load_image_file(BytesIO(contents))
    result = client.run_workflow(
    workspace_name="hackathon-3cwzk",
    workflow_id="detect-and-classify",
    images={
        "image": image,
    },
    )
    anger = result[0]["output_image"][0]["predictions"]["predictions"]["anger"]["confidence"]
    fear = result[0]["output_image"][0]["predictions"]["predictions"]["fear"]["confidence"]
    return {"anger": anger, "fear": fear}


def face_recognition_task(image_bytes: bytes) -> Dict[str, bool]:
    unknown_image = face_recognition.load_image_file(BytesIO(image_bytes))
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    result = face_recognition.compare_faces([known_encoding], unknown_encoding)
    # Convert numpy bool to Python bool
    return {"match": bool(result[0])}


def emotion_recognition_task(image_bytes: bytes) -> Dict[str, float]:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    result = client.run_workflow(
        workspace_name="hackathon-3cwzk",
        workflow_id="detect-and-classify",
        images={"image": base64_image},
    )
    
    try:
        # Handle the actual response structure (list of dictionaries)
        if isinstance(result, list) and len(result) > 0:
            if 'output_image' in result[0] and result[0]['output_image']:
                output_image = result[0]['output_image']
                if isinstance(output_image, list) and len(output_image) > 0:
                    if 'predictions' in output_image[0]:
                        predictions = output_image[0]['predictions']
                        if 'predictions' in predictions:
                            emotions = predictions['predictions']
                            anger = emotions.get('anger', {}).get('confidence', 0.0)
                            fear = emotions.get('fear', {}).get('confidence', 0.0)
                            return {"anger": anger, "fear": fear}
        
        print(f"Unexpected API response structure: {result}")
        return {"anger": 0.0, "fear": 0.0}
            
    except Exception as e:
        print(f"Error processing emotion recognition: {str(e)}")
        print(f"API response: {result}")
        return {"anger": 0.0, "fear": 0.0}

# ----------------------------------------------------
# Route 3: Combined Endpoint (Face + Emotion in Parallel)
# ----------------------------------------------------
@app.post("/analyze_face_and_emotion")
async def analyze_face_and_emotion(file: UploadFile = File(...)):
    contents = await file.read()

    # Using ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        face_future = executor.submit(face_recognition_task, contents)
        emotion_future = executor.submit(emotion_recognition_task, contents)

        face_result = face_future.result()
        emotion_result = emotion_future.result()

    return {
        "face_recognition": face_result,
        "emotion_recognition": emotion_result
    }


# ----------------------------------------------------
# Route 4: Test Endpoint to Examine API Response
# ----------------------------------------------------
@app.post("/test_emotion_api")
async def test_emotion_api(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    result = client.run_workflow(
        workspace_name="hackathon-3cwzk",
        workflow_id="detect-and-classify",
        images={"image": base64_image},
    )
    
    # Convert any non-serializable parts to their string representation
    def make_serializable(obj):
        if isinstance(obj, (dict, list)):
            return obj
        else:
            return str(obj)
    
    # Process the result to make it JSON serializable
    serializable_result = {}
    for k, v in result.items():
        if isinstance(v, dict):
            serializable_result[k] = {k2: make_serializable(v2) for k2, v2 in v.items()}
        else:
            serializable_result[k] = make_serializable(v)
    
    return {"raw_response": serializable_result}
class AuthRequest(BaseModel):
    expected_image_base64: str
    auth_image_base64: str

@app.post("/auth")
async def authenticate_user(payload: AuthRequest):
    try:
        expected_bytes = base64.b64decode(payload.expected_image_base64)
        auth_bytes = base64.b64decode(payload.auth_image_base64)

        expected_image = face_recognition.load_image_file(BytesIO(expected_bytes))
        auth_image = face_recognition.load_image_file(BytesIO(auth_bytes))

        expected_encoding = face_recognition.face_encodings(expected_image)[0]
        auth_encoding = face_recognition.face_encodings(auth_image)[0]
        results = face_recognition.compare_faces([expected_encoding], auth_encoding)
        # Convert numpy bool to Python bool
        if results[0] == np.True_:
            result = True
        else:
            result = False    
        if result:
            return {"match": True}
        else:
            return {"match": False}
    except Exception as e:
        print(f"Error during authentication: {str(e)}")
        return {"error": str(e)}
