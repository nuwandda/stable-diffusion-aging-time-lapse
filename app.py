from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
from io import BytesIO
import base64
import traceback


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Aging Time Lapse API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Aging Time Lapse with FastAPI"}


@app.post("/api/aging/")
async def generate_image(imgPromptCreate: _schemas.ImageCreate = _fapi.Depends()):
    
    try:
        image_10, image_30, image_50, image_70 = await _services.generate_image(imgPrompt=imgPromptCreate)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    buffered = BytesIO()
    image_10.save(buffered, format="JPEG")
    encoded_img_10 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_30.save(buffered, format="JPEG")
    encoded_img_30 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_50.save(buffered, format="JPEG")
    encoded_img_50 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_70.save(buffered, format="JPEG")
    encoded_img_70 = base64.b64encode(buffered.getvalue())
    payload = {
        "mime" : "image/jpg",
        "image_10": encoded_img_10,
        "image_30": encoded_img_30,
        "image_50": encoded_img_50,
        "image_70": encoded_img_70
        }
    
    return payload
