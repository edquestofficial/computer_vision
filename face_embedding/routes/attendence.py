from fastapi import APIRouter, UploadFile, File
import random

router = APIRouter()

@router.post("/check_image")
async def check_image(photo: UploadFile = File(...)):
    # Read the uploaded image (optional — you can ignore if not needed)
    await photo.read()

    # Randomly choose True or False
    result = random.choice([True, False])

    return {"status": result}