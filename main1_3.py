import asyncio
from fastapi import (
    FastAPI, APIRouter, File, UploadFile, Form, HTTPException
)
from fastapi.responses import JSONResponse
from typing import List
import requests

# Import only the NEW "manager" function from your vector_store.py
from vector_store import process_registration_object

app = FastAPI()
router = APIRouter()




@router.post("/register-faces")
async def register_faces_dynamic(
    # This is your "id: 14"
    employee_id: str = Form(...),
    # This is your "name: Gobind"
    name: str = Form(...),
    # This is your "images: [...]"
    photos: List[UploadFile] = File(...)
):
    """
    Receives the form "object" (id, name, photos) and passes
    it directly to vector_store.py for processing.
    """
    
    print(f"FastAPI route received {len(photos)} images for {employee_id}.")
    print("Passing entire object to vector_store for processing...")

    try:
        # 1. Call the ONE function in vector_store.py in a thread
        # This function does ALL the work, including the loop
        result = await asyncio.to_thread(
            process_registration_object, 
            employee_id, 
            name, 
            photos
        )
        
        # 2. Return the final report from vector_store
        return JSONResponse(
            status_code=201, 
            content=result
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Include the router in your main app
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Run this file with: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)










