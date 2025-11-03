from fastapi import APIRouter, UploadFile, File
import random
from db_config import get_connection
router = APIRouter()

@router.get("/attendence")
async def get_attendence():
    # Read the uploaded image (optional — you can ignore if not needed)
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM attendence")
    result = cursor.fetchall()
    
    cursor.close()
    connection.close()
    return result