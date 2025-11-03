import asyncio
from fastapi import APIRouter,File, UploadFile,Form
# from conn import mydb, cursor
# from util.face_match import face_encoding,incert
from db_config import get_connection
# from embedding_operation import facegenerating_embedding, face_embedding_search
import os
import shutil
from fastapi.responses import JSONResponse
from typing import List
import requests
router = APIRouter()

from .vector_store import process_registration_object, create_embedding_for_file

@router.get("/login")
async def companyadmin_login(username: str, password: str):
    try:
        # cursor.execute("SELECT * FROM company_admin WHERE username = %s AND password = %s", (username, password))
        # result = cursor.fetchone()

        connection = get_connection()

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM company_admin WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchall()
        
        cursor.close()
        connection.close()
        if not result:
            return {"message": "Invalid credentials"}
        return {"message": "Company Admin Login Successful",
                "data":result}
    except Exception as e:
        return {"error": str(e)}
    


@router.post("/employee")
async def add_employee(
    Company_alias: str ,
    name: str ,
    username: str ,
    password: str ,
    role: str ,
    created_by: str ,
    photos: List[UploadFile] = File(...),
):
    """
    Add an employee, save photos to disk, insert record in DB,
    and generate face embeddings for each uploaded photo.
    """
    try:
        base_path = r"C:\Users\edquestofficial\Desktop\Yogi\embeddingface\data"
        os.makedirs(base_path, exist_ok=True)

        # Ensure exactly 4 photos are provided
        if len(photos) != 4:
            return JSONResponse(
                status_code=400,
                content={"error": "Exactly 4 photos are required."}
            )

        saved_paths = []
        for photo in photos:
            file_path = os.path.join(base_path, photo.filename)
            with open(file_path, "wb") as f:
                f.write(await photo.read())
            saved_paths.append(file_path)

        print("Saved photo paths:", saved_paths)

        # Connect to the database
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        if role.lower() == "admin":
            return JSONResponse(
                status_code=403,
                content={"error": "Cannot add Admin. Only Edquest can add Admin users."}
            )

        # Get company ID
        cursor.execute(
            "SELECT id FROM company_details WHERE alias_name = %s", 
            (Company_alias,)
        )
        company = cursor.fetchone()
        if not company:
            return JSONResponse(
                status_code=404,
                content={"error": "Company not found."}
            )

        company_id = company["id"]

        # Store one representative photo (e.g., first one)
        with open(saved_paths[0], "rb") as f:
            photo_data = f.read()

        insert_query = f"""
            INSERT INTO {Company_alias}_employees 
            (company_id, name, photo, username, password, role, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (company_id, name, photo_data, username, password, role, created_by))
        connection.commit()

        print("Employee inserted successfully in DB.")

        # Call face embedding function in a background thread (non-blocking)
        print("Starting face embedding generation...")
        result = await asyncio.to_thread(
            process_registration_object, 
            username, 
            name, 
            saved_paths
)


        cursor.close()
        connection.close()

        return JSONResponse(
            status_code=201,
            content={
                "message": "Employee added successfully.",
                "photos_saved": saved_paths,
                "embedding_results": result
            }
        )

    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.delete("/employee")
async def delete_employee(Company_alias: str, username: str):

    table_name = f"{Company_alias}_employees"
    try:
        connection = get_connection()

        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"UPDATE {table_name} SET active=0 WHERE username = %s", (username,))
        cursor.close()
        connection.close()
        return {"message": "Employee deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
  