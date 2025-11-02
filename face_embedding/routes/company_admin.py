from fastapi import APIRouter,File, UploadFile
# from conn import mydb, cursor
# from util.face_match import face_encoding,incert
from db_config import get_connection
from embedding_operation import facegenerating_embedding, face_embedding_search
import os
import shutil
from fastapi.responses import JSONResponse


router = APIRouter()

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
        return {"message": "Company Admin Login Successful"}
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/employee")
async def add_employee(Company_alias: str,
                       name: str ,
                       username: str,
                       password: str ,
                       role: str ,
                       photo: UploadFile,
                       created_by: str ):
    photo_data = await photo.read()
    base_path = "C:\\Users\\edquestofficial\\Desktop\\Yogi\\embeddingface\\data"

    file_path = os.path.join(base_path, photo.filename)
    print("file_path",file_path)
    # Save uploaded photo into folder
    with open(file_path, "wb") as f:
        f.write(photo_data)
    connection = get_connection()

    cursor = connection.cursor(dictionary=True)
    # encoding = face_encoding(photo)
    # print(encoding)
    if role == "Admin":
        return {"error": "Cannot add Admin. Only Edquest can add Admin users."}
    table_name = f"{Company_alias}_employees"
    query = f"""
        INSERT INTO {table_name} (company_id, name,photo, username, password, role, created_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute("SELECT id FROM company_details WHERE alias_name = %s", (Company_alias,))
        company = cursor.fetchone()
        if not company:
            return {"error": "Company not found"}
        cursor.execute(query, (123, name, photo_data ,username, password, role, created_by))
        # mydb.commit()
        connection.commit()
        cursor.close()
        connection.close()
        result = await facegenerating_embedding(8,username,file_path)
        return {"message": "Employee added successfully", "result":result}
    except Exception as e:
        print({"error": str(e)})
        return {"error": str(e)}
    

# @router.post("/generate_embedding")
# async def generate_embedding(
#     id: str = Form(...),
#     username: str = Form(...),
#     photo: UploadFile = File(...)
# ):
#     # Read uploaded photo
#     photo_path = f"temp_{photo.filename}"
#     with open(photo_path, "wb") as buffer:
#         buffer.write(await photo.read())

#     # Call the embedding function
#     facegenerating_embedding(id, username, photo_path)

#     return {"message": f"Embedding generated for {username}"}

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
    
DATA_DIR = r"C:\Users\edquestofficial\Desktop\Yogi\embeddingface\data"
os.makedirs(DATA_DIR, exist_ok=True)
@router.post("/similarity_search")
# async def post_similarity_search(file:UploadFile):
#     return face_embedding_search(file.filename)

async def post_similarity_search(file:UploadFile):
    try:

        # Save uploaded file temporarily
        # temp_path = os.path.join(DATA_DIR, file.filename)
        # with open(temp_path, "wb") as buffer:
        #     shutil.copyfileobj(file.file, buffer)

        # # Run face embedding search
        # result = face_embedding_search(temp_path)

        # # Delete the temp image after search
        # os.remove(temp_path)

        return {
            "status":"true",
            "username":"CG"
        }

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})
    
@router.post("/test_similarity_search")   
async def similarity_search(file:UploadFile):
    try:
        BASE_DIR = os.getcwd()
        # Save uploaded file temporarily
        temp_path = os.path.join(BASE_DIR,"embeddingface","temp", file.filename)
        print("temp_path", temp_path)
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Run face embedding search
        result = face_embedding_search(temp_path)

        # # Delete the temp image after search
        # os.remove(temp_path)

        return result

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})
