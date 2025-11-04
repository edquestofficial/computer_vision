# Make sure these imports are at the top of your routes/company.py file
import pandas as pd
import io
import numpy as np
import asyncio
from fastapi import APIRouter,File, UploadFile,Form
import  base64
# from conn import mydb, cursor
# from util.face_match import face_encoding,incert
from db_config import get_connection
# from embedding_operation import facegenerating_embedding, face_embedding_search
import os
import shutil
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
import requests
from util.mailer import send_mail

router = APIRouter()

from .vector_store import process_registration_object, create_embedding_for_file

@router.get("/login")
async def companyadmin_login(username: str, password: str):
    try:
        # cursor.execute("SELECT * FROM company_admin WHERE username = %s AND password = %s", (username, password))
        # result = cursor.fetchone()

        connection = get_connection()

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username, role,company_id,active FROM ed_employees WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()
        cursor.execute("SELECT alias_name FROM company_details WHERE id = %s",(result["company_id"],))
        name = cursor.fetchone()

        cursor.close()
        connection.close()
        if not result:
            return {"message": "Invalid credentials"}
        return {"message": "Company Admin Login Successful",
                "data":result,
                "alias_name":name["alias_name"]}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/employees")
async def get_employees(username:str,alias_name:str):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True ,buffered=True)


    query = f"""SELECT * FROM  {alias_name}_employees WHERE active = 1"""
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    details = []
    for data in result:
        data["photo"] = ""
    return result

@router.post("/update_employee")
async def update_employee(name:str,username:str,role:str,id:int,alias_name:str,updated_by:str):
    query = f"UPDATE {alias_name}_employees SET name = %s,username=%s , role = %s , modified_by = %s,modified_at= CURRENT_TIMESTAMP() WHERE id = %s "
    connection = get_connection()
    cursor = connection.cursor(dictionary=True ,buffered=True)
    try:
        cursor.execute(query,(name,username,role,updated_by,id ))
        cursor.close()
        connection.close()
        return {"message":"updated"}
    except Exception as e :
        return {"error":str(e)}

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
        send  = send_mail(username,password, role)
        if send :
            print("mail send to HR.")
        else:
            print("Mail not send.")
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

@router.get("/role")
async def get_role():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True ,buffered=True)
    query = f"""SELECT * FROM  roles"""
    cursor.execute(query)
    result = cursor.fetchall()
    return result

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
    




# --- YOUR REQUIRED HEADERS ---
EXPECTED_HEADERS = [
    "s no.", "date", "name", "company name", "city", "state", "contact 1",
    "inquiry type", "e mail", "requirements", "status", "skybound person",
    "cold/hot/warm", "open/closed", "next follow up"
]
# -----------------------------
# @router.post("/setup-database")
# async def setup_database():
#     """
#     This is a one-time endpoint to create the inquiries table.
#     """
    
#     # ! IMPORTANT: Make sure this matches the table_name in your other function
#     table_name = "excel" 
    
#     # This is the SQL query from above
#     create_table_query = f"""
#     CREATE TABLE IF NOT EXISTS `{table_name}` (
#         `id` INT AUTO_INCREMENT PRIMARY KEY,
#         `date` DATETIME,
#         `name` VARCHAR(255),
#         `company name` VARCHAR(255),
#         `city` VARCHAR(255),
#         `state` VARCHAR(255),
#         `contact 1` VARCHAR(100),
#         `inquiry type` VARCHAR(255),
#         `e mail` VARCHAR(255),
#         `requirements` TEXT,
#         `status` VARCHAR(100),
#         `skybound person` VARCHAR(255),
#         `cold/hot/warm` VARCHAR(50),
#         `open/closed` VARCHAR(50),
#         `next follow up` DATETIME
#     );
#     """
    
#     connection = None
#     try:
#         connection = get_connection()
#         if not connection:
#             return JSONResponse(status_code=500, content={"error": "Database connection failed."})
        
#         cursor = connection.cursor()
#         cursor.execute(create_table_query)
#         connection.commit()
#         cursor.close()
        
#         return {"message": f"Table '{table_name}' created successfully (or already exists)."}

#     except Exception as e:
#         if connection:
#             connection.rollback()
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"An error occurred: {str(e)}"}
#         )
#     finally:
#         if connection:
#             connection.close()



@router.post("/import-excel/")
async def import_excel_data(file: UploadFile = File(...)):
    """
    This endpoint validates an Excel file's headers (case-insensitive)
    and, if valid, inserts the data into a MySQL database.
    """
    connection = None  # Initialize connection to None
    try:
        # Read the file's content into memory
        contents = await file.read()
        buffer = io.BytesIO(contents)
        df = pd.read_excel(buffer)

        # --- 1. Header Validation ---
        
        # Create a mapping of {Original Header: lowercase_header}
        header_map = {col: str(col).strip().lower() for col in df.columns}
        
        # Get a set of the standardized headers from the file
        standardized_file_headers = set(header_map.values())
        
        # Get a set of your required headers
        required_set = set(EXPECTED_HEADERS)

        # Check if all required headers are present in the file
        if not required_set.issubset(standardized_file_headers):
            missing_headers = list(required_set - standardized_file_headers)
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid file format. Missing required headers.",
                    "missing": missing_headers
                }
            )

# --- 2. Data Processing (ALL FIXES APPLIED) ---
        
        # Rename the DataFrame columns to your standardized lowercase names
        df = df.rename(columns=header_map)
        
        # --- FIX 1: Handle NaN, empty strings, and single spaces ---
        # This replaces all of them with None, which becomes NULL in MySQL
        df = df.replace({np.nan: None, '': None, ' ': None})
        
        # --- FIX 2: Handle bad DATE columns ---
        # 'errors=coerce' turns any bad date (like 'pending') into 'NaT'
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['next follow up'] = pd.to_datetime(df['next follow up'], errors='coerce')
        
        # --- FIX 3: Handle bad NUMBER columns ---
        # This fixes errors like "Incorrect integer value: ' ' for column 's no.'"
        # It turns any bad number (like 'N/A' or text) into 'NaN' (Not a Number)
        df['s no.'] = pd.to_numeric(df['s no.'], errors='coerce')
        
        # --- FIX 4: Convert all 'NaT' and 'NaN' into None ---
        # This final step makes the data safe for MySQL, which accepts NULL.
        # This single line replaces BOTH the old .replace({np.nan: None})
        # and .replace({pd.NaT: None})
        df = df.replace({pd.NaT: None, np.nan: None})        
        # Convert the DataFrame to a list of dictionaries
        data_rows = df.to_dict(orient="records")

        # --- 3. Database Insertion ---
        
        # ! IMPORTANT: Change this to your actual table name
        table_name = "excel" 
        
        connection = get_connection()
        if not connection:
            return JSONResponse(status_code=500, content={"error": "Database connection failed."})
        
        cursor = connection.cursor()

        # Build the SQL query dynamically
        # The backticks `` are important for names with spaces or symbols
        sql_columns = ", ".join([f"`{h}`" for h in EXPECTED_HEADERS])
        
        # This creates `(%s, %s, %s, ...)`
        sql_placeholders = ", ".join(["%s"] * len(EXPECTED_HEADERS))
        
        insert_query = f"INSERT INTO {table_name} ({sql_columns}) VALUES ({sql_placeholders})"
        
        # Prepare all rows for batch insertion
        rows_to_insert = []
        for row in data_rows:
            # Create a tuple of values *in the correct order*
            values_tuple = tuple(row[h] for h in EXPECTED_HEADERS)
            rows_to_insert.append(values_tuple)

        # Execute all inserts in a single, efficient transaction
        if rows_to_insert:
            cursor.executemany(insert_query, rows_to_insert)
            connection.commit()
            
        cursor.close()

        return {
            "message": "File validated and data saved successfully!",
            "filename": file.filename,
            "records_saved": len(rows_to_insert)
        }

    except Exception as e:
        # If anything goes wrong, roll back any changes
        if connection:
            connection.rollback()
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )
    finally:
        # Ensure the file and database connection are always closed
        if connection:
            connection.close()
        await file.close()
  

# @router.post("/fix-status-column")
# async def fix_status_column():
#     """
#     This is a one-time endpoint to alter the 'status' column
#     from VARCHAR to TEXT to allow longer data.
#     """
    
#     # ! IMPORTANT: Make sure this is your real table name
#     table_name = "excel" 
    
#     alter_query = f"""
#     ALTER TABLE `{table_name}` 
#     MODIFY COLUMN `status` TEXT;
#     """
    
#     connection = None
#     try:
#         connection = get_connection()
#         if not connection:
#             return JSONResponse(status_code=500, content={"error": "Database connection failed."})
        
#         cursor = connection.cursor()
#         cursor.execute(alter_query)
#         connection.commit()
#         cursor.close()
        
#         return {"message": f"Table '{table_name}' status column successfully changed to TEXT."}

#     except Exception as e:
#         if connection:
#             connection.rollback()
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"An error occurred: {str(e)}"}
#         )
#     finally:
#         if connection:
#             connection.close()
