from fastapi import APIRouter,File, UploadFile, Form
from db_config import get_connection


router = APIRouter()

connection = get_connection()
cursor = connection.cursor(dictionary=True)

@router.get("/login")
async def superadmin_login(username: str, password: str):
    try:
        cursor.execute("SELECT * FROM super_admin WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()
        if not result:
            return {"message": "Invalid credentials"}
        return {"message": "Super Admin Login Successful"}
    except Exception as e:
        return {"error": str(e)}

@router.post("/company")
async def add_company(
    name: str,
    alias_name:str,
    location: str,
    created_by: str,
    logo: UploadFile
):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    logo_data = await logo.read()

    query = """
        INSERT INTO company_details (name, alias_name, location, created_by, logo)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor.execute("SELECT id FROM company_details WHERE alias_name = %s", (alias_name,))
        existing = cursor.fetchone()
        if existing:
            return {"error": "Company with this alias name already exists"}
        cursor.execute(query, (name, alias_name, location, created_by, logo_data))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        return {"error": str(e)}
    
    # Create company-specific employee table
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        query = " CREATE TABLE IF NOT EXISTS `{}_employees` (id INT AUTO_INCREMENT PRIMARY KEY, company_id INT, name VARCHAR(50),photo LONGBLOB ,username VARCHAR(255) UNIQUE, password VARCHAR(255), role VARCHAR(50), created_by VARCHAR(50), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, modified_by VARCHAR(50), modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, active BOOLEAN DEFAULT TRUE)".format(alias_name)
        cursor.execute(query)
        connection.commit()
        return {"message": "Company added successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@router.delete("/company")
async def delete_company(Company_alias: str):
    try:
        cursor.execute("UPDATE company_details SET active=0 WHERE alias_name = %s", (Company_alias,))
        connection.commit()
        # # Drop company-specific employee table
        # table_name = f"{Company_alias}_employees"
        # cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        # mydb.commit()
        return {"message": "Company and associated employee table deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/employee")
async def add_employee(Company_alias: str,
                       name: str ,
                       username: str,
                       password: str ,
                       created_by: str ):
    table_name = f"{Company_alias}_employees"
    query = f"""
        INSERT INTO {table_name} (company_id, name, username, password, role, created_by)
        VALUES ((SELECT id FROM company_details WHERE alias_name = %s), %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute("SELECT id FROM company_details WHERE alias_name = %s", (Company_alias,))
        company = cursor.fetchone()
        if not company:
            return {"error": "Company not found"}
        cursor.execute(query, (Company_alias, name, username, password, "Admin", created_by))
        connection.commit()
        return {"message": "Employee added successfully"}
    except Exception as e:
        return {"error": str(e)}

@router.delete("/employee")
async def delete_employee(Company_alias: str, username: str):
    table_name = f"{Company_alias}_employees"
    try:
        cursor.execute(f"UPDATE {table_name} SET active=0 WHERE username = %s", (username,))
        connection.commit()
        return {"message": "Employee deleted successfully"}
    except Exception as e:
        return {"error": str(e)}







# from fastapi.responses import StreamingResponse
# import io

# @router.get("/company/{company_id}/logo")
# async def get_company_logo(company_id: int):
#     cursor.execute("SELECT logo FROM company_details WHERE id = %s", (company_id,))
#     result = cursor.fetchone()

#     if not result or not result[0]:
#         return {"error": "Logo not found"}

#     logo_bytes = result[0]
#     return StreamingResponse(io.BytesIO(logo_bytes), media_type="image/png")
