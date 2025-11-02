from fastapi import FastAPI, HTTPException
from db_config import get_connection

app = FastAPI(title="MySQL FastAPI Example")


from fastapi.middleware.cors import CORSMiddleware
# from data.super_admin import router as super_router
from routes.company_admin import router as company_admin_router
from routes.attendence import router as attendence_router 
# from routes.face_match import router as face_match_router 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(company_admin_router, prefix="/companyadmin")
# app.include_router(super_router, prefix="/superadmin")
app.include_router(attendence_router, prefix="/attendence")
app.include_router(attendence_router, prefix="/facematch")

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI + MySQL!"}

@app.get("/employees")
def get_employees():
    connection = get_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees;")  # Replace with your table name
    result = cursor.fetchall()
    
    cursor.close()
    connection.close()
    return {"employees": result}

@app.get("/employee/{emp_id}")
def get_employee(emp_id: int):
    connection = get_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE id = %s;", (emp_id,))
    result = cursor.fetchone()
    
    cursor.close()
    connection.close()

    if not result:
        raise HTTPException(status_code=404, detail="Employee not found")

    return result
