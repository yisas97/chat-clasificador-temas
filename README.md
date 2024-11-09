# 1. Instalar Python
# 2. Crea un nuevo entorno virtual con Python
python -m venv env-clasificador-new
# 3. Activa el entorno virtual
.\env-clasificador-new\Scripts\activate
# 4. Actualiza pip
python -m pip install --upgrade pip
# 5. Levanta con python
python .\src\main.py

# 6. Para consultar en POSTMAN
http://localhost:8000/analyze
METODO: POST
-------------------------
FORM-DATA:
file(type: file): "el archivo" 
method:kmeans o lda
generate_summary:false