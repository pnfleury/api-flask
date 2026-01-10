FROM python:3.12-slim   

WORKDIR /app                

# Copia dependências
COPY requirements.txt .    
RUN pip install --no-cache-dir -r requirements.txt

# copia app e modelo para /app
COPY app.py .                   	
COPY modelo.joblib .		

 # porta da API
EXPOSE 5000                

# comando de inicialização
CMD ["python", "app.py"]    
