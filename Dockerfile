FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/api_deployment.py .
CMD ["uvicorn", "api_deployment.py:app", "--host", "0.0.0.0", "--port", "8000"]