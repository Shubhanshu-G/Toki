FROM python:3.12-slim
COPY . /app
WORKDIR  /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502"]