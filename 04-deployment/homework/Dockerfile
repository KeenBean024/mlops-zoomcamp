FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
COPY [ "model2.bin", "model.bin" ]
COPY ["starter.py", "Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
ENTRYPOINT ["python", "starter.py", "05", "2023"]