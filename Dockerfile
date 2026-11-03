FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY models/api_reference.joblib models/api_reference.json ./models/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
USER 65534
CMD ["uvicorn", "predictive_maintenance.api:app", "--host", "0.0.0.0", "--port", "8000"]
