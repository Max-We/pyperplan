FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    matplotlib

COPY . .
RUN pip install --no-cache-dir .


CMD ["sh", "-c", "python dev/evaluate_agile.py || true; tail -f /dev/null"]
