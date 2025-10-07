FROM python:3.10.17

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]

