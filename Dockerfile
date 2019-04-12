FROM python:3
WORKDIR /app
COPY . /app
cmd ["python","abcd.py"]
