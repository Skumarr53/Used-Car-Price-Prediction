FROM python:3.7.5-slim

ENV APP /app
RUN mkdir $APP
WORKDIR $APP

EXPOSE 5000

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

COPY . .

CMD ["python", "application.py"]
