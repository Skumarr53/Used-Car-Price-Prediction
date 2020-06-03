ARG VARIANT="3.7"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

ENV APP /app
RUN mkdir $APP
WORKDIR $APP

EXPOSE 5000

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

COPY . .

CMD ["python", "application.py"]