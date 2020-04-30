FROM python:alpine3.7

# Set up and activate virtual environment
ENV VIRTUAL_ENV "/CarPricePredictEnv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN apk add --no-cache python3-dev gcc \
    gfortran musl-dev libopenblas-dev liblapack-dev cython\
    && pip3 install --upgrade pip 
COPY . /app
WORKDIR /app
RUN pip3 --no-cache-dir install -r requirements.txt 
EXPOSE 5001  
CMD [ "python","application.py" ]