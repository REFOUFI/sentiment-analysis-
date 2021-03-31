#FROM python:3.6-slim-stretch
FROM ubuntu:20.04

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN apt-get install -y libgomp1

ADD requirements.txt /
RUN pip install -r /requirements.txt



ADD . /app
WORKDIR /app

EXPOSE 5000
CMD [ "/usr/bin/python3.6" , "app.py"]
