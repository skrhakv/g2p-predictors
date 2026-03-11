FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

RUN mkdir /opt/af2bind && cd /opt/af2bind
WORKDIR /opt/af2bind

# install essential packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
       git \
       wget \
       unzip \
       aria2 

# install python 3.12
RUN apt update -y && apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz && \
    tar xzf Python-3.12.3.tgz && \
    cd Python-3.12.3 && \
    ./configure --enable-optimizations && \
    make altinstall

# install biotite for post-processing
RUN python3.12 -m pip install biotite pandas

# copy source code
COPY ./src /opt/af2bind

# force re-downloading the data
ARG CACHEBUST=1 
RUN wget https://zenodo.org/records/17683380/files/analysis_files.tar.gz?download=1 -O analysis_files.tar.gz && \
    tar -xzf analysis_files.tar.gz && \
    rm analysis_files.tar.gz
