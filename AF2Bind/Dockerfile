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

# install python packages and download model parameters
RUN python3.12 -m pip install "jax[cuda]==0.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && python3.12 -m pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1 \
    && aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar \
    && mkdir af2bind_params \
    && wget -qnc https://github.com/sokrypton/af2bind/raw/main/attempt_7_2k_lam0-03.zip \
    && unzip attempt_7_2k_lam0-03.zip -d af2bind_params \
    && mkdir params \
    && tar -xf alphafold_params_2021-07-14.tar -C params

# copy source code
COPY ./src /opt/af2bind