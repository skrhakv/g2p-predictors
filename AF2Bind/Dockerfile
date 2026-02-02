FROM python:3.12.3-slim-bullseye

RUN mkdir /opt/af2bind && cd /opt/af2bind
WORKDIR /opt/af2bind

RUN apt-get update && apt-get install -y --no-install-recommends \
       git \
       wget \
       unzip \
       aria2
       
RUN python3 -m pip install "jax[cuda]==0.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && python3 -m pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1 \
    && aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar \
    && mkdir af2bind_params \
    && wget -qnc https://github.com/sokrypton/af2bind/raw/main/attempt_7_2k_lam0-03.zip \
    && unzip attempt_7_2k_lam0-03.zip -d af2bind_params \
    && mkdir params \
    && tar -xf alphafold_params_2021-07-14.tar -C params

COPY ./src /opt/af2bind/src