FROM debian:bookworm-20260112-slim

ARG P2RANK_URL=https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz

RUN apt-get update \
  && apt-get -y --no-install-recommends install openjdk-17-jdk wget

RUN mkdir /opt/p2rank && cd /opt/p2rank 
WORKDIR /opt/p2rank

RUN wget -O p2rank.tar.gz $P2RANK_URL -q \
  && tar -xf p2rank.tar.gz \
  && rm p2rank.tar.gz \
  # p2rank archive contains version, we need to get rid of that
  && mv p2rank_*/* ./ \
  && rm -r p2rank_*
