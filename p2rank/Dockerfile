#
# Stage 1: Download p2rank source at pinned commit (develop / 2.6-alpha)
#
FROM debian:bookworm-20260112-slim AS p2rank-src

ARG P2RANK_COMMIT=af1d6eeb1855bd15cc63df24e6a84a71ac99f2ea

RUN apt-get update \
  && apt-get -y --no-install-recommends install wget ca-certificates

RUN wget -q https://github.com/rdk/p2rank/archive/${P2RANK_COMMIT}.tar.gz -O p2rank.tar.gz \
  && tar -xzf p2rank.tar.gz \
  && mv p2rank-${P2RANK_COMMIT} /p2rank \
  && rm p2rank.tar.gz

#
# Stage 2: Build p2rank
#
FROM debian:bookworm-20260112-slim AS p2rank-build

RUN apt-get update \
  && apt-get -y --no-install-recommends install openjdk-17-jdk ca-certificates

WORKDIR /build
COPY --from=p2rank-src /p2rank ./

RUN chmod +x ./gradlew && ./gradlew --no-daemon assemble

#
# Stage 3: Runtime image
#
FROM debian:bookworm-20260112-slim

RUN apt-get update \
  && apt-get -y --no-install-recommends install openjdk-17-jdk

WORKDIR /opt/p2rank

COPY --from=p2rank-build /build/distro ./
