#
# Stage 1: Build p2rank 2.6-alpha from source
#
FROM debian:bookworm-20260112-slim AS p2rank-build

RUN apt-get update \
  && apt-get -y --no-install-recommends install openjdk-17-jdk ca-certificates

WORKDIR /build
COPY p2rank/ ./

RUN chmod +x ./gradlew && ./gradlew --no-daemon assemble

#
# Stage 2: Runtime image
#
FROM debian:bookworm-20260112-slim

RUN apt-get update \
  && apt-get -y --no-install-recommends install openjdk-17-jdk

WORKDIR /opt/p2rank

COPY --from=p2rank-build /build/distro ./
