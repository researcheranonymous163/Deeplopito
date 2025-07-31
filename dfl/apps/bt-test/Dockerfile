FROM ubuntu:24.04

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists/,type=cache,sharing=locked \
    --mount=target=/var/cache/apt/,type=cache,sharing=locked \
    	apt update && apt install -y python3.12 python3.12-venv qbittorrent-nox git

COPY node-qBittorrent-config /root/.config/qBittorrent/

WORKDIR /opt/bt-test/
ENV VIRTUAL_ENV=/opt/bt-test/.venv/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3.12 -m venv $VIRTUAL_ENV

COPY ./requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip/,sharing=locked \
    pip install -r ./requirements.txt

COPY ./ ./

ENTRYPOINT ["./docker-entrypoint.sh"]