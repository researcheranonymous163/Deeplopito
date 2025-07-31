FROM docker.io/library/debian:trixie-slim

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt/,sharing=locked \
  	--mount=type=cache,target=/var/lib/apt/,sharing=locked \
  		apt update && apt install -y qbittorrent-nox curl python3

COPY ./config/ /root/.config/qBittorrent/

ENTRYPOINT [ "qbittorrent-nox" ]

# See https://github.com/qbittorrent/qBittorrent/wiki/How-to-use-portable-mode.
CMD ["--relative-fastresume"]