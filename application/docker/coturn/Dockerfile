FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    coturn \
    && rm -rf /var/lib/apt/lists/*

COPY turnserver.conf /etc/turnserver.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 443

ENTRYPOINT ["/entrypoint.sh"]
CMD ["turnserver", "-c", "/etc/turnserver.conf"]
