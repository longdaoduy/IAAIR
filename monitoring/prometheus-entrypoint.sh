#!/bin/sh
# Render prometheus.yml template by expanding environment variables,
# then start Prometheus.
set -e

sed \
  -e "s|\${GRAFANA_CLOUD_PROM_USERNAME}|${GRAFANA_CLOUD_PROM_USERNAME}|g" \
  -e "s|\${GRAFANA_CLOUD_API_KEY}|${GRAFANA_CLOUD_API_KEY}|g" \
  -e "s|\${GRAFANA_CLOUD_REMOTE_WRITE_URL}|${GRAFANA_CLOUD_REMOTE_WRITE_URL}|g" \
  /etc/prometheus/prometheus.template.yml > /etc/prometheus/prometheus.yml

exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.console.templates=/etc/prometheus/consoles \
  --storage.tsdb.retention.time=200h \
  --web.enable-lifecycle
