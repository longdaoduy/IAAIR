# IAAIR Grafana Cloud Monitoring Setup

This guide explains how to set up monitoring for the IAAIR system using **Grafana Cloud** for dashboards and alerting, with Prometheus scraping metrics locally and pushing them via `remote_write`.

## Architecture Overview

```
IAAIR API (Port 8000)
    ↓ /metrics endpoint
Prometheus (local, Port 9090)
    ↓ remote_write
Grafana Cloud Mimir (daongochoa2002.grafana.net)
    ↓ stored metrics
Grafana Cloud Dashboards
    https://daongochoa2002.grafana.net
```

**Alternative (no local Prometheus):**

```
IAAIR API (Port 8000)
    ↓ /metrics endpoint
Grafana Alloy (local agent)
    ↓ remote_write
Grafana Cloud Mimir → Grafana Cloud Dashboards
```

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client
```

### 2. Configure Grafana Cloud Credentials

```bash
cd monitoring
cp .env.example .env
```

Edit `.env` and fill in your Grafana Cloud credentials:

1. Go to **https://daongochoa2002.grafana.net**
2. Navigate to **Connections → Add new connection → Prometheus**
3. Copy your **Instance ID** → `GRAFANA_CLOUD_PROM_USERNAME`
4. Generate an **API Key** with `MetricsPublisher` role → `GRAFANA_CLOUD_API_KEY`
5. Copy the **Remote Write URL** → `GRAFANA_CLOUD_REMOTE_WRITE_URL`
6. Update `prometheus.yml` `remote_write.url` if your region differs

### 3. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

This starts:
- **Prometheus** — scrapes IAAIR `/metrics` and pushes to Grafana Cloud
- **Alloy** — alternative lightweight agent (also pushes to Grafana Cloud)
- **Alertmanager** — local alert routing (optional)

### 4. Access Dashboards

- **Grafana Cloud**: https://daongochoa2002.grafana.net
- **Prometheus (local)**: http://localhost:9090
- **Alertmanager (local)**: http://localhost:9093
- **Alloy UI**: http://localhost:12345

### 5. Import the Dashboard

1. Go to https://daongochoa2002.grafana.net → **Dashboards → Import**
2. Upload `grafana/dashboards/iaair-performance.json`
3. Select the **Prometheus** datasource (auto-provisioned from remote_write)

## Metrics Collected

### Request Metrics
- `iaair_requests_total` - Total requests by endpoint, routing strategy, query type
- `iaair_request_duration_seconds` - Request latency distribution
- `iaair_active_requests` - Current active requests

### Component Performance
- `iaair_embedding_duration_seconds` - Embedding generation time
- `iaair_vector_search_duration_seconds` - Vector search latency
- `iaair_graph_search_duration_seconds` - Graph search latency
- `iaair_reranking_duration_seconds` - Reranking latency
- `iaair_ai_response_duration_seconds` - AI response generation time

### Cache Performance
- `iaair_cache_hits_total` - Cache hits by type
- `iaair_cache_misses_total` - Cache misses by type
- `iaair_cache_size` - Current cache sizes

### Results & Quality
- `iaair_results_count` - Number of results returned
- `iaair_routing_strategy_performance_seconds` - Performance by routing strategy

### System Health
- `iaair_errors_total` - Errors by type and component
- `iaair_system_info` - System information

## Grafana Dashboard

The pre-configured dashboard includes:

1. **Overview Panel**: Request rate, active requests, response time, error rate
2. **Latency Analysis**: Response time distribution and component breakdown
3. **Cache Performance**: Hit rates and cache sizes
4. **Routing Analysis**: Strategy usage and performance
5. **Result Quality**: Result count distributions

### Key Visualizations

- **Response Time Percentiles**: 50th, 95th, 99th percentiles
- **Component Latency Breakdown**: Time spent in each component
- **Cache Hit Rate Trends**: Performance of caching system
- **Routing Strategy Effectiveness**: Which strategies perform best
- **Error Rate Monitoring**: System health indicators

## Alerting Rules

### Critical Alerts
- **Service Down**: API is unreachable
- **High Error Rate**: >10% error rate for 1 minute

### Warning Alerts
- **High Latency**: 95th percentile >5 seconds for 2 minutes
- **Low Cache Hit Rate**: <30% hit rate for 5 minutes
- **High Active Requests**: >20 concurrent requests
- **Slow Components**: Embedding >2s, Vector Search >5s

### Configuration

Alerts are configured in `iaair_alerts.yml` and managed by Alertmanager. Configure notification channels in `alertmanager.yml`:

- Email notifications
- Webhook integrations
- Slack/Teams integration

## Monitoring Best Practices

### 1. Dashboard Usage
- Monitor the overview panel for real-time health
- Check component latency breakdown for bottlenecks
- Review cache performance regularly
- Analyze routing strategy effectiveness

### 2. Alert Response
- **High Latency**: Check component breakdown, optimize slow components
- **Low Cache Hit Rate**: Analyze query patterns, increase cache sizes
- **High Error Rate**: Check logs, investigate failing components
- **Service Down**: Check container/process status

### 3. Performance Optimization
- Use latency breakdown to identify bottlenecks
- Monitor cache hit rates to optimize caching strategy
- Analyze routing performance to improve query classification
- Track result quality metrics

## Advanced Configuration

### Custom Metrics

Add custom metrics in `PrometheusMonitor.py`:

```python
custom_metric = Counter(
    'iaair_custom_metric',
    'Description',
    ['label1', 'label2'],
    registry=self.registry
)
```

### Dashboard Customization

Edit `grafana/dashboards/iaair-performance.json` to:
- Add new panels
- Modify visualizations
- Create custom alerts
- Add business metrics

### Multi-Environment Setup

For production deployments:

1. **Separate Configurations**:
   ```yaml
   # docker-compose.prod.yml
   environment:
     - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
   ```

2. **External Prometheus**:
   ```yaml
   # Use external Prometheus instance
   external_links:
     - prometheus:prometheus-host
   ```

3. **Persistent Storage**:
   ```yaml
   volumes:
     - /data/grafana:/var/lib/grafana
     - /data/prometheus:/prometheus
   ```

## Troubleshooting

### Common Issues

1. **Metrics Not Appearing in Grafana Cloud**:
   - Check `/metrics` endpoint: `curl http://localhost:8000/metrics`
   - Verify Prometheus targets: http://localhost:9090/targets
   - Check remote_write status: http://localhost:9090 → Status → TSDB Status
   - Check container logs: `docker-compose logs prometheus`
   - Verify `.env` credentials are correct
   - Confirm remote_write URL matches your Grafana Cloud region

2. **Authentication Errors on remote_write**:
   - Regenerate API key at https://daongochoa2002.grafana.net/org/apikeys
   - Ensure API key has `MetricsPublisher` role
   - Check `GRAFANA_CLOUD_PROM_USERNAME` is the Instance ID (numeric), not your email

3. **Dashboard Not Loading**:
   - Import `grafana/dashboards/iaair-performance.json` manually
   - Verify the Prometheus datasource is configured in Grafana Cloud
   - Check that metric names match (prefix: `iaair_`)

4. **Alloy Not Connecting**:
   - Check Alloy logs: `docker-compose logs alloy`
   - Verify `.env` has `GRAFANA_CLOUD_REMOTE_WRITE_URL`
   - Test connectivity: `curl -u <username>:<api_key> <remote_write_url>`

### Log Analysis

Enable debug logging for monitoring components:

```python
logging.getLogger('models.engines.PrometheusMonitor').setLevel(logging.DEBUG)
```

## Production Deployment

### Security
- Store `.env` secrets in a vault (never commit `.env` to git)
- Use Grafana Cloud's built-in HTTPS and SSO
- Restrict API key permissions to `MetricsPublisher` only
- Secure local Prometheus scrape endpoints

### Scaling
- Use Grafana Alloy instead of full Prometheus for lighter footprint
- Grafana Cloud Mimir handles storage, retention, and HA automatically
- No need for Prometheus federation — remote_write handles it

### Maintenance
- Regular backup of Grafana dashboards
- Monitor monitoring stack resource usage
- Implement log rotation
- Plan for metrics retention policies

## Integration with Existing Tools

### CI/CD Integration
```bash
# Check metrics endpoint in health checks
curl -f http://localhost:8000/metrics > /dev/null
```

### Load Testing
Monitor metrics during load tests to identify performance characteristics and bottlenecks.

### Development Workflow
Use local monitoring stack during development to catch performance issues early.

## Metric Collection Overhead

The monitoring system is designed to be lightweight:
- Metrics collection: <1ms overhead per request
- Memory usage: ~10MB for metrics storage
- CPU impact: <1% additional load
- Network: Minimal impact (metrics scraped every 10s)

## Next Steps

1. **Deploy monitoring stack**: `docker-compose up -d`
2. **Configure alerts**: Update contact information in `alertmanager.yml`  
3. **Customize dashboard**: Add business-specific panels
4. **Set up automation**: Integrate with deployment pipelines
5. **Train team**: Ensure team knows how to use dashboards and respond to alerts

The monitoring system provides comprehensive visibility into IAAIR performance, enabling proactive optimization and quick issue resolution.