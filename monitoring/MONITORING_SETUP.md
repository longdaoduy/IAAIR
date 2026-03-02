# IAAIR Grafana/Prometheus Monitoring Setup

This guide explains how to set up comprehensive monitoring for the IAAIR system using Prometheus for metrics collection and Grafana for visualization and alerting.

## Architecture Overview

```
IAAIR API (Port 8000)
    ↓ /metrics endpoint
Prometheus (Port 9090)
    ↓ scrapes metrics
Grafana (Port 3000)
    ↓ queries Prometheus
Alertmanager (Port 9093)
    ↓ sends alerts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client
```

### 2. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/iaair123)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

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

1. **Metrics Not Appearing**:
   - Check `/metrics` endpoint: `curl http://localhost:8000/metrics`
   - Verify Prometheus targets: http://localhost:9090/targets
   - Check container logs: `docker-compose logs prometheus`

2. **Dashboard Not Loading**:
   - Verify Grafana datasource configuration
   - Check Prometheus connectivity from Grafana
   - Reload dashboards: `docker-compose restart grafana`

3. **Alerts Not Firing**:
   - Check alert rules syntax in Prometheus
   - Verify Alertmanager configuration
   - Test alert expressions in Prometheus UI

### Log Analysis

Enable debug logging for monitoring components:

```python
logging.getLogger('models.engines.PrometheusMonitor').setLevel(logging.DEBUG)
```

## Production Deployment

### Security
- Change default passwords
- Enable HTTPS for Grafana
- Use authentication providers (LDAP, OAuth)
- Secure Prometheus scrape endpoints

### Scaling
- Use Prometheus federation for multiple instances
- Implement service discovery for dynamic targets
- Use remote storage for long-term retention
- Set up Grafana clustering for high availability

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