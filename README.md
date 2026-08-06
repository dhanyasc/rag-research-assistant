# RAG Research Assistant – Production Deployment

A production-grade Document Q&A system deployed on AWS ECS with Docker containers, GitHub Actions CI/CD, Prometheus monitoring, Grafana dashboards, and JWT authentication.

## Architecture

```
                    ┌──────────────┐
                    │  GitHub      │
                    │  Actions     │
                    │  CI/CD       │
                    └──────┬───────┘
                           │ push to ECR
                    ┌──────▼───────┐
                    │  AWS ECR     │
                    │  (Docker)    │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │     AWS ECS Fargate     │
              │  ┌───────────────────┐  │
              │  │  RAG API (FastAPI)│  │
              │  │  /upload /ask     │  │
              │  │  /auth /metrics   │  │
              │  └────────┬──────────┘  │
              └───────────┼─────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
     ┌──────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
     │ Prometheus   │ │ JWT    │ │ CloudWatch  │
     │ (metrics)    │ │ Auth   │ │ (logs)      │
     └──────┬──────┘ └────────┘ └─────────────┘
            │
     ┌──────▼──────┐
     │   Grafana    │
     │ (dashboards) │
     └─────────────┘
```

## Quick Start (Local)

```bash
# Start everything with Docker Compose
docker-compose up --build

# API:        http://localhost:8000
# Swagger:    http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

## API Usage

### Register & Login

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secure123"}'

# Response: {"access_token": "eyJ...", "token_type": "bearer"}
```

### Upload & Query (requires JWT)

```bash
TOKEN="eyJ..."

# Upload document
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@paper.pdf"

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "top_k": 3}'
```

## Deploy to AWS

```bash
# One-time setup
chmod +x aws/setup-aws.sh
./aws/setup-aws.sh

# Set GitHub Secrets:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_ACCOUNT_ID

# Then every push to main auto-deploys via GitHub Actions
```

## CI/CD Pipeline

On every push:
1. **Test** – runs 58-test pytest suite
2. **Lint** – ruff code quality checks
3. **Deploy** (main only) – builds Docker image, pushes to ECR, updates ECS service with zero-downtime rolling deployment

## Monitoring

### Prometheus Metrics (`/metrics`)

| Metric | Type | Description |
|--------|------|-------------|
| `rag_query_latency_seconds` | Histogram | End-to-end query latency |
| `rag_query_confidence` | Histogram | Answer confidence distribution |
| `rag_grounded_answers_total` | Counter | Grounded vs ungrounded answers |
| `rag_query_total` | Counter | Total queries processed |
| `rag_documents_loaded` | Gauge | Document chunks in store |
| `http_requests_total` | Counter | HTTP requests by method/endpoint/status |
| `http_request_duration_seconds` | Histogram | HTTP request latency |

### Grafana Dashboard

Pre-configured dashboard with panels for:
- Query latency percentiles (p50/p95/p99)
- Requests per second by endpoint
- Confidence score distribution
- Grounded vs ungrounded answer ratio
- 24h uptime percentage
- Active users and document count

## Testing

```bash
pytest test_rag.py -v
```

58 tests covering document processing, vector search, RAG chain, authentication, metrics, and integration.

## Project Structure

```
├── main.py                          # FastAPI app with auth + metrics
├── auth.py                          # JWT authentication (PBKDF2 + HS256)
├── metrics.py                       # Prometheus metrics (zero-dep)
├── document_processor.py            # Text extraction & semantic chunking
├── vector_store.py                  # Keyword search engine
├── rag_chain.py                     # Answer generation & guardrails
├── test_rag.py                      # 58-test suite
├── Dockerfile                       # Multi-stage production build
├── docker-compose.yml               # App + Prometheus + Grafana
├── requirements.txt
├── .github/workflows/ci-cd.yml      # GitHub Actions pipeline
├── aws/
│   ├── ecs-task-definition.json     # ECS Fargate task config
│   └── setup-aws.sh                 # One-time AWS infra setup
└── monitoring/
    ├── prometheus.yml               # Prometheus scrape config
    └── grafana/
        ├── provisioning/            # Auto-provision datasource + dashboard
        └── dashboards/
            └── rag-dashboard.json   # Production Grafana dashboard
```

## Author

Dhanya Sri Cherukuri – [@dhanyasc](https://github.com/dhanyasc)
