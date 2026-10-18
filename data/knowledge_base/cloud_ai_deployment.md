# Cloud AI Deployment and MLOps

## From prototype to deployment
A deployed model must start reliably, respond within a budget, and fail loudly. Containerize the service with Docker: pin dependencies, copy only what is needed, run as a non-root user, and define a health check. Cloud platforms (AWS ECS or Lambda, Azure App Service, Google Cloud Run) all run the same container - start with the free tier and set billing alerts before experimenting.

## Serving patterns
Real-time APIs suit interactive features; batch scoring suits nightly reports; queues (Celery, SQS) decouple slow inference from web requests. For LLM-backed features, isolate provider calls behind a timeout and a fallback path so an outage degrades gracefully instead of breaking the product - the same design used by this application's optional AI assistant.

## Configuration and secrets
Configuration lives in environment variables; secrets never enter the repository. Provide a .env.example with placeholder names, keep real credentials in .env (git-ignored), and give every secret a clear rotation story. In logs, mask emails, phone numbers, and tokens; log correlation IDs instead of personal data.

## Monitoring and drift
Track latency, error rate, and cost per request from day one. For models, log input distributions and compare them against training data: data drift (users change) and concept drift (the relationship changes) both silently degrade accuracy. Alert on metric shifts, not just crashes.

## CI/CD for ML
Every push should run linting, unit tests, and evaluation checks on a sample; every merge should build the container; deployments should be reversible in one command. Keep model artifacts versioned (MLflow, DVC, or the platform registry) so any prediction can be traced to a specific data version, code commit, and model build. Reproducibility is what separates an experiment from a system.
