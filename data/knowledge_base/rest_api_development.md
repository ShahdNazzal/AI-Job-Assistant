# REST API Development for ML Services

## Why models need APIs
A model in a notebook helps no one. Wrapping a model in a REST API turns it into a service other systems can call: a web app, a cron job, or another team's pipeline. In Python, FastAPI is the current favorite for ML serving because it validates request bodies with Pydantic, generates OpenAPI documentation automatically, and supports asynchronous handlers.

## Designing clean endpoints
Use nouns for resources and HTTP verbs for actions: POST /predict with a JSON body, GET /health for liveness checks. Return appropriate status codes - 200 for success, 422 for validation errors, 500 for unexpected failures - and put error details in a predictable JSON shape. Version the API from day one (/v1/predict) so changes never break callers silently.

## Validation and error handling
Validate every input at the boundary with Pydantic models: types, ranges, and required fields. Reject empty or oversized payloads early. Catch model-level failures and translate them into friendly messages; never leak stack traces or environment variables to clients. Log enough to debug - request IDs, latency, feature values - but never log secrets or personal data.

## Performance basics
Measure before optimizing. Typical ML API costs are model loading (do it once at startup), per-request inference, and serialization. For CPU-bound inference, keep request handling synchronous and run multiple worker processes; for I/O-bound work, async helps. Add timeouts everywhere, including calls to external LLM providers, so a slow dependency cannot hang your service.

## Testing an API
Write unit tests for business logic with pytest, then integration tests that call the API with a test client: valid requests, invalid requests, and edge cases like empty text. A service without tests is a prototype; with tests, it is engineering.
