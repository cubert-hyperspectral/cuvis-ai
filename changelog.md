# Changelog

## Unreleased (vs `cuvis-ai-v2`)
- Introduced the gRPC service stack (`cuvis_ai/grpc/*`) with proto definitions, session management, callbacks, helpers, and the production server wiring.
- Added generated protobuf stubs plus Buf config, containerization assets (`Dockerfile`, `docker-compose.yml`, `.env.example`), and onboarding updates in `README.md`/`CONTRIBUTING.md`.
- Documented the gRPC surface and deployment (`docs/api/grpc_api.md`, `docs/deployment/grpc_deployment.md`) alongside the detailed blueprint/implementation notes under `docs_dev/cubert/ALL_4917/*`.
- Expanded automated coverage with comprehensive gRPC and pipeline tests (`tests/grpc_api/*`, updated pipeline/training tests) and supporting fixtures.
- Refined pipeline/node logic (selector, losses, metrics, canvas, dataset handling, training config) and removed outdated torch example scripts in favor of the new API flows.
