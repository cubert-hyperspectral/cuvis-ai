# Deployment

How to run cuvis-ai remotely and integrate it into other services.
Deployment is the umbrella; gRPC sits inside it as the wire protocol
that powers the deployment story.

If you only need to run a pipeline locally against a `.cu3s` file, use
[`restore-pipeline`](../workflows/restore-pipeline.md) instead — no
deployment required.

## Start here

<div class="grid cards" markdown>

-   :material-information: **[Architecture](overview.md)**

    ---

    What the cuvis-ai gRPC service is, how it's organised, and the typical request/response lifecycle.

-   :material-cloud-upload: **[Deployment Guide](grpc-deployment.md)**

    ---

    Prerequisites, install, TLS, Docker, Kubernetes, monitoring, security, troubleshooting.

-   :material-cog-transfer: **[gRPC Workflow](grpc-workflow.md)**

    ---

    Operational walk-through: starting the server, connecting from a client, training and inferring remotely.

</div>

## API reference

<div class="grid cards" markdown>

-   :material-account-key: **[Session Management](api/session.md)**

    ---

    Creating and tearing down sessions.

-   :material-tune: **[Configuration API](api/config.md)**

    ---

    Server- and pipeline-side configuration.

-   :material-graph: **[Pipeline API](api/pipeline.md)**

    ---

    Load, inspect, and run pipelines remotely.

-   :material-school: **[Training & Inference](api/training-inference.md)**

    ---

    Train pipelines and run inference over gRPC.

-   :material-alert-circle: **[Types & Errors](api/types-errors.md)**

    ---

    Wire types and the error taxonomy.

</div>

## Client patterns

<div class="grid cards" markdown>

-   :material-pattern: **[Connections & Sessions](client-connections.md)**

    ---

    Connection management and session lifecycle patterns.

-   :material-cog-sync: **[Workflows & Error Handling](client-workflows.md)**

    ---

    Configuration, training, inference, and error-handling patterns.

-   :material-chart-timeline: **[Sequence Diagrams](sequence-diagrams.md)**

    ---

    Visual workflows for the major operations.

</div>
