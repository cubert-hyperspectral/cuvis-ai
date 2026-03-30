!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

## Guides

<div class="grid cards" markdown>

-   :material-puzzle: **[Architecture](architecture.md)**

    ---

    Plugin architecture, loading mechanisms, caching, and structure

-   :material-cog: **[Internals](internals.md)**

    ---

    Node registration, isolation, versioning, security, and lifecycle

-   :material-hammer-wrench: **[Development Quick Start](dev-quickstart.md)**

    ---

    Create plugins from scratch: project structure, node implementation, testing

-   :material-package: **[Packaging](packaging.md)**

    ---

    Package, distribute, and publish plugins via Git or PyPI

-   :material-package-variant: **[Loading Plugins](loading.md)**

    ---

    Find, install, and load plugins from registries

-   :material-play-box: **[Using Plugin Nodes](using-nodes.md)**

    ---

    Use plugin nodes in pipelines and manage versions

</div>

## Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection

See the [central plugin registry](../../configs/plugins/registry.yaml) for all registered plugins, or [contribute your own](../development/contributing.md#plugin-contribution-workflow).
