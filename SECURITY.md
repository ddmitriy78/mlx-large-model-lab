# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's security-advisory
workflow for this repository. Do not include exploit details, private prompts,
credentials, or sensitive telemetry in a public issue.

Include:

- the affected version or commit
- the relevant server, Control Center, CLI, or storage component
- reproduction steps using non-sensitive data
- impact and any suggested mitigation

## Deployment boundary

MLX Large Model Lab binds its model API and Control Center to loopback by
default. Either service can require the same local Bearer API key, managed from
the Control Center and stored with owner-only permissions. This shared-key
option is not a production identity system: the project does not provide
multi-user authorization, tenant isolation, TLS termination, or an
internet-facing deployment profile. Operators who change a bind address are
responsible for adding appropriate network and application controls.

Model artifacts execute through the MLX and Hugging Face ecosystem. Review
upstream model licenses and code requirements before downloading or enabling
remote/custom model code.
