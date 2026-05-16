# Security Policy

KVLens is a research codebase intended for studying KV-cache eviction strategies. It is not a production inference engine and has no service surface.

## Reporting a vulnerability

If you find a security issue (e.g., a credential leak in commit history, a dependency CVE that affects example scripts, or unsafe code in the engine), please **do not open a public issue**. Instead, use GitHub's private vulnerability reporting:

https://github.com/gcgarriga/KVLens/security/advisories/new

I'll respond within a reasonable window. There is no SLA.

## Supported versions

Only the latest tag on `main` is supported. Older tags (`v0.1`, `v0.1.1`) are preserved for reproducibility of the original study but will not receive patches.
