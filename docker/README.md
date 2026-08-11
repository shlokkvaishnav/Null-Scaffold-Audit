
# docker/

**Purpose:** Container definitions for reproducible execution.

**Contains:** Dockerfiles and compose definitions, with every base image pinned by sha256 digest.

**Does not contain:** Base images referenced by tag. A tag is mutable and silently breaks reproducibility across build dates.

**Governing rule:** reproducible builds. Given the same inputs, a build produces the same outputs — which requires the base image to be the same image, not merely the same name.
