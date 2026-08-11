
# configs/

**Purpose:** Versioned experiment and run configurations.

**Contains:** Declarative configuration describing what to run — never how to run it.

**Does not contain:** Secrets, absolute paths, or machine-specific values.

**Governing rule:** Ablations are configuration, not code (VISION.md section 5). Turning a component off is a flag, so it costs nothing to run — and something that costs nothing to run gets run.
