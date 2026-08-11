<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# engine/

**Purpose:** The domain-independent core. It orchestrates the discovery loop and knows no science.

**Contains:** The plugin contract, the registry, the orchestrator, configuration schemas, and the workflow that sequences them.

**Does not contain:** Any scientific concept whatsoever. No units, no physical constants, no domain names, no import from `plugins/`, no branch on which plugin is loaded.

**Governing rule:** Constitution Article 5. This is the project's central architectural claim — violating it does not degrade the architecture, it refutes it. Enforced in CI by `tools/check_domain_independence.py`, not by convention.
