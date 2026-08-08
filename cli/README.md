<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# cli/

**Purpose:** The command-line entry point.

**Contains:** Command definitions, argument parsing, and output formatting.

**Does not contain:** Business logic. The CLI is a thin adapter over the engine; anything worth testing belongs somewhere it can be tested without a subprocess.

**Governing rule:** Every documented claim must be regenerable by a command (Constitution Article 13). This package is where those commands are defined.
