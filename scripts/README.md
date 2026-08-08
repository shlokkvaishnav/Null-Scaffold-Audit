<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# scripts/

**Purpose:** Operational entry points.

**Contains:** Thin runnable wrappers for reproducing benchmarks, regenerating artifacts, and operational maintenance.

**Does not contain:** Logic that belongs in a tested package. A script is an entry point, not a home.

**Governing rule:** Anything a script does that is worth relying on is worth testing, which means it belongs in a package the script calls.
