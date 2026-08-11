<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# sdk/

**Purpose:** The stable surface that external plugin and algorithm authors build against.

**Contains:** Public types, base classes, contract definitions, and the helpers an external author needs — and nothing they should not depend on.

**Does not contain:** Internals. Anything exported here is a compatibility commitment.

**Governing rule:** Success criterion 1 (BOOTSTRAP.md section 19) is a researcher outside this project adding a domain without modifying `engine/`. This package is what makes that possible, and breaking it breaks the claim.
