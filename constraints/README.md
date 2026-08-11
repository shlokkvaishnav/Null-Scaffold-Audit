<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# constraints/

**Purpose:** Search-space constraints that prune candidate hypotheses before they are scored.

**Contains:** Dimensional consistency, monotonicity, boundary behavior, conservation laws — composable and reusable across every algorithm and every domain.

**Does not contain:** Anything specific to one search method. A constraint is a property of the problem, not of the searcher.

**Governing rule:** Constraints are written once, correctly, with tests, and everything downstream inherits them. This is where the compounding leverage is (VISION.md section 7).
