
# validators/

**Purpose:** Determining whether a hypothesis is admissible, independently of how it was generated.

**Contains:** Validity checks applied to a candidate after search produces it.

**Does not contain:** Any reference to the generating algorithm.

**Governing rule:** Whether a hypothesis is admissible cannot depend on how it was produced. Separating validation from generation is what makes cross-method comparison meaningful rather than circular (VISION.md section 6).
