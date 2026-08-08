<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# plugins/

**Purpose:** Scientific domains. The only place in this repository where science lives.

**Contains:** One package per domain, each declaring what its observations are, what hypotheses look like, what makes one admissible, what the trivial baseline is, and how its data may honestly be split.

**Does not contain:** Anything the engine depends on. Dependencies flow one way: plugins know the contract, the engine does not know plugins.

**Governing rule:** Adding a domain requires no change to `engine/`. When it does, the interface is wrong and the interface gets fixed — the special case is not added (Constitution Article 15).
