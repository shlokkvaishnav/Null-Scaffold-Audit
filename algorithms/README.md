<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# algorithms/

**Purpose:** Discovery algorithm implementations, behind a stable interface.

**Contains:** Search strategies — genetic programming, sparse regression, neural-guided search, language-model-proposed candidates — each interchangeable behind one contract.

**Does not contain:** Domain knowledge. How you search is orthogonal to what you are searching for.

**Governing rule:** Constitution Article 6. Any algorithm must be removable without the engine noticing. If the engine needs to know which algorithm is running, the interface is wrong.
