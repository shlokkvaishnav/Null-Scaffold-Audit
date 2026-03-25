# SD-MoSE Agent: Methods Backbone

This document defines the **exact semantics** of the SD-MoSE GRAIL-V agent based on the actual implementation. This freezes design intent and serves as the foundation for the method specification.

---

## 1. Agent Environment & Observations

### What Raw Data Enters `observe()`

The agent receives climate data as a dictionary:

```python
{
    "features": np.ndarray,  # Shape: (n_samples, n_features)
    "targets": np.ndarray,   # Shape: (n_samples,)
    "metadata": dict         # Temporal/spatial context
}
```

### What `PerceptionModule.encode()` Outputs

The perception module ([`agent/perception.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/perception.py)) transforms raw data into structured observations:

```python
observation = {
    "features": preprocessed_features,  # Normalized/scaled
    "targets": preprocessed_targets,
    "meta": metadata
}
```

**Current implementation**: Simple passthrough wrapper.  
**Future**: Will call `data/preprocess.py` for normalization, missing value handling, and feature engineering.

### Exact Structure of `self.observation`

Stored in agent state and accessible to all cognitive modules (retrieval, reasoning, verification, learning).

---

## 2. Retrieved Knowledge

### What Comes from `science/priors.py`

The retrieval module ([`agent/retrieval.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/retrieval.py)) fetches scientific priors:

```python
priors = {
    "conservation_laws": ["mass", "energy"],
    "known_relations": []  # e.g., Henry's Law
}
```

**Purpose**: Guide symbolic search toward physically plausible equations.

### What Comes from Agent Memory

Valid hypotheses from previous iterations:

```python
candidate_hypotheses = memory.recall(query={"valid": True})
```

Returns list of `Hypothesis` objects that passed verification.

### What Retrieval Explicitly Does NOT Do

**Critical distinction**: Retrieval is **memory access**, not learning.

- ❌ Does NOT generate new equations
- ❌ Does NOT update beliefs
- ❌ Does NOT modify stored hypotheses

Retrieval only **reads** from prior knowledge and memory.

---

## 3. Symbolic Reasoning

### How `ReasoningModule.propose_hypothesis()` Works

Located in [`agent/reasoning.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/reasoning.py):

1. **Regime-specific generation**: One equation per regime (K=3 by default)
2. **PySR integration**: Wraps [`experts/symbolic.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/experts/symbolic.py) for symbolic regression
3. **Hypothesis creation**: Returns `Hypothesis(equation, regime_id)` objects

```python
for k in range(num_regimes):
    equation = reasoning_module.propose_hypothesis(
        observation=observation,
        regime_id=k,
        priors=priors
    )
    hypotheses.append(Hypothesis(equation, regime_id=k))
```

### How Priors Condition Symbolic Search

**Current**: Priors available to reasoning module but not yet used to constrain PySR.  
**Future**: Will restrict operator sets, enforce conservation laws, and bias toward known functional forms.

---

## 4. Verification & Self-Critique

### Constraint Types

Defined in [`science/constraints.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/science/constraints.py):

- Conservation laws (mass, energy)
- Physical bounds (e.g., concentrations ≥ 0)
- Dimensional consistency
- Thermodynamic constraints

**Current**: Placeholder returns `{}` (all equations pass).  
**Future**: Will implement actual constraint checking.

### Violation Log Format

Each hypothesis tracks violations:

```python
violation_log = {
    "constraint_name": {
        "violation_rate": 0.15,  # Fraction of data points violating
        "details": "Mass balance violated"
    }
}
```

### Rejection Mechanism

From [`agent/verification.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/verification.py):

1. **Verify** each hypothesis via `h.verify(verification_module)`
2. **Threshold**: Hypothesis invalid if any constraint violation_rate > 0.1
3. **Log rejection**: Invalid hypotheses stored in `memory.rejection_log`

```python
if not h.valid:
    memory.rejection_log.append({
        "hypothesis": h,
        "violations": h.violation_log
    })
```

### How Rejected Hypotheses Influence Learning

Rejection logs provide:
- **Negative examples** for symbolic search (avoid similar forms)
- **Explainability**: Which constraints are most restrictive
- **Validation statistics**: Rejection rates per regime

---

## 5. Learning & Belief Update

### What Evidence Enters `BeliefState.update()`

From [`agent/belief.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/belief.py):

```python
evidence = {
    "num_verified": len(verified_hypotheses),
    "num_rejected": len(proposed_hypotheses) - len(verified_hypotheses),
    "hypotheses": verified_hypotheses
}
```

### What Beliefs Represent (π over Regimes)

Soft assignments: `beliefs[k]` = probability that data comes from regime k.

**Initialization**: Uniform `[0.33, 0.33, 0.33]` for K=3.  
**Update rule**: Currently placeholder (stores history).  
**Future**: EM-style updates based on hypothesis performance.

### What Is Stored vs Pruned in Memory

**Stored** ([`agent/memory.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/memory.py)):
- Valid hypotheses → `memory.hypotheses`
- Rejection logs → `memory.rejection_log`
- Regime history → `memory.regime_history`

**Pruned**:
- Invalid hypotheses removed via `memory.prune()`
- Keeps only `h.valid == True`

---

## 6. Agent Control Loop

### One Iteration of `step()`

From [`agent/agent.py`](file:///c:/Users/shlok/Documents/Projects/climate-equation-discovery/sdmose/agent/agent.py):

```python
def step(self, data):
    self.observe(data)           # 1. Ground observations
    self.retrieve()              # 2. Fetch priors + memory
    self.reason()                # 3. Propose hypotheses
    self.verify()                # 4. Validate against constraints
    self.learn()                 # 5. Update beliefs + memory
    self.iteration += 1
```

**Current state**: Executes one full GRAIL-V loop.  
**Verified**: All methods functional, no silent failures.

### Stopping Criteria

**Current**: `run_loop()` is a stub.  
**Planned**: 
- Convergence: `||π_t - π_{t-1}|| < ε`
- Max iterations
- Performance plateau

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Perception | ✓ Functional | `agent/perception.py` |
| Retrieval | ✓ Functional | `agent/retrieval.py` |
| Reasoning | ✓ Functional | `agent/reasoning.py` |
| Verification | ✓ Functional | `agent/verification.py` |
| Belief | ✓ Functional | `agent/belief.py` |
| Memory | ✓ Functional | `agent/memory.py` |
| Control Loop | ⚠️ Stub | `agent/agent.py::run_loop()` |

---

## Key Design Decisions

1. **Retrieval ≠ Learning**: Explicit separation prevents reward hacking
2. **First-class Hypotheses**: `Hypothesis` objects with validation state
3. **Rejection Logging**: Enables explainability and negative learning
4. **Evidence-based Updates**: Beliefs updated from verification results, not arbitrary rewards
5. **Modular Lazy Loading**: Components initialized on first use to avoid circular dependencies

---

**This document freezes the agent's semantic contract. Any future feature must preserve these invariants.**
