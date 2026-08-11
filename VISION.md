# Scientific Discovery Engine

**A platform for computational scientific discovery**

*Founding document · v0.0.0*

---

## 1. The problem we are actually solving

Consider what happens when a researcher sets out to discover a governing equation from data.

They obtain observations. They clean them. They decide how to split them into training and evaluation sets. They choose a search method. They define what counts as a good hypothesis. They check whether the discovered expression is physically admissible. They compare against baselines. They produce a table and a figure. They write the paper.

Roughly one tenth of that work is the contribution. The other nine tenths is infrastructure — and it is written from scratch, by hand, for every single paper.

This is not a criticism of researchers. It is a structural fact about how computational science is currently practiced, and it has three consequences that compound.

**Results are not comparable.** When two papers report equation-recovery rates on the same benchmark, they are almost never measuring the same thing. One holds out a random 20%; another extrapolates outside the training range. One counts a match if the expression is symbolically equivalent after simplification; another accepts numerical agreement within tolerance. One reports the best of ten seeds; another reports the median. Both numbers are honest. Neither can be placed beside the other, and the field has no mechanism that would notice.

**Methods are not separable from their harnesses.** A promising idea arrives welded to the loader, the splitter, the scorer, and the plotting script it was born with. Testing it on your data means porting it or reimplementing it, and reimplementation quietly changes results. So good ideas propagate slowly, and bad ideas persist because refuting them is more work than ignoring them.

**Nothing accumulates.** Each paper leaves behind a repository that runs on one machine, for one dataset, until a dependency moves. The specific result may survive in the literature; the capability does not survive anywhere. The next researcher starts from the same blank file.

The waste is real, but the deeper cost is epistemic. A field where every result is produced by bespoke, unaudited, unreproducible infrastructure is a field that cannot efficiently tell which of its ideas are true.

---

## 2. Why the same code keeps getting rewritten

The obvious reply is: write a library. People have. Excellent libraries exist for symbolic regression, for sparse identification of dynamics, for Bayesian model selection.

They do not solve this problem, and the reason is worth stating precisely.

A library gives you a *component*. It answers "how do I run this method?" It does not answer "how do I compare this method against another one, under identical conditions, with results I can regenerate next year." The moment you want the second thing, you are back to writing a harness — and everyone writes a different one.

The gap is not a missing library. It is a **missing interface**: a stable, documented contract that separates the parts of scientific discovery that are genuinely domain-specific from the parts that are not.

That separation turns out to be sharper than it first appears.

**Domain-specific** — what an observation is, what units the variables carry, which expressions are physically admissible, what the trivial baseline is, which split respects the data's structure, what a domain expert would call a discovery.

**Not domain-specific** — running a search to convergence, tracking a Pareto front over accuracy and complexity, managing seeds, recording provenance, executing ablations, comparing methods, generating a results table, guaranteeing that rerunning produces the same numbers.

Every discovery project reimplements the second list. Almost none of it needed to be rewritten.

---

## 3. The thesis

> **The scientific discovery loop is domain-independent infrastructure. Separating it from the science is what makes results comparable, and comparability is what lets a field accumulate knowledge instead of anecdotes.**

The Scientific Discovery Engine exists to build that separation and hold it.

The core carries no scientific content whatsoever. It does not know what an equation is, what a unit is, what physics or biology or finance are. It knows the *shape* of a discovery problem: there are observations; there is a way to propose hypotheses; there is a way to score them; there is a way to reject the inadmissible ones; there is a way to rank what survives and report it.

Everything scientific lives in plugins, behind that contract.

This is a strong claim and it is falsifiable. If a domain arrives that cannot be expressed through the interface without special-casing the core, the thesis is wrong. We would rather find that out than avoid finding it out, which is why §9 exists.

---

## 4. What we are borrowing

Three systems solved a structurally similar problem, and we are deliberately copying their move.

**LLVM.** Its lasting contribution was not a superior optimizer. It was a stable intermediate representation that let language frontends and hardware backends evolve independently. A new language written against the IR inherits every backend. A new chip that implements the IR inherits every language. The value sits in the interface, not in any component behind it — and it compounds with each addition.

**pytest.** It has no idea what your code does. It knows what a test is, how to collect them, how to run them, how to report. That indifference is exactly why it works for every Python project written since. Domain ignorance in the core is not a limitation to apologize for; it is the property that makes generality possible.

**Kubernetes.** It separated *declaring* a workload from *placing* one. Anything that speaks the API gets scheduling, health checking, and rollout for free. The declaration is the contract; everything above and below it moves independently.

Each drew a line in the right place and then defended it, for years, against every reasonable-sounding request to make an exception. The line is the product. The discipline of not crossing it is what makes the line worth anything.

Our line is between the discovery loop and the science.

---

## 5. The second thesis: honesty is an infrastructure problem

There is a failure mode in computational science that gets discussed as an ethics problem and is actually an engineering problem.

A researcher runs a benchmark. They copy a number into a README. Months pass. The code changes — a default shifts, a seed is fixed, a baseline is retuned. The number in the README does not change, because nothing in the system connects them. It was correct when written and it is wrong now, and no one finds out until a reviewer runs the code.

Nobody lied. The infrastructure simply permitted a claim to detach from its evidence, and then time did the rest.

This generalizes. The ablation that was never actually run. The baseline that was never actually tuned. The variance across seeds that was never actually measured. The comparison against a method configured badly enough to lose. None of these require dishonesty. Each requires only that the easy path and the rigorous path differ, and that a deadline arrives.

**So we treat rigor as a property of the system rather than of the researcher.**

- Numbers in documentation are generated artifacts. A command regenerates every table, every figure, every reported statistic. Hand-typed numbers are defects.
- Every experiment records its configuration, git SHA, seed, resolved environment, and hardware. A result you cannot regenerate is not a result.
- Ablations are configuration, not code. Turning a component off is a flag, so it costs nothing to run — and something that costs nothing to run gets run.
- Baselines are first-class citizens with the same tuning budget as the proposed method. A method that only beats an untuned baseline has not been shown to beat anything.
- Negative results are reported in the same table, at the same prominence, without softening.

The aim is that **the honest path is the path of least resistance.** Integrity that depends on remembering to be careful is integrity that fails under deadline. Integrity built into the tooling holds when no one is watching, which is when it matters.

We hold this standard against ourselves first. This project's own reported results are subject to it without exception, including when they are unflattering.

---

## 6. The architecture that follows

The thesis determines the structure. Little of it is arbitrary.

**A core that knows no science.** `engine/` orchestrates the discovery loop and contains no scientific concept, no domain name, and no import from any plugin. Enforced in CI, not by convention — a rule that depends on reviewer attention is a rule that erodes.

**Plugins as the only place science lives.** A plugin declares what its observations are, what hypotheses look like, what makes one admissible, what the trivial baseline is, and how the data may honestly be split. Adding a domain requires no change to the core. When it does, the interface is wrong, and we fix the interface rather than adding the special case.

**Algorithms behind a separate interface.** How you *search* is orthogonal to what you are searching *for*. Genetic programming, sparse regression, neural-guided search, and language-model-proposed candidates are interchangeable behind one contract. Any algorithm must be removable without the engine noticing.

**Constraints as a distinct concern.** Dimensional consistency, monotonicity, boundary behavior, conservation — these prune the search space and they are not properties of any algorithm. They belong on their own, composable and reusable across every method.

**Validation independent of generation.** Whether a hypothesis is admissible cannot depend on how it was produced. Separating these is what makes cross-method comparison meaningful rather than circular.

**Experiments as records, not scripts.** An experiment is a declarative object with a configuration, an environment, a seed, and a set of outputs — replayable by construction. Not a script that happened to run once on a laptop that has since been reformatted.

**Reporting as a pipeline stage.** Tables and figures are generated from recorded experiments by code that lives in the repository. This is what makes §5 mechanically true rather than aspirational.

---

## 7. What becomes possible

If the separation holds, several things stop being projects and start being commands.

**Method comparison becomes trivial.** Run five discovery algorithms on the same domain, under identical splits, seeds, and scoring, and get one table. Today this is a paper's worth of engineering. It should be an afternoon.

**Domain transfer becomes trivial in the other direction.** A method validated on physics runs unchanged on biochemistry, materials, or finance. Its generality becomes a measurable property rather than a claim in a discussion section.

**Ablation becomes routine.** When disabling a component is a config flag, the honest question — *which parts of my method actually contribute?* — gets asked, because asking is free.

**Constraints become shared infrastructure.** Someone writes dimensional analysis once, correctly, with tests. Everyone downstream gets it. This is the LLVM compounding effect applied to scientific priors, and it is where the leverage really is.

**Reproduction becomes the default.** A result from two years ago regenerates from a recorded configuration, because that is the only way results were ever produced.

**Negative results become cheap to publish.** Much of what makes negative results scarce is that they cost as much engineering as positive ones and carry less reward. Drive the engineering cost toward zero and the calculus changes.

None of this requires a new algorithm. It requires that the boring nine tenths be built once, properly, by someone willing to treat it as the actual work.

---

## 8. What we refuse to build

A platform is defined by its refusals at least as much as its features.

**Not a wrapper around one library.** If SDE becomes a convenient way to call one particular search backend, it has failed, no matter how popular that backend is.

**Not AutoML.** We are not searching for the model with the best score. We are supporting the discovery of *interpretable, admissible, scientifically meaningful* structure. Those objectives point in different directions, and conflating them produces systems that optimize a metric while answering no question.

**Not a domain application.** SDE is not a climate tool, a physics tool, or a biology tool. Domains demonstrate the platform; they never become its identity. A repository whose name promises a domain its code does not contain has already lost this argument.

**Not a notebook collection.** Notebooks are for exploration. They are not reproducible, not testable, not composable, and not a deliverable here.

**Not a place for prototypes.** Prototypes are valuable and they live on branches. The main branch holds work that meets the standard.

**Not a system that flatters its own results.** Where the evidence and the narrative disagree, the narrative is what changes.

---

## 9. How we will know we were wrong

A vision document that cannot be refuted is marketing. Here are the specific observations that would falsify this one.

**The core acquires scientific content.** If `engine/` ends up containing units, physical constants, domain names, or a branch on which plugin is loaded, the separation did not hold. The interface was insufficient and we papered over it.

**Adding a domain requires core changes.** The plugin contract is the whole claim. If the third domain needs the engine modified, one plugin was never a demonstration of generality.

**The abstraction costs more than it returns.** If expressing a real problem through the interface is meaningfully harder than writing it directly, researchers will write it directly, correctly. Adoption is the measurement; there is no argument that overrides it.

**Reproduction fails.** If a recorded experiment does not regenerate its recorded metrics, the central promise is broken and everything downstream of it is worth less than it appears.

**Results stay incomparable anyway.** If two methods run through SDE still cannot be honestly compared — because the differences that matter live somewhere the contract does not reach — then the line was drawn in the wrong place.

We would rather discover any of these early and say so plainly than defend a thesis past the point of evidence. That posture is the same one we ask of the science this platform is meant to serve, and it would be incoherent to demand it of others while exempting ourselves.

---

## 10. Where we are

Version 0.0.0. The foundation, and honestly not much else.

This document, the [Bootstrap Specification](BOOTSTRAP.md), and the [Constitution](CONSTITUTION.md) exist before the implementation they govern — deliberately, because the central claim of this project is architectural, and an architectural claim made after the fact is indistinguishable from a description.

What follows, in order: the repository foundation, then roughly twenty-five RFCs establishing subsystem architecture, then implementation against engineering tasks precise enough to execute without invention.

The measure at v1.0 will not be feature count. It will be whether a researcher who has never spoken to us can add their scientific domain, get a result they trust, and reproduce it a year later.

---

## 11. An invitation

If you have written the same benchmark harness three times, in three repositories, for three papers — this is for you.

If you have tried to compare your method against a published one and found that reimplementing it honestly would take a month — this is for you.

If you have ever had a reviewer ask for an ablation you knew was correct but could not run in the time available — this is for you.

The scientific contribution in computational discovery is real, and it is small, and it is currently buried under nine tenths infrastructure that nobody wanted to write and everybody wrote anyway. We think that infrastructure should exist once, be excellent, be shared, and get out of the way.

That is the whole idea. Everything else in this repository is execution.

> **Build infrastructure that accelerates scientific discovery, not just another machine learning application.**
