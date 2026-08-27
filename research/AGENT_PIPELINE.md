# The role-rotating research pipeline

This describes how research/GIT_WORKFLOW.md's process runs in practice when
**one** Claude Code session carries it out, picking up whichever of three
roles — researcher, implementer, reviewer — GitHub state says is next, and
coordinating only through that state (issues, branches, PRs, labels), never
through in-session memory of "what I decided as researcher an hour ago."
Nothing here changes GIT_WORKFLOW.md's actual policy; it says how that
policy gets executed by one party that plays three parts in rotation.

This document originally described three separate sessions running these
roles concurrently. That model is retired — see "Why one session now"
below — but the role definitions and the loop-readiness decision logic were
written to be role-agnostic from the start, so nothing about them changed.

## Why GitHub state instead of live coordination

Every handoff between roles has to be a durable, auditable artifact — the
same reason DECISION_LOG.md exists ("this prevents the repository from
becoming dependent on the researcher's memory"). This matters even more now
that one session plays every role: without a hard rule to re-derive each
role's state from GitHub rather than from what the session remembers writing
as a different role minutes earlier, the roles quietly merge into one voice
and the separation of concerns (proposer / builder / independent checker)
stops meaning anything. Issues and PR comments are the coordination layer,
full stop — not context carried over in the conversation.

A consequence worth being explicit about: this also means no role needs the
others to have "just run." The pipeline degrades gracefully to "whichever
role GitHub state says is next," which is what makes the single-session
rotation in this document possible, and what made the old three-session
version possible before it.

## Why one session now

Three independent sessions were the safest way to get real role separation:
a researcher session literally could not see the implementer's reasoning,
so its issue-filing couldn't be biased by knowing how the fix would land.
Running one session through all three roles trades that hard separation for
operational simplicity — one thing to schedule, one context window, one
place to look. The role instructions below are unchanged; what changes is
**how the session decides which role to play this tick**, and an explicit
rule against a role trusting its own prior-role reasoning (see "Guarding
role separation within one session").

## Roles

### Researcher

Finds the next real gap and turns it into exactly one properly-scoped issue.

1. Read `README.md`'s "Open research questions / next experiments" and
   `research/DECISION_LOG.md` first — don't re-propose something already
   listed, already ruled out, or already ABANDONED per a past decision.
2. Read `research/AUDIT_METHODOLOGY.md` if the question touches the audit's
   statistical design (margins, TOST, the oracle ceiling).
3. File one issue using the "Research question" template
   (`.github/ISSUE_TEMPLATE/research-question.yml`), filling in every field —
   the issue body *is* the spec, not a summary of one filed elsewhere.
4. Pick the branch-type dropdown honestly. `research/` for something
   substantial; `experiment/` for one scoped run; don't inflate a small
   question to sound more important, and don't compress a substantial
   investigation into something that looks like a quick check.
5. Leave it labeled `stage:proposed`. Do not implement anything, do not open
   a branch, do not touch code.

**Correctly scoped** means: answerable by one branch, with one clear
hypothesis, not a research program disguised as a single issue. "Does DARTS
beat budget-matched random search on NATS-Bench CIFAR-10" is scoped.
"Validate the whole NAS pivot" is not — that's the kind of thing that becomes
several issues.

### Implementer

Claims one issue, answers it, opens a PR.

1. `gh issue list --label stage:proposed` to see what's open.
2. Claim it: `gh issue edit <n> --add-assignee @me` and swap the label to
   `stage:claimed` — this is what stops two implementer sessions from
   duplicating the same work silently.
3. Create the branch with the prefix the issue specified:
   `git checkout -b research/<topic>` (or `experiment/`, `analysis/`,
   `method/`, `reproduction/`).
4. First commit: copy the issue body into `SPEC.md` at the root of wherever
   the work lives, verbatim, per `research/GIT_WORKFLOW.md` and
   `research/SPEC_TEMPLATE.md`'s own instructions. Do not silently improve or
   reinterpret the spec while copying it — if it's wrong, comment on the
   issue and get it fixed there first.
5. Implement, run the actual audit/experiment, record results as real
   artifacts (not just claims in a commit message) — see
   `research/AUDIT_METHODOLOGY.md` for what an `AuditReport`'s provenance
   fields capture and why.
6. Open the PR. The PR template auto-populates; fill in every section through
   "Decision (implementer's self-assessment)" honestly, including if the
   honest answer is REVISE or ABANDON — a self-assessed ABANDON is still a
   useful PR, it documents why an approach didn't pan out.
7. Swap the issue's label to `stage:in-review` (or close the issue and
   reference the PR, whichever the repo's convention ends up being — pick one
   and stay consistent).

**Never:** invent scope beyond what the issue specifies (if the work reveals
a bigger or different question, that's a new issue for the researcher, not a
scope-creep on this branch); merge anything; mark your own PR approved.

### Reviewer

Independent check against `research/GIT_WORKFLOW.md`'s actual criteria, not a
generic code review.

1. `gh pr list --label stage:in-review`.
2. Read the PR, the linked issue, `SPEC.md` on the branch, and the actual
   results artifacts — not just the PR description's claims about them.
3. Check CI is green.
4. Go through the nine merge criteria and the "when not to merge" list in
   `research/GIT_WORKFLOW.md` explicitly — the PR template's checklist exists
   so this isn't done from memory. Any unchecked box gets a comment, not a
   silent pass.
5. "Discuss and refine," concretely, without live messaging: leave PR review
   comments requesting specific changes; the implementer session picks those
   up next time it runs, pushes, and the reviewer re-reviews. This can cycle
   as many times as it needs to — it's just normal GitHub PR review, running
   asynchronously across sessions that may be hours or days apart.
6. Post the decision: MERGE / ARCHIVE / REVISE / ABANDON / REPRODUCE, with
   the reasoning `research/DECISION_LOG.md`-style — specific enough that
   someone reading only this comment later understands why.
7. If MERGE: label the PR `stage:approved-pending-merge` and stop. **Do not
   merge.** A human merges.
8. If ARCHIVE / ABANDON / REPRODUCE: label accordingly, and add an entry to
   `research/DECISION_LOG.md` — the reviewer is the one who has just read the
   full evidence, so this is the right point to record why, not something to
   leave for later.

**Never:** push a code change to the branch (comment instead); merge under
any circumstance; wave a criterion through without checking it because the
result looked good.

## What the human still does

Reads reviewer-approved PRs and clicks merge. Reads ARCHIVE/ABANDON
decisions and can override them (nothing here removes human judgment, it
just means the human isn't needed for every intermediate step).

## Role selection (one tick, one role)

Each tick, before doing anything else, run these `gh` queries in order and
play the **first** role whose condition is true. Stop after finishing that
role's one unit of work (one PR reviewed, one issue implemented, one issue
filed) — do not chain into a second role in the same tick, even if its
condition also holds. This priority order clears the pipeline
downstream-first, so work already in flight finishes before new work starts:

1. **`gh pr list --label stage:in-review`** returns anything → play
   **Reviewer** on the oldest PR.
2. Else, **`gh issue list --label stage:proposed --assignee ""`** returns
   anything → play **Implementer** on the oldest unclaimed issue.
3. Else, **`gh issue list --label stage:proposed`**'s count is below
   threshold N (default N = 3 — enough runway that the implementer role
   never stalls waiting on the researcher role's next tick, small enough
   that stale proposals don't pile up unimplemented) → play **Researcher**
   and file exactly one issue.
4. Else, idle: log which condition was checked and why nothing fired, then
   stop until next tick.

Follow the matching role's numbered instructions above exactly — the role
selection only decides *which* numbered list to execute, it does not change
what's in them.

## Guarding role separation within one session

The three roles used to be enforced by being different sessions that
literally could not see each other's reasoning. One session rotating roles
has to enforce that separation by rule instead of by architecture:

- **Re-derive state from GitHub, not from this conversation's memory.**
  When playing Reviewer, read the PR, the linked issue, and `SPEC.md` on the
  branch as if seeing them for the first time — do not reuse
  implementer-role reasoning from earlier in the same session as a shortcut.
  If the implementer-role work happened this session, that is exactly the
  case most likely to produce a rubber-stamp review, because the reviewer
  role already "knows" the conclusion is right.
- **Never review or approve your own PR**, regardless of which session
  wrote it. If the PR under review at step 1 was opened by this same
  session's implementer-role turn, the review still has to independently
  re-derive MERGE / ARCHIVE / REVISE / ABANDON / REPRODUCE from the nine
  merge criteria — an approval that amounts to "I already decided this was
  good when I wrote it" is not a review.
- **Never let the researcher role scope an issue around a solution the
  session already has in mind.** The researcher role's job is to find a
  real gap per README.md / DECISION_LOG.md, not to pre-stage easy work for
  the implementer role it's about to become.
- If a tick's role selection would have this session review its own
  immediately-prior work (Implementer this tick, and next tick's Reviewer
  query would pick up that same PR), that's expected and fine — the rules
  above are what keep it honest, not avoiding the sequence.

## Running this as a loop

Role selection above is exactly what a scheduled loop polls each tick — see
the harness's `/loop` skill (dynamic pacing, no fixed interval needed) or a
cron. Nothing about the label taxonomy, templates, or role instructions
needs to change to run this way; only the three separate per-role loops
originally sketched here collapsed into the single decision tree above.
