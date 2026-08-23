# The three-session research pipeline

This describes how research/GIT_WORKFLOW.md's process runs in practice when
three separate Claude Code sessions — researcher, implementer, reviewer —
carry it out, coordinating only through GitHub state (issues, branches, PRs,
labels), not through live messaging between sessions. Nothing here changes
GIT_WORKFLOW.md's actual policy; it says how that policy gets executed by
three independent parties who may never run at the same time.

## Why GitHub state instead of live coordination

Every handoff between roles has to be a durable, auditable artifact — the
same reason DECISION_LOG.md exists ("this prevents the repository from
becoming dependent on the researcher's memory"). A live conversation between
two sessions vanishes unless someone copies it into a PR anyway, so it never
was the real coordination layer. Issues and PR comments are.

A consequence worth being explicit about: this also means no role needs the
other two to be running right now. The researcher can file five issues today;
the implementer can pick one up next week; the reviewer can process a PR
whenever it's ready. The pipeline degrades gracefully to "whoever's turn it
is next," which is also what makes it loop-ready later (see the bottom of
this document).

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
just means the human isn't needed for every intermediate step). Decides when
to start a fresh researcher/implementer/reviewer cycle.

## Loop-readiness (design note, not built yet)

The label states above are exactly what a scheduled loop would poll:

- A researcher loop checks whether `stage:proposed` count is below some
  threshold N; if so, look for the next gap and file one issue, then stop
  until next tick.
- An implementer loop checks for any `stage:proposed` issue with no
  assignee; claims and works the oldest one; stops until next tick.
- A reviewer loop checks `stage:in-review` PRs; processes the oldest one;
  stops until next tick.

Each of those is a single `gh` query plus the same instructions already
given to the manual sessions above — nothing about the label taxonomy,
templates, or process needs to change to run this way later. Whether to
actually schedule it (this harness's `/loop`, or a cron) is a separate,
later decision — this section exists so that decision doesn't require a
redesign when it's made.
