# G4 mid-run learning — Step 2

Step 2 turns every evaluated candidate into reusable search knowledge while the
campaign is still running.

## Immutable judge

`JudgeLaw` is frozen and content-addressed. The learner may change candidate
priority, weights, population, and mutation choices. It cannot weaken exactness,
legality, timing, confidence, or promotion requirements. A capsule created under
a different judge digest is refused.

## Boundary + Rule + Residue = Pattern

- **Boundary**: environment, domain, operation, and exact constraints.
- **Rule**: a candidate's reversible ACTG genome, features, and parameters.
- **Residue**: compile failure, illegality, inexactness, inadequate speed,
  insufficient paired wins, low confidence, or promotion.
- **Pattern**: repeated contextual feature/residue evidence strong enough to
  increase sampling, suppress a body, or direct mutation.

The learner updates after every experiment. The next candidate is therefore
chosen using all evidence gathered earlier in the same run.

## Contextual weights and interactions

Weights are conditioned on multiple transferable contexts:

- global;
- environment;
- domain;
- operation;
- environment/domain pair;
- domain/operation pair;
- each declared boundary constraint.

Feature pairs also receive interaction weights. This allows evidence such as
"reuse A alone failed, reuse B alone failed, reuse A+B won" to reward the
combination rather than falsely rewarding each feature independently.

Parent/child comparisons credit changed features first. Explicit ablation and
counterfactual generators can be layered onto the same API.

## Exploration and no-repeat memory

Ranking combines learned evidence with an uncertainty bonus. Unknown candidates
remain explorable; repeatedly poor candidates fall in priority. Exact evaluated
candidate/boundary pairs are never repeated. Mathematically impossible candidates
can be pruned permanently with an explicit reason.

## G4 Capsule

`G4Capsule` is opt-in and writes only beneath the caller-selected path. State and
experiences are written atomically, checksummed, deduplicated by content, and can
be exported as a `.g4capsule` ZIP.

## Supervisor

`MidRunSupervisor` performs the synchronous loop:

```text
rank unexplored candidates
-> evaluate the best current candidate
-> certify with the immutable judge
-> update contextual and interaction weights
-> extract residue patterns
-> checkpoint the capsule
-> optionally generate new candidates
-> repeat
```

A deterministic synthetic demonstration is available:

```bash
rns-g4-learn-demo --capsule G4_Capsule --budget 20 --output learning.json
```

The demo proves control flow, transfer, persistence, and no-repeat behavior. It
is not a hardware performance claim.
