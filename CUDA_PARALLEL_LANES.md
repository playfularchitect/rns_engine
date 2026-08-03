# Parallel Learning Lanes for CUDA Rail and Kernel Search

The CUDA search uses a shared-prior / bounded-divergence / teaching-boundary architecture.

## Shared prior

Every lane receives the same immutable evidence body:

- identical input matrices and exact shared scales;
- identical radix-128 digit planes;
- identical required grouped INT32 outputs;
- identical seven-rail residue targets;
- identical direct Python-big-integer answer;
- identical capacity and no-wrap receipts;
- identical warmup count and measurement protocol.

No lane may redefine correctness.

## Independent lanes

Initial lanes should differ in one controlled design choice:

- `lane-smallest-product`: extra Mersenne exponents `(11, 29, 31)`;
- `lane-balanced`: extra Mersenne exponents `(23, 24, 25)`;
- reduction-kernel variants inside each lawful profile;
- reconstruction ordering variants;
- grouped-coefficient launch/packing variants.

Inside one benchmark epoch, lanes may tune independently. Their outputs remain comparable because the shared fixture and proof laws are frozen.

## Hard-law firewall

Before performance evidence is accepted, every lane must pass:

1. exact grouped-partial equality;
2. exact residue equality on every rail;
3. exact centered reconstruction;
4. pairwise coprimality;
5. local INT32 safety;
6. global signed uniqueness;
7. deterministic fixture identity.

A fast failing lane has no score. GCD is not learned or approximated.

## Teaching boundary

At a fixed trial boundary:

- pool lawful integer evidence: failure counts, bytes moved, launch counts, register-spill counts, and completed output counts;
- preserve timing samples with their provenance rather than averaging incompatible runs blindly;
- select one coherent kernel/configuration body as the next teacher;
- retain alternate lawful bodies in an archive to prevent premature monoculture.

The first deterministic selection tuple is:

```text
(correctness_failures,
 validation_failures,
 median_elapsed_ns,
 bytes_moved,
 launch_count,
 lane_id)
```

Correctness and validation fields must both be zero before timing matters. `lane_id` is the fixed tie-break.

## Epoch rule

An epoch is a fixed number of completed repetitions on a fixed matrix shape, never wall-clock duration. This keeps work membership reproducible even though elapsed time is measured.

Suggested initial epoch:

```text
5 warmups + 21 measured repetitions per shape/profile
```

The next shared prior contains the winning complete kernel body plus pooled benchmark evidence. It does not average coupled CUDA source, register layouts, or launch strategies into a synthetic kernel that no lane tested.

## What can merge and what must be selected

Merge lawfully:

- exact pass/fail counts;
- operation counts;
- byte counts;
- launch counts;
- fixture hashes;
- hardware metadata;
- independent timing samples with provenance.

Select coherently:

- CUDA kernel source;
- tile shape;
- register layout;
- reduction strategy;
- reconstruction order;
- rail profile.

The rule is: merge evidence; select a complete executable body.
