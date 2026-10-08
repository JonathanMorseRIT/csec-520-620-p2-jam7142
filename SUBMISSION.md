# Submission Manifest — Project 2

Fill this in before submitting. The grading agent reads it first.

## Student
- Jonathan Morse (jam7142)

## Course level
- [ ] 520 (undergraduate)
- [x] 620 (graduate)

## Dataset
- Name / source / version / URL: UNSW-NB15, using `UNSW_NB15_training-set.csv` from the course-provided UNSW-NB15 dataset archive on Google Drive.
- How to obtain it (command or steps): Download the course-provided UNSW-NB15 archive from the Google Drive link in `README.md`, extract `UNSW_NB15_training-set.csv`, and place it at `data/UNSW_NB15_training-set.csv`.
- Label column (used for EVALUATION ONLY): `attack_cat`
- Subsample size and how you chose it: 3,730 rows. The configuration specifies a maximum balanced subsample of 4,000 rows. The provided balanced sampler preserves class coverage and produces 400 samples per class except Worms, which has only 130 available samples. Labels are used for balanced sampling and evaluation only, not as clustering features.

## How to reproduce
```bash
make setup
# Download the course Drive archive and extract data/UNSW_NB15_training-set.csv
# See data/README.md; make data is an optional mirror fallback / checksum check.
make reproduce
```

Anything non-default the grader must know (data download, runtime): The UNSW-NB15 archive must be downloaded separately and `UNSW_NB15_training-set.csv` placed in the `data/` directory. The final experiments use 30 K-means initialization restarts (`n_init: 30`) and a k-selection sweep from 2 through 15.

## Your implementation
- Confirm `config.yaml` has `kmeans.implementation: scratch`: [x]
- Initialization used (`random` / `kmeans++`): `kmeans++`
- How you handle **empty clusters**: If a cluster receives no points during an update, its previous centroid is retained rather than attempting to compute a mean over an empty set.
- Convergence criterion and tolerance: Iteration stops when either the maximum Euclidean centroid shift is less than `1.0e-4` or cluster labels are unchanged from the previous iteration. The maximum iteration count is 300.
- Agreement with the reference (each run’s `reference_check` in metrics.json; same geometry):
  - Euclidean `inertia_ratio` / `ari_vs_reference`: 1.0139007428142124 / 0.7549630512178934
  - Mahalanobis `inertia_ratio` / `ari_vs_reference`: 1.0172420406639098 / 0.7203943472057279

## Choosing k
- k you report, and the evidence (elbow / silhouette): `k = 11`. During the initial Euclidean sweep over `k = 2` through `15` using the original restart budget (`n_init = 10`), `k = 11` produced the highest common-space silhouette coefficient (`0.45133087955648876`). The inertia curve also showed a substantial decrease from approximately 55,336 at `k = 10` to 43,754 at `k = 11`, followed by a much smaller decrease to approximately 43,125 at `k = 12`. The restart budget was subsequently increased to 30 to investigate reference agreement and sensitivity to local minima. Under the increased restart budget, the Euclidean sweep selected `k = 10`; however, `k = 11` was retained as the previously selected value so that k was not reselected after beginning the Euclidean-versus-Mahalanobis comparison and remained fixed across both primary runs.
- If the best silhouette k differs from the number of true classes, explain: The number of clusters was selected using an unsupervised silhouette criterion rather than the known number of `attack_cat` classes. Ground-truth labels were reserved for sampling and evaluation and were not used to directly determine k. Therefore, the selected number of clusters is not required to equal the number of true classes.

## Claimed results (must match `results/metrics.json`)

| Metric | Euclidean | Mahalanobis |
|---|---:|---:|
| Silhouette (common Euclidean) | 0.45133087955648876 | 0.44020603552067894 |
| Silhouette (configured geometry) | 0.45133087955648876 | 0.4177513078978834 |
| V-measure | 0.2566781973561918 | 0.27584917779928964 |
| Accuracy | 0.3064343163538874 | 0.3302949061662198 |
| Macro F1 | 0.23474189019267167 | 0.24409701408832157 |

- Primary comparison criterion declared before weight experiments: Common-space Euclidean silhouette (`silhouette_euclidean`). This evaluates the Euclidean and Mahalanobis partitions in the same standardized Euclidean feature space, providing a common geometry for the primary comparison.

- The diagonal C you chose, its feature order, and why: The input features were standardized before clustering, so differences in their original measurement scales were already controlled. For the Mahalanobis experiment, I used a feature-group-based diagonal that assigns weight 2.0 to the connection-context/count features (`ct_srv_src`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, and `ct_srv_dst`) and weight 1.0 to all other retained features. I hypothesized that emphasizing these features, which describe repeated source, destination, port, and service behavior, could provide a useful alternative clustering geometry to weighting all standardized features equally. The ground-truth `attack_cat` labels were not used to construct this diagonal.

In the exact `feature_names` order recorded in `metrics.json`, the diagonal is:

`C = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0]`

- Same sample, k, seed and restart budget across the primary runs: [x]

- Tradeoffs and whether any independent validation was used: The Mahalanobis weighting did not improve the predeclared primary criterion. Common-space Euclidean silhouette decreased slightly from 0.4513 to 0.4402. However, agreement with the known attack categories improved: V-measure increased from 0.2567 to 0.2758, majority-vote accuracy increased from 0.3064 to 0.3303, and macro F1 increased from 0.2347 to 0.2441. This suggests that emphasizing connection-context features produced clusters that aligned somewhat better with the known categories at the cost of slightly lower separation in the common Euclidean geometry. The class-based metrics are descriptive in-sample agreement measures rather than held-out detection accuracy. No independent validation dataset was used, so no claim of improved generalization is made.

## AI-use acknowledgment
- Double checking correctness of code changes to kmeans.py
- Analyzing command line results
- Debugging assistance
- Proofreading