# SIMCA Anomaly Detection

Many inspection tasks are authentication tasks: they ask whether a material is what it claims
to be, or, on a sorting belt, whether a pixel is lentils or anything else. A classifier
trained on known classes cannot answer that. It divides the whole spectral space between the
classes it has seen, so a material it has never seen still lands in one of them, and if that
class is the product, the material passes [3].

A class model describes one class and leaves the rest of the space unassigned. A pixel is
rejected because it does not resemble the product, without anyone having recorded what it is
instead. SIMCA (soft independent modelling of class analogy [1]) is the best-known class model
in chemometrics. In cuvis.AI it is a chain of three nodes, fitted with
[statistical training](../../concepts/training.md) on normal data only.

**Run the example:**

- [`cuvis_ai/configs/pipeline/anomaly/simca/simca_statistical.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/pipeline/anomaly/simca/simca_statistical.yaml), the packaged pipeline.
- [`cuvis_ai/configs/trainrun/simca_statistical.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/trainrun/simca_statistical.yaml), the trainrun that fits it.

```bash
uv run restore-trainrun \
  --trainrun-path cuvis_ai/configs/trainrun/simca_statistical.yaml \
  --mode train
```

The run fits the four statistical nodes in graph order, evaluates the val and test splits and
saves the pipeline with its weights under `outputs/simca_statistical/trained_models/`. That
pair runs on new recordings with `restore-pipeline`; see
[Restore a Pipeline](../../workflows/restore-pipeline.md).

## What SIMCA measures

A PCA fitted on normal spectra keeps a few components. Every pixel then has two distances to
that model:

- The score distance (Hotelling T2) says how unusual the pixel is inside the model: its scores lie far from the scores of normal pixels.
- The orthogonal distance (Q) says how much of the pixel the model cannot describe. It is the squared residual between the spectrum and its reconstruction from the components.

A foreign material usually shows in Q, because its spectrum has a shape the normal components
do not contain. A normal material in an unusual state (much brighter, much darker) shows in
T2. The decision combines both.

## The three nodes

| Node | Inputs | Outputs | The fit learns |
|---|---|---|---|
| `TrainablePCA` | `data` | `projected`, `reconstructed` | mean and components |
| `SubspaceDistances` | `data`, `projected`, `reconstructed` | `score_distance`, `orthogonal_distance` | mean and covariance of the scores |
| `SIMCADecision` | `score_distance`, `orthogonal_distance` | `combined_distance`, `decisions` | scale and degrees of freedom of each distance |

The wiring has one fan-out to keep in mind: `SubspaceDistances.data` takes the same cube the
PCA was given, here the normalized one.

```yaml
connections:
- source: MinMaxNormalizer.outputs.normalized
  target: TrainablePCA.inputs.data
- source: MinMaxNormalizer.outputs.normalized
  target: SubspaceDistances.inputs.data
- source: TrainablePCA.outputs.projected
  target: SubspaceDistances.inputs.projected
- source: TrainablePCA.outputs.reconstructed
  target: SubspaceDistances.inputs.reconstructed
- source: SubspaceDistances.outputs.score_distance
  target: SIMCADecision.inputs.score_distance
- source: SubspaceDistances.outputs.orthogonal_distance
  target: SIMCADecision.inputs.orthogonal_distance
```

`n_components` is set on both `TrainablePCA` and `SubspaceDistances` and must be the same
number. The pipeline refuses to connect the two nodes when it differs.

## The decision

`SIMCADecision` follows the data-driven variant of SIMCA [2]. It models each distance of
normal data as a scaled chi-square variable and estimates the scale and the degrees of freedom
from the fit data. The combined distance

```text
f = N_h * h / h0 + N_q * q / q0
```

is the anomaly score: larger means less like the product. A pixel is flagged when `f` exceeds
the `1 - alpha` quantile of the chi-square distribution with `N_h + N_q` degrees of freedom.

`alpha` is the share of normal pixels the limit gives up. At the pixel level that adds up
quickly: with `alpha: 0.01`, a frame of 100,000 normal pixels has about 1,000 flagged ones.
For imaging, expect to use a much smaller `alpha`, or a minimum-area filter after the decision.
`alpha` can be changed in a saved pipeline without a new fit; the limit is derived from it when
the weights are loaded.

## Swapping the parts

The two new nodes do not depend on each other's internals, so each can be replaced.

`SubspaceDistances` only needs scores and a reconstruction. Any reduction node with a
`projected` and a `reconstructed` output can feed it in place of the PCA.

`combined_distance` is a plain score map. To use a fixed limit taken from calibration frames
in place of the chi-square limit, connect it to `ScoreThreshold` (`keep: above`) and use that
node's `decisions`. `QuantileBinaryDecider` does not work for this: it flags the same share
of every frame, with or without an anomaly in it.

`SIMCADecision` accepts any two non-negative distances, so a different model of normal can
supply either one.

## What to keep in mind

- Fit on normal data only. Every node in the chain learns what normal looks like, so foreign material in the train split widens the limits, and nothing warns about it.
- Check the limit on normal frames the fit has not seen. The chi-square limit assumes that the two distances are independent and follow the fitted shape. Count the share of flagged pixels on those frames and compare it with `alpha`; if the two disagree, use a fixed limit from the same frames.
- Keep `n_components` small. Every component the PCA keeps is taken out of the residual, so with too many of them the orthogonal distance has nothing left to show; on the lentils demo recording it separates stones with 3 components and no longer does with 10. Choose the number on normal frames the fit has not seen.
- Keep `n_components` below the rank of the data. The fit rejects a component without variance, because neither distance would see a deviation along it.
- For ranking metrics, read `combined_distance` directly. The `logits` input of `AnomalyDetectionMetrics` passes its values through a sigmoid, which saturates on a distance, so do not connect the two.

## References

1. S. Wold, "Pattern recognition by means of disjoint principal components models", Pattern Recognition 8 (1976) 127-139.
2. A. L. Pomerantsev, O. Ye. Rodionova, "Concept and role of extreme objects in PCA/SIMCA", Journal of Chemometrics 28 (2014) 429-438.
3. O. Ye. Rodionova, A. V. Titova, A. L. Pomerantsev, "Discriminant analysis is an inappropriate method of authentication", TrAC Trends in Analytical Chemistry 78 (2016) 17-22.
