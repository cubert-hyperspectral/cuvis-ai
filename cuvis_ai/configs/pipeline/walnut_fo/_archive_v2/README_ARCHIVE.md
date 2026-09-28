# Archived v2 FO pipelines (archived 2026-09-25)

The ten v2 walnut FO pipelines (two-bank CIR / RGB, the CIR feature bank, the PatchCore + SSFT
ensemble, and their gated variants) and the six `cuvis_picker_*.pt` weights of the cuvis.next picker
copies. The multi-scale pipelines in the parent folder superseded them on 24 Sep
(THOR_DEPLOY_NOTES §11 - §13).

They stay loadable from here: each yaml's weights are the `.pt` of the same name next to it. The
`ens_pc_ssft` pair needs the ssft plugin, which does not load on cuvis-ai-core 0.17.4 until its
migration (a separate task). Snapshot and smoke scripts that glob `walnut_fo/*.yaml` no longer see
these pipelines.
