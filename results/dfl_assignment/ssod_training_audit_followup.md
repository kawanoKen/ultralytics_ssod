# SSOD training follow-up audit

## Scope

Static audit of the current SSOD trainer, dataset/augmentation path, EMA/teacher lifecycle, DDP sampling, warmup,
checkpoint/resume, and assignment-stability loss. No training or corrective patch was run as part of this audit.

## Confirmed findings

### Critical — Post-burn-in resume does not restore the Mean-Teacher state

The current trainer updates `self.teacher` after optimizer steps, but checkpoints save only `self.ema`; no teacher
weights or teacher update counter are saved. The current resume path added earlier at
`ssod_train.py:566-583` reconstructs a teacher from the original supervised checkpoint and explicitly assumes the
teacher is frozen. That assumption is now false after `teacher.update(self.model)` was added at lines 762 and 1194.

- With `burn_in_epochs=0`, teacher and checkpoint EMA start together and receive the same updates, so they should be
  effectively identical. Exact resume can initialize teacher from the restored checkpoint EMA, not the supervised
  initialization.
- With `burn_in_epochs>0`, teacher has a different initialization time/update counter. Exact resume requires saving
  and restoring teacher weights plus its `updates` value.
- The present path silently changes pseudo-label generation after resume. This is a correctness bug, not just an
  efficiency issue.

### High — `close_mosaic` misses the SSOD loader, and simply calling `close_mosaic()` is insufficient

The inherited `_close_dataloader_mosaic()` only modifies `self.train_loader.dataset`. The SSOD phase calls it and then
resets `self.ssod_train_loader` (`ssod_train.py:929-931`), without changing the unlabeled dataset.

There is a second layer to the bug: base `YOLODataset.close_mosaic()` sets `hyp.mosaic=0`, but
`YOLODataset_ssod.build_transforms()` immediately overwrites it from `hyp.mosaic_ssod` (dataset_ssod.py:37). Therefore
even applying the inherited close method directly to the SSOD dataset would rebuild mosaic with probability
`mosaic_ssod` unless the SSOD-specific fields are also zeroed or the subclass overrides `close_mosaic()`.

Impact: with the current `close_mosaic=10, mosaic_ssod=1`, unlabeled teacher/student images retain mosaic through the
last ten epochs. Resume inside the close-mosaic window has the same problem.

### High — Warmup duration uses the wrong epoch length when burn-in is zero

`nw` is computed from the labeled loader length (`warmup_epochs * nb`) at `ssod_train.py:526`, while the active loop
for `burn_in_epochs=0` is the much longer unlabeled loader (`nb_ssod`). For CrowdHuman 5% with the experimental setup,
the `max(..., 100)` fallback makes the configured three warmup epochs only 100 SSOD iterations—less than one SSOD
epoch—rather than approximately `3 * nb_ssod`.

Impact: the recorded experiments did not use the warmup schedule implied by `warmup_epochs=3`. This affects all
methods using this trainer equally but matters when claiming parity with a reference implementation.

### Medium — Labeled DDP sampler epoch is never advanced during the SSOD phase

During burn-in, `self.train_loader.sampler.set_epoch(epoch)` is called at line 624. During the SSOD phase only
`self.ssod_train_loader.sampler.set_epoch(epoch)` is called (line 926). The labeled loader is re-created and repeatedly
cycled inside each SSOD epoch, but its distributed sampler remains at the same epoch seed.

Impact: labeled sample order repeats across SSOD epochs. Image augmentation RNG may still change, but distributed
shuffle is not epoch-varying as expected.

### Medium — Optimizer auto-selection iteration estimate mixes global and per-rank batch quantities

At line 466 the unlabeled term divides dataset size by `max(batch_size_ssod, nbs)`, where `batch_size_ssod` has already
been divided by world size, while other trainer quantities use global batch size. This estimate is passed to
`optimizer=auto`. It does not change the present CrowdHuman choice (both estimates lead to SGD), but it can select a
different optimizer or learning rate for other dataset/world-size combinations.

### Low — SSOD dataset transform construction mutates shared hyperparameters

`YOLODataset_ssod.build_transforms()` assigns `hyp.mosaic`, `hyp.mixup`, and `hyp.cutmix` from the SSOD-specific fields
in place. This can overwrite the trainer configuration saved/logged for labeled augmentation and contributes to the
close-mosaic failure. The current experiment uses equal/default values, so the immediate numerical impact is limited.

## Teacher clarification

The supplied note is correct for the **current, fixed** code: both `self.ema` and `self.teacher` call `update()` after
each optimizer step (base optimizer step updates the former; SSOD trainer updates the latter). With burn-in zero they
are redundant and converge identically, apart from any lifecycle/resume divergence.

However, the earlier R1 runs were made before the `teacher.update(self.model)` calls existed; at that time only
`update_attr()` ran and the teacher was frozen at initialization. Existing confidence-only/DFL-selection results also
appear to come from that legacy trainer. Those comparisons remain internally comparable if all used the frozen
teacher, but they must be labeled as legacy/frozen-teacher and are not directly comparable to future runs using the
new tracking teacher without rerunning the baseline.

## Checked and not found to be erroneous

- DA hooks are registered on student neck modules, not on the teacher.
- `assignment_stability.py` has no EMA/teacher lifecycle.
- Weak teacher and strong student views share geometry; strong augmentation is appearance-only.
- Baseline TaskAlignedAssigner output remains the training assignment for R1; candidate assignments do not feed back
  into the positive mask.
- R1 localization normalization uses the stability-weighted TaskAligned target-score sum, avoiding a simple global
  loss-scale reduction.

## Priority before further SSOD training

1. Define and test the intended teacher policy, then save/restore it exactly.
2. Close SSOD mosaic through an SSOD-specific dataset method and test the transform probability at the boundary and
   after resume.
3. Compute warmup from the active phase loader and advance both DDP samplers per epoch.
4. Run a short uninterrupted-vs-resumed equivalence test before any new full experiment.

## Remediation applied

The issues above were fixed before further SSOD training:

- Tracking-teacher weights and EMA update count are now saved in resumable checkpoints and restored on both normal
  resume and NaN recovery. Post-burn legacy checkpoints without teacher state are rejected because their frozen vs
  tracking teacher history cannot be inferred safely. Stripped inference checkpoints discard the extra teacher.
- The trainer closes and resets both labeled and unlabeled loaders. `YOLODataset_ssod.close_mosaic()` explicitly
  zeros `mosaic_ssod`, `mixup_ssod`, and `cutmix_ssod`, and transform construction no longer mutates shared args.
- Warmup iteration count is phase-aware and uses the SSOD loader length when `burn_in_epochs=0`.
- Both labeled and unlabeled distributed samplers receive the epoch during the SSOD phase.
- Optimizer auto-selection now uses the global unlabeled batch size consistently.

Focused lifecycle and assignment-stability regression tests pass (12 tests total). No full training was started.
