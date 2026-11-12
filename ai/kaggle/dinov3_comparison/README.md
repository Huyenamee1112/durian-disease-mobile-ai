# DINOv3 comparison with the existing DINOv2 recipe

Notebook: https://www.kaggle.com/code/huyentran999/durian-dinov3-comparison

Runs seeds 42/43/44 sequentially in one private GPU notebook. Same dataset splits,
224px, batch 8, augmentation, ImageNet normalization, AdamW learning rates,
weight decay, label smoothing, gradient clipping, plateau scheduler, 20-epoch cap,
5-epoch patience and final-two-block fine-tuning as the existing DINOv2 recipe.

Select each checkpoint solely by validation macro-F1, then report test for all seeds.
The comparison is against an existing single-seed DINOv2 result, not a paired
multi-seed comparison of both architectures. No claim of statistical superiority.
The test has already been consulted in previous experiments; independent new
field data remains needed for final confirmation.

Outputs: seed_42/, seed_43/, seed_44/ (full checkpoints and reports),
comparison_summary.json with means and sample standard deviations.
No overwrite of the earlier DINOv3 notebook, checkpoints or app model.
