# DINOv3 ViT-S/16 field training

Private notebook: https://www.kaggle.com/code/huyentran999/durian-dinov3-vit-s16-field

Dataset: huyentran999/durian-field-balanced110, existing train/val/test split.
Backbone: timm vit_small_patch16_dinov3.lvd1689m, pretrained on LVD-1689M.
Uses the timm-provided weights and remains subject to the DINOv3 license.

Recipe: 256px square padding, batch 16, seed 42, fine-tune final 2 blocks,
head LR 3e-4, backbone LR 1e-5, AdamW, 25 epochs maximum, patience 6.
Select by validation macro-F1 and evaluate test once after restoring the best
checkpoint. Exact file hashes are checked across splits before training.
This check cannot establish absence of near-duplicate images or tree-level leakage.

Outputs in /kaggle/working:
- dinov3_vits16_field_4class_best.pth (full backbone + head, optimizer, config)
- dinov3_vits16_field_4class_report.json (validation and test metrics)
- training_history.json, run_config.json, split_manifest.json
- confusion_matrix.csv
- failure.json only if execution fails

The checkpoint uses nn.Sequential(timm backbone with num_classes=0,
nn.Sequential(LayerNorm(num_features), Dropout(.2), Linear(num_features,4))).
Load class_names and preprocessing from checkpoint config; do not load into
the existing DINOv2 app class directly. App integration is a separate step.

attempts.json tracks submissions for this request. Limit: 3 TOTAL submissions,
including the first. Read remote status before any retry after a network timeout.
Retries fix runtime failures, not tune hyperparameters against test results.
