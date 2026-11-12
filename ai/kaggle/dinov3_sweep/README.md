# DINOv3 validation-only sweep

Private notebook: huyentran999/durian-dinov3-validation-sweep
Four configurations: final blocks {2,4} x backbone LR {3e-6,1e-5}, seed42.
Image size256, square pad, mild augmentation; batch16; head LR1e-4.
Train head5 epochs then fine-tune up to40, warmup3 then cosine, patience8.
Each checkpoint selected by validation macro-F1. Select configuration by
seed42 validation macro-F1 (ties keep first); run selected configuration on
seeds43/44. Evaluate test ONLY after all choices and training are complete.
Report all seeds, sample std, all search validation scores.
Six training runs total, submitted as one notebook, capped at3 hours.
Search checkpoints are discarded after completion; retain the three selected
seed checkpoints and every validation history/report.

Existing test has been seen in prior experiments; these results remain
exploratory. No guaranteed improvement or statistically confirmed superiority.
No changes to app, prior checkpoints or dataset splits.
