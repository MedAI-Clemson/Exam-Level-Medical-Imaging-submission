# MNIST bag composition experiment

## Objective

Evaluate how the fraction of positive instances and the number of instances in a bag affect five multiple instance learning (MIL) pooling methods. The recognition task remains fixed: predict whether a bag contains the digit 9. This removes the differences in modality, diagnosis, and image acquisition that complicate sparsity comparisons across medical datasets.

This document describes the implemented experiment, not completed results. The operational instructions are in [README.md](README.md); defaults are in [sweep.yaml](sweep.yaml).

## Experimental conditions

The five methods are average pooling, max pooling, ABMIL (dot-product), ABMIL (MLP), and transformer pooling. They use the existing shared model definitions. ABMIL (MLP) is the tanh attention implementation; it does not include a separate sigmoid gating branch.

Each method is trained separately for every condition below. There are 12 conditions per method and 60 training runs, using one seed (123) and no repeated runs or cross-validation.

| Images per bag | Positive-instance density | Digit 9 images in each positive bag | Distractor images in each positive bag |
|---|---|---|---|
| 10 | 10% | 1 | 9 |
| 10 | 20% | 2 | 8 |
| 10 | 50% | 5 | 5 |
| 20 | 5% | 1 | 19 |
| 20 | 10% | 2 | 18 |
| 20 | 25% | 5 | 15 |
| 20 | 50% | 10 | 10 |
| 100 | 1% | 1 | 99 |
| 100 | 5% | 5 | 95 |
| 100 | 10% | 10 | 90 |
| 100 | 25% | 25 | 75 |
| 100 | 50% | 50 | 50 |

All negative bags contain only non-9 digits and have the same size as positive bags in that condition. Density refers to positive bags, not the fraction of positive bags in the dataset. A bag containing one or more 9s has label 1; a bag containing no 9s has label 0.

## Source images and bag construction

Before constructing any bags, the official MNIST training images are divided into disjoint training and validation source pools, stratified by digit, using an approximately 80/20 split. The official MNIST test images supply the test source pool. Source images cannot cross these three partitions.

For every condition, construct 1,000 training bags, 200 validation bags, and 1,000 test bags. Each partition contains equal numbers of positive and negative bags. Positive bags contain exactly the target count in the table. For every distractor position, sample a digit class uniformly from the nine non-target classes, then sample an image of that class. Target images are sampled from the digit-9 pool. Sampling is with replacement, allowing a source image to occur more than once within a bag or across bags in the same partition. Shuffle instance order within each bag.

Generate and save the bags once, before training. Every method receives the identical bags for a given condition, including instance order. Conditions use separate deterministic sampling seeds derived from the main seed, bag size, density, and partition; bags are not nested or explicitly matched across density conditions. The manifests record source indices, bag labels, and exact target counts. Within each run, bag composition remains fixed across epochs; only training bag order is shuffled.

## Model and training configuration

Use a ResNet-50 initialized with ImageNet weights, with encoder parameters frozen and the encoder in evaluation mode. Initialize the MIL classifier and pooling parameters randomly. The two ABMIL implementations use 16 heads; the transformer uses the existing two-layer, 16-head implementation with its feature projection and bag token. Architecture details otherwise remain those in the shared model code.

Resize images to 224 by 224 pixels, replicate grayscale into three channels, and apply ImageNet channel normalization. Do not apply random augmentation in this experiment. Encoder, classifier, and transformer dropout are set to zero.

Train with binary cross-entropy on bag labels only. Use Adam with learning rate 0.001 and weight decay 0.001, a StepLR schedule multiplying learning rate by 0.1 every eight epochs, one bag per batch, and 10 epochs. Use the same settings for every method and condition. Training at bag size 100 processes five times as many images per epoch as training at size 20; this design fixes bags and optimizer updates, not total image exposure or compute.

Track training and validation loss, AUROC, and Brier score each epoch. Evaluate the test set only after the last epoch. Do not choose an epoch or configuration using test results. No model checkpoints are saved; an interrupted job restarts training from epoch one.

## Workflow and records

The separate `mnist-sweep` workflow reuses the medical workflow's model definitions and Palmetto SLURM profile. Preparation runs on the controller, downloads MNIST, caches ImageNet weights, and creates shared manifests. Each GPU job trains one method and composition condition. A final local step combines all completed results into `summary.csv`.

The default output directory is `Image Datasets/results/mnist_composition_v2/`. Preserve the effective sweep settings, `design.json`, per-run resolved settings and manifest hash, epoch histories, test predictions, summaries, and logs. Use a new run name for changes to the experimental settings. Keep the repository commit identifier with the archived results to identify the shared model implementation.

## Outcomes and planned presentation

The primary outcome is test bag-level AUROC. Secondary outcomes are test Brier score and binary cross-entropy loss. Report one value per method and condition; do not describe these as means across folds or repetitions.

Plot positive-instance density on the horizontal axis and AUROC on the vertical axis, with separate panels for bag sizes 10, 20, and 100 and one curve per method. Use the same AUROC scale in both panels. Present Brier scores in a companion figure or table. Compare methods within the same composition condition before describing how their relative performance changes with density.

This initial single-seed experiment is descriptive: no repetition-based confidence intervals or significance tests are produced. Reused source images make synthetic bags statistically dependent, so an ordinary bag bootstrap does not represent independent-image uncertainty. Density changes both positive and distractor counts at fixed bag size, and changing bag size at fixed density changes the absolute amount of both. Results describe sensitivity to bag composition, not a completely isolated effect of density. Training and test density are matched; cross-density generalization is outside this experiment.

## Incremental extension

The expanded sweep retains the 30 completed v2 runs after explicit verification and adds 30 new runs. Each new condition has a separate configuration and manifest. The original combined design and results are preserved; verified receipts allow the combined summary to include them without retraining. Follow the migration commands in README.md before extending the existing folder. The earlier 20-epoch trainable-encoder experiment is separate and is not eligible for reuse in this frozen 10-epoch sweep.

