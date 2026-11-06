# Sources and method adaptations

These are paper-based implementations tailored to the existing experiments, not claims of exact reproduction of the authors' reported results. Encoder, objective, and training settings differ by method. Comparisons therefore concern complete methods, not pooling alone.

## DTFD-MIL: AFS

- Paper: https://arxiv.org/abs/2203.12081
- Author repository: https://github.com/hrzhang1123/DTFD-MIL
- Reference commit: `10964f4dcc27c65ce110a0e9a3b9240bff58da8a`
- Reviewed: `Main_DTFD_MIL.py`, `Model/Attention.py`, `Model/network.py`.
- License: MIT; notice in `licenses/DTFD-MIL.txt`.

Frozen ImageNet ResNet-50 through layer3 produces 1024-dimensional image features. A bias-free 1024-to-512 projection and gated attention classify four randomly assigned pseudo-bags. **AFS** forwards each pseudo-bag's attention-weighted feature vector to the second gated-attention/classifier tier. There is no max/min instance selection.

Two Adam optimizers use lr 1e-4 and weight decay 1e-4, with gradient clipping at 5. Both schedules run 200 epochs and reduce lr by 0.2 after epoch 100. The implementation explicitly detaches AFS features for the second-tier update. This preserves the reference's effective separation of parameter updates while avoiding its in-place update of a retained computation graph.

Small bags use `min(4, bag_length)` nonempty groups. Evaluation uses reproducible bag-specific pseudo-bag partitions and averages probabilities over `inference_repeats` (default 1, matching the reference). Validation loss is correctly computed from probabilities; the reference's direct `nll_loss` call on unlogged probabilities is not reproduced.

## CLAM

- Paper: https://doi.org/10.1038/s41551-020-00682-w
- Author repository: https://github.com/mahmoodlab/CLAM
- Reference commit: `53e2409d4a8189c682c173382964a85f114f923c`
- Reviewed: `models/model_clam.py`, `main.py`.
- License: GPL-3.0; text in `licenses/CLAM.txt` and `COPYING`.

Use **CLAM-SB**, gated attention, frozen ImageNet ResNet-50 layer3 features, a 512-dimensional projection, and 256-dimensional attention branches. Each class has a separate two-output instance classifier. Training combines 0.7 times bag cross-entropy with 0.3 times instance clustering loss. The latter uses the smooth top-1 SVM formula (unit margin and temperature); its stable logsumexp implementation follows the mathematical loss used by https://github.com/oval-group/smooth-topk.

Adam uses lr 1e-4 and weight decay 1e-5; dropout is 0.25. The default budget is 200 epochs. The usual eight top and eight bottom instances become `min(8, floor(N/2))` each for small bags, with disjoint sets even under tied attention scores. A singleton uses bag loss only. Multiclass X-ray enables out-of-class instance supervision (subtyping), as supported by CLAM.

MNIST uses validation-loss early stopping with patience 20 and minimum 50 epochs, then evaluates the best in-memory state on its existing test set. No model weight files are saved.

## ProMIL

- Paper: https://arxiv.org/abs/2306.10535
- Author repository: https://github.com/gmum/ProMIL
- Reference commit: `3bb79f7fc68d66957e457db1250d0ab49ccc880b`
- Reviewed: `smallLinear.py`, `smallCNN.py`, `utils2.py`, `utils3.py`; paper Section 5.2.
- License: MIT; notice in `licenses/ProMIL.txt`.

Instance probabilities are sorted and aggregated using a learnable Bernstein estimator. Computation stays in log space. Both probability and its complement are aggregated directly to avoid cancellation near one. The coefficient ordering matches the author implementation: `alpha^(n-k) * (1-alpha)^k` for ascending instance scores. The sigmoid parameterization is unconstrained by ReLU, following `smallCNN.py`, with raw estimator initialization 0.2. Learned sigmoid parameters are exported per fold; they are not assumed to equal the observed pathology proportion.

For **MNIST**, use a randomly initialized, trainable LeNet-style encoder (20/50 convolutional filters, 500-dimensional embedding) on 28x28 grayscale images. This instantiates the LeNet family described in the paper; the public ProMIL repository does not provide a complete MNIST training model. Adam uses lr 1e-4, betas (0.99, 0.999), weight decay 1e-5, a 100-epoch limit, and validation-loss early stopping with patience 15. Existing MNIST bags and test images are unchanged; their positive-bag definition remains the user's experiment, not the paper's different percentage-threshold dataset.

For **medical datasets and Achilles sparsity**, adapt the paper's ultrasound protocol: ImageNet ResNet-18, fully connected layers of 128 and 64 units, 25 frozen-encoder epochs, then 100 epochs of whole-network fine-tuning. Training resizes to 200 and center-crops to 172; evaluation resizes to 172. All modalities retain their existing decoding/windowing before RGB repetition and ImageNet normalization. Using this ultrasound setup for MRI, CT, and X-ray is a modality adaptation. The paper reports hyperparameter search ranges rather than a universal selected optimum; lr 1e-4, dropout 0.2, weight decay 1e-5, and no scheduler are explicit starting choices within those ranges, not claimed published optima.

The original ProMIL is binary. **Three-class X-ray is explicitly an OVR adaptation**: one instance predictor and Bernstein parameter per class, one-vs-rest binary losses, then normalization of class bag probabilities for prediction. It must be labeled “ProMIL (OVR)” in multiclass result tables.

## Shared evaluation adaptations

The existing ten-fold medical/Achilles protocol has no independent model-selection set. Those runs evaluate a fixed final epoch rather than selecting checkpoints on the same held-out patients. Only MNIST uses its separate validation partition for early stopping. There are no hyperparameter searches or repeated seeds in this initial sweep. This is deliberately different from repeated/tuned evaluations in the papers.

CLAM and DTFD share a deterministic frozen feature cache as a computational convenience. It is not fitted to any images, so extracting validation/test features does not train on their labels. ProMIL processes raw images and trains its own encoder according to its profile. No existing instance-pretraining checkpoints or RadImageNet weights are used.
