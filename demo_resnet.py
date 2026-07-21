"""
demo_resnet.py
==============
Train the ResNet1D component detector on PURE synthetic spectra only, then
detect components in unseen competitive-adsorption mixtures. Compares its
multi-label detection to the RandomForest baseline.
"""
import numpy as np
from synthetic import build_dataset
from sers_mixture import (SERSMixtureClassifier, AugmentConfig, preprocess,
                          multilabel_metrics, names_to_indicator)
from resnet1d import ResNet1DDetector

data = build_dataset(n_components=6, n_feat=500, seed=0)
names = data["names"]
pure = preprocess(data["pure_raw"])
test = preprocess(data["test_specs"])
true_names = [[names[i] for i in t] for t in data["test_true"]]
Y_true = names_to_indicator(true_names, names)

aug = AugmentConfig(n_per_pure=150, noise_frac=0.03, shift_max=2,
                    baseline_amp=0.05, seed=0)

print("=" * 60)
print("ResNet1D vs RandomForest  (trained on PURE spectra only)")
print("=" * 60)

# ---- ResNet1D ----
print("[ResNet1D] training...")
rn = ResNet1DDetector(names, prob_threshold=0.5, epochs=25, base=16, augment=aug)
rn.fit(pure)
pred_rn = rn.predict(test)
m_rn = multilabel_metrics(Y_true, names_to_indicator(pred_rn, names))

# ---- RandomForest baseline ----
rf = SERSMixtureClassifier(names, prob_threshold=0.30, augment=aug)
rf.fit(pure)
pred_rf = [d["components"] for d in rf.predict(test, return_details=True)]
m_rf = multilabel_metrics(Y_true, names_to_indicator(pred_rf, names))

print("-" * 60)
print(f"{'metric':20s} | {'ResNet1D':>10} | {'RandomForest':>12}")
for k in m_rn:
    print(f"{k:20s} | {m_rn[k]:10.3f} | {m_rf[k]:12.3f}")
print("=" * 60)
