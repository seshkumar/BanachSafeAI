# BanachSafeAI

## Neural Networks with Built-In Safety Guarantees

A neural network architecture where **robustness**, **privacy**, and **expressivity** are mathematical consequences of a single design choice -- the geometry of l^p Banach spaces. Every prediction carries a **certified safety guarantee** computed exactly from the architecture's weights. No post-hoc testing. No empirical estimation.

---

### Quick Links

| Resource | Link |
|----------|------|
| Interactive Demo (Google Colab) | [Open in Colab](https://colab.research.google.com/drive/11xAXxd9RqQm0abBs7k6Y-FCFksoA6_lS?usp=sharing) |
| Project Video (3 min) | [Watch](video/banachsafeai_demo.mp4) |

---

## The Problem

AI systems deployed in safety-critical settings -- medical devices, brain-computer interfaces (BCIs, systems that decode neural signals to control devices such as wheelchairs and exoskeletons), clinical decision support -- are tested for safety **after** they are built. Companies run adversarial benchmarks, submit the results to regulators (MHRA, FDA, EU notified bodies), and hope for approval.

This approach has three fundamental limitations:

1. **You can only test attacks you have already thought of.** A new attack tomorrow could bypass every test you ran yesterday. Passing 1,000 tests provides no guarantee about the 1,001st.

2. **The company chooses which tests to report.** A manufacturer can run 100 adversarial benchmarks, fail 30, and submit only the 70 that passed. The regulator has no way to know.

3. **Testing provides no guarantee about the worst case.** Even if the system survives every test, there is no proof that it will behave correctly on an input it has never seen before.

What if, instead of testing the system and hoping, we could **prove mathematically** that it is safe?

That is what the Banach ResNet does.

---

## The Key Idea: The Lipschitz Constant

Every neural network is a function: it takes an input (e.g., brain signals) and produces an output (e.g., "left hand"). The critical question is: **how sensitive is this function to small changes in the input?**

The **Lipschitz constant** L is a single number that answers this question:

> **If the input changes by at most delta, the output changes by at most L x delta.**
>
> No exceptions. No probabilities. A mathematical fact.

In standard neural networks, **nobody knows what L is**. It must be estimated by running experiments.

In the Banach ResNet, L is **computed exactly** from the architecture's weights:

> **L_p = product over all layers of (1 + eta_k x ||A_k||)**

where K is the number of layers, eta_k is a step size, and ||A_k|| is the spectral norm of the k-th layer's weight matrix. Every term in this product is known. There is no approximation.

---

## Three Safety Properties from One Number

The Lipschitz constant L_p is the single number that controls all three safety properties:

### Robustness

The certified radius **r = margin / L_p** defines a **safe zone** around each input. Any perturbation smaller than r -- electrode noise, session drift, movement artifacts -- is **mathematically guaranteed** not to change the prediction.

A standard neural network gives you a prediction and nothing else. The Banach ResNet gives you a prediction **and a guarantee**.

### Privacy

The same L_p bounds how much any single training example can influence the model's gradients (per-sample sensitivity). This enables differentially private training (DP-SGD) **without the gradient clipping** that all existing methods require. Clipping distorts gradients and degrades accuracy; the Lipschitz bound eliminates this.

### Expressivity

At **p = 2** (standard Euclidean geometry), the duality map reduces to the identity and the network collapses to an affine function -- it cannot model nonlinear patterns. This is called the **Hilbert degeneracy**, confirmed experimentally.

Moving **p away from 2** restores nonlinear computation. Cross-validation selects the value of p that balances accuracy against the strength of the safety certificate.

---

## The Architecture

Each residual layer applies the **duality map** as its activation function:

> **J_p(z) = sign(z) x |z|^(p-1)**

The update rule is:

> **h_{k+1} = h_k - eta_k x J_p(A_k h_k + b_k)**

This is a step of mirror descent in l^p geometry. Spectral normalisation on all weight matrices ensures the Lipschitz constant is computable in closed form.

| p value | Activation shape | Behaviour |
|---------|-----------------|-----------|
| p = 2 | Identity (linear) | Network collapses to affine -- cannot learn patterns |
| p = 3 | Quadratic | Nonlinear + certified bounds (best for BCI data) |
| p = 1.5 | Sublinear | Strong privacy bounds, less expressive |

---

## Proof-of-Concept Results

Validated on the **BNCI2014-001** benchmark: 9 subjects, 22 EEG channels, 4-class motor imagery (left hand, right hand, feet, tongue).

| Geometry (p) | Accuracy | Lipschitz L_p | Mean Certified Radius | Note |
|:---:|:---:|:---:|:---:|---|
| 1.2 | 42.9% | 4.2 | 0.1026 | |
| 1.5 | 44.6% | 21.0 | 0.0619 | |
| **2.0** | **39.1%** | **2.7** | **0.2225** | **Hilbert degeneracy (affine)** |
| **3.0** | **45.7%** | **11.35** | **0.0653** | **CV-selected best geometry** |
| 4.0 | 40.1% | 16.5 | 0.0764 | |

### Example Certificate (p = 3.0)

```
Prediction:              Left-hand motor imagery
Classification margin:   0.742
Lipschitz constant L_p:  11.35
Certified radius r:      0.065

This means: no input perturbation smaller than 0.065
can change this prediction. This is a mathematical proof.
```

---

## From Toolkit to Regulatory Submission

The end product is not just a trained model -- it is a **certificate**: a document stating, for each prediction, the robustness radius, the privacy budget consumed, and the expressivity verification.

Today, companies submitting AI medical devices provide empirical test reports (adversarial attack results, stress tests). BanachSafeAI generates **mathematical proof** instead -- deterministic, per-prediction, auditable.

Three regulatory frameworks converging in 2026 create immediate demand:

- **EU AI Act** Article 15 (enforcement August 2026): requires robustness evidence for high-risk AI
- **UK MHRA**: publishing new AI medical device framework in 2026
- **US FDA TPLC** (2025): demands lifecycle robustness evidence for AI-enabled medical devices

No existing toolkit provides the mathematical evidence these frameworks are beginning to require.

---

## Interactive Demo

**Run in ~2 minutes on CPU. No GPU required. No installation needed.**

[Open in Google Colab](https://colab.research.google.com/drive/11xAXxd9RqQm0abBs7k6Y-FCFksoA6_lS?usp=sharing)

The notebook demonstrates:

1. The duality map activation function at different values of p
2. Training Banach ResNets at six geometry values
3. A per-prediction robustness certificate with certified radius, margin, and Lipschitz constant
4. The p-sweep showing how one parameter controls accuracy, robustness, and Lipschitz bound
5. Per-prediction certificate visualisation colour-coded by confidence
6. The Hilbert degeneracy at p = 2 confirmed experimentally

To run: open the link, click **Runtime** then **Run all**. All outputs are pre-rendered -- the notebook functions as a visual document even without execution.

---

## Project Video

A 3-minute overview of the project, the architecture, and the safety certificate framework.

[Watch the video](video/banachsafeai_demo.mp4)

---

## Summary

|  | Standard Neural Networks | Banach ResNet |
|---|---|---|
| **Robustness** | Test with known attacks, hope for the best | Mathematical proof: certified radius per prediction |
| **Privacy** | Clip gradients (distorts learning) | Architecture bounds gradients (no distortion) |
| **Expressivity** | Unknown relationship to safety | Controlled by geometry parameter p |
| **Lipschitz constant** | Unknown | Computed exactly from weights |
| **Regulatory evidence** | Empirical test reports | Mathematical certificates |

---

**One architecture. One parameter. Three safety guarantees. Zero post-hoc testing.**

---

## Contact

K. S. Sesh Kumar

Brevan Howard Centre for Financial Analysis, Imperial Business School

Homepage: [seshkumar.github.io](https://seshkumar.github.io/)
