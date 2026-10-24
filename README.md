---
title: ZoogGuess V2
emoji: 🐾
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# ZoogGuess V2: Neural-Bayesian Animal Guesser

An autonomous, self-contained 20 Questions deduction engine combining:
* **Bayesian Belief Distribution:** Soft-pruning inference supporting nuanced user responses (*Probably, Doubtful, Don't Know*).
* **SmolLM2-135M-Instruct:** On-device autoregressive Small Language Model synthesizing natural, single-clause interrogative questions.
* **Ground-Truth Knowledge Base:** 175 biological profiles categorized across anatomical, ecological, and behavioral traits.
