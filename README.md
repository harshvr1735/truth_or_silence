# README.txt

## Team Name - NLPandas
## Project Name - Truth or Silence
---

## Project Description

This project investigates methods for detecting hallucinated or unreliable answers produced by large language models (LLMs). We evaluate a confidence-based abstention strategy using token-level log probabilities from the **Mistral-7B-Instruct** model on the **TruthfulQA multiple-choice dataset**.

Our approach computes a **log-probability margin between the most likely and second-most likely answer tokens** and uses this margin as a confidence signal. When the margin falls below a threshold, the system abstains instead of answering. We analyze how this method improves answer reliability and investigate factors that affect model confidence, including question length and number of answer choices. We also analyze a **Verbalized Confidence Model** which asks the LLM for it's confidence in it's answer to evaluate the accuracy.

All core experimental code for this project was written by our team. AI tools (e.g., ChatGPT) were occasionally used to assist with **debugging, troubleshooting errors, and improving code clarity**, but the experimental design, implementation, and analysis were developed by our team.

---

# Libraries used

• **PyTorch** – [https://pytorch.org/](https://pytorch.org/)
Used for running the Mistral-7B model, GPU inference, and tensor computations.

• **HuggingFace Transformers** – [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
Used for loading the pretrained Mistral-7B-Instruct model and tokenizer.

• **HuggingFace Datasets** – [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
Used for loading the TruthfulQA multiple-choice dataset.

• **Accelerate** – [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)
Used to efficiently run the large language model on GPU.

• **BitsAndBytes** – [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
Used for 4-bit quantization to allow efficient inference with the 7B parameter model.

• **NumPy** – [https://numpy.org/](https://numpy.org/)
Used for numerical computations and statistical calculations.

• **Pandas** – [https://pandas.pydata.org/](https://pandas.pydata.org/)
Used to store experiment results and perform tabular analysis.

• **Matplotlib** – [https://matplotlib.org/](https://matplotlib.org/)
Used to generate visualizations and experiment result plots.

• **SentencePiece** – [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)
Used internally by the tokenizer for text tokenization.

• **Safetensors** – [https://github.com/huggingface/safetensors](https://github.com/huggingface/safetensors)
Used for efficient model weight loading.

---

# Publicly available codes used

• **HuggingFace Transformers model loading examples**
[https://huggingface.co/docs/transformers/model_doc/mistral](https://huggingface.co/docs/transformers/model_doc/mistral)
Used as reference for loading the Mistral-7B-Instruct model and tokenizer.

• **TruthfulQA dataset loader (HuggingFace Datasets)**
[https://huggingface.co/datasets/truthful_qa](https://huggingface.co/datasets/truthful_qa)
Used to access the TruthfulQA multiple-choice dataset.

• **PyTorch inference and tensor operations documentation**
[https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
Used as reference for implementing model inference and extracting token probabilities.

These references were used as guidance for interacting with the libraries. The experimental pipeline, confidence computation, evaluation procedures, and analysis code were implemented by our team.

---

# Scripts/functions written by our team

• **V1 - CS 175 truth_or_silence_(Tokens).ipynb / V1 - CS 175 truth_or_silence_(Tokens).py**
~700–900 lines

Implements the main hallucination detection method using **token-level log-probability margins**. The script loads the Mistral-7B-Instruct model, prompts the model with multiple-choice questions from TruthfulQA, extracts token-level probabilities, computes the confidence margin, and applies a threshold-based abstention strategy.

It also performs extensive analysis including:

* accuracy vs coverage tradeoff
* precision, recall, and F1 evaluation
* abstention outcome analysis
* baseline error overlap detection
* margin distribution analysis
* false positive / false negative analysis
* ROC-style evaluation
* analysis of factors affecting confidence (question length and number of answer choices)

The notebook also generates all visualizations used in the final report.

---

• **V2 - CS 175 truth_or_silence_(Verbalized confidence).ipynb / V2 - CS 175 truth_or_silence_(Verbalized confidence).py**
~250–350 lines

Implements an alternative approach where the model is prompted to **verbalize its confidence** in its answer. This notebook evaluates whether the self-reported confidence values correlate with answer correctness.

The analysis includes:

* distribution of verbalized confidence values
* comparison between confidence scores and answer accuracy
* histogram visualization of confidence calibration

Results from this experiment showed that verbalized confidence signals were **poorly calibrated**, which motivated focusing the main analysis on token probability margins instead.

---

• **Helper functions implemented in the notebooks**
~150–200 lines

Includes functions written by our team for:

* prompt construction for multiple-choice questions
* token probability extraction
* margin computation
* prediction evaluation
* threshold-based abstention
* precision/recall/F1 calculation
* experiment logging and visualization
