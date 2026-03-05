# Markov Chain Text Analysis

This repository contains a Python implementation of Markov Chains for text analysis.

## Features
* **Advanced Text Cleaning:** Pipeline using Regex, NLTK, Wordninja, and a local LLM (Qwen 3.5:2b) to sanitize raw data.
* **1st-Order Markov Model:** Constructs a 27x27 stochastic matrix to map character transition probabilities.
* **Model Evaluation:** Calculates a performance score by testing the transition probabilities against unseen validation text.
* **Higher-Order Markov Chains:** Implements a 3rd-order Markov model to predict character generation based on triplets.
