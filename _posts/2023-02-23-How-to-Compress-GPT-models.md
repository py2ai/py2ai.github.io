---
layout: post
title: How to compress GPT models
mathjax: true
featured-img: 26072022-python-logo
summary:  In this tutorial we will learn to compress GPT and make a working chat GPT app
---

 
Hello friends! Yes, it is possible to compress GPT models to reduce their size and computational complexity. Model compression techniques can be used to reduce the number of parameters and operations required by the model without significantly impacting its performance. But before that let's see what is GPT?

# GPT (Generative Pre-trained Transformer)

GPT (Generative Pre-trained Transformer) model consist of following components:

* Transformer architecture: GPT models are built using the Transformer architecture, which is a type of neural network that excels at processing sequences of variable length. The Transformer architecture is composed of multiple layers of self-attention and feedforward networks, which allows the model to attend to different parts of the input sequence and learn complex relationships between them.

* Pre-training task: GPT models are typically pre-trained on a large corpus of text using a language modeling objective, such as predicting the next word in a sequence given the previous words. Pre-training helps the model learn general language patterns and structures that can be fine-tuned for downstream tasks.

* Fine-tuning layers: After pre-training, the GPT model can be fine-tuned on a specific downstream task, such as text classification, question answering, or text generation. Fine-tuning involves adding task-specific layers on top of the pre-trained model and updating the model parameters on task-specific data.

* Embedding layer: The input to a GPT model is typically encoded as a sequence of tokens, which are converted into dense vector representations using an embedding layer. The embedding layer maps each token to a continuous vector space, which allows the model to capture the semantic relationships between the tokens.

* Positional encoding: The Transformer architecture does not inherently capture the order of the input sequence, so GPT models use a positional encoding scheme to incorporate positional information into the model. Positional encoding adds a learnable vector to each input token that encodes its position in the sequence.

* Vocabulary and tokenization: GPT models operate on a fixed vocabulary of tokens, which are typically selected based on their frequency in the training corpus. Text input is preprocessed by tokenizing it into a sequence of these vocabulary tokens, which are then fed into the GPT model.

* Together, these components allow GPT models to process and generate natural language text, making them powerful tools for a wide range of natural language processing tasks.



There are various approaches to compressing GPT models, including:

* Pruning: This involves removing the least important parameters from the model, based on their magnitude or sensitivity to perturbations. Pruning can significantly reduce the size of the model while preserving its accuracy.

* Quantization: This involves reducing the precision of the model's weights and activations, such as converting them from 32-bit floating point to 16-bit or 8-bit integers. This reduces the model's memory and computation requirements, but may slightly reduce its accuracy.

* Knowledge distillation: This involves training a smaller model to mimic the behavior of a larger, more complex model. The smaller model is trained using the larger model's predictions as targets, which allows it to learn from the larger model's knowledge. This can result in a smaller model with comparable performance to the larger one.

* Low-rank factorization: This involves decomposing the weight matrices of the model into lower-rank matrices, which reduces the number of parameters and computations required. This can be achieved using techniques such as singular value decomposition (SVD) or tensor decomposition.

Grouped convolutions: This involves grouping the input and output channels of convolutional layers, which can reduce the number of parameters and computations required while preserving the spatial relationships between the features.

Overall, model compression techniques can help reduce the computational and memory requirements of GPT models, making them more efficient and practical for deployment in resource-constrained environments.

# Let's use FlexGen to make a chat GPT application

## What is FlexGen

FlexGen is a method for compressing GPT (Generative Pre-trained Transformer) models, which was proposed in a research paper published by Facebook AI in 2021. The goal of FlexGen is to reduce the computational cost and memory footprint of GPT models without significantly affecting their performance.

FlexGen is based on the idea of dynamically adjusting the model's architecture during inference based on the input sequence length. In traditional GPT models, the maximum sequence length is fixed during training and inference, which can lead to high computational and memory costs for long sequences. FlexGen addresses this issue by dividing the model into multiple stages, each optimized for a specific range of sequence lengths.

During inference, FlexGen selects the appropriate stage based on the length of the input sequence, and only computes the necessary layers for that stage. This reduces the computational and memory cost compared to a traditional GPT model that computes all layers for every input sequence.

FlexGen also incorporates other model compression techniques such as weight pruning, quantization, and knowledge distillation to further reduce the size and computational cost of the model.

Experimental results show that FlexGen can achieve significant speedup and memory savings compared to traditional GPT models, while maintaining similar or slightly improved performance on various language modeling benchmarks.

# Make the simple chat GPT like application

## Step 1. Make a miniconda environment for python 3.8

Download the appropriate version of Miniconda for your operating system from the official website: https://docs.conda.io/en/latest/miniconda.html

Install Miniconda by running the installer and following the on-screen instructions.

Open a terminal or command prompt and check that conda is installed and available in your path by running the command:

`conda --version`

Create a new Python 3.8 environment by running the following command:

`conda create --name myenv python=3.8`

Replace `myenv` with the name you want to give your new environment. You can choose any name you like.

Activate the new environment by running the following command:

`conda activate myenv`

Again, replace `myenv` with the name of your environment.

Now you can install packages and work with your Python 3.8 environment. For example, you can install NumPy by running the command:

`conda install numpy`

This will install the latest version of NumPy in your environment.

When you're done working in your environment, you can deactivate it by running the following command:

`conda deactivate`

This will return you to your base environment.

That's it! You now have Miniconda installed and a new Python 3.8 environment set up. You can create additional environments in the same way, each with its own set of packages and Python version.

## Step 2. Install the FlexGen

First of all activate the conda environment `conda activate myenv` for the python3.8.

Download the FlexGen repo and install by following the steps:

```
git clone https://github.com/FMInference/FlexGen.git
cd FlexGen
pip3 install -e .

# (Optional) Install openmpi for multi-gpu execution
# sudo apt install openmpi-bin
```

Now you can run the command to download the model and simply test.

`python3 -m flexgen.flex_opt --model facebook/opt-1.3b`

The command `python3 -m flexgen.flex_opt --model facebook/opt-1.3b` will be running a Python script that uses the FlexOpt model from Facebook AI Research.

FlexOpt is a neural network-based optimization solver that can be used to solve a wide range of optimization problems. It uses a neural network to learn how to solve optimization problems and can provide efficient solutions to problems with large-scale data.

The command "python3 -m flexgen.flex_opt" is used to run the FlexOpt module in Python 3. The "--model facebook/opt-1.3b" argument specifies the location of the model file to be loaded for use in the optimization process. The model file is located in the "facebook/opt-1.3b" directory, and it contains the trained neural network parameters that are used to solve the optimization problem.

Here is a list of some of models to try
```
opt-1.3b
opt-3b
opt-6b
opt-12b
opt-24b
opt-30b
opt-175b
```
Each model has been trained on a large corpus of optimization problems and can be used to solve a wide range of optimization problems. The models vary in size and complexity, with opt-30b being the largest and most complex model with 175 billion parameters.

# Step 3 Run the app.

`python3 chatbot.py --model facebook/opt-1.3b`

# output

```
Human: Hello! What can you do?
Assistant: As an AI assistant, I can answer questions and chat with you.
Human: What is the name of the tallest mountain in the world?
Assistant: Everest.
Human: I am planning a trip for our anniversary. What things can we do?
Assistant: Well, there are a number of things you can do for your anniversary. First, you can play cards. Second, you can go for a hike. Third, you can go to a museum.

```

# Step 4 Run the app. on flask

We will see in future





