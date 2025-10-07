---
title: 'Introduction to LLM Profiling'
description: 'A profiling tutorial for Nvidia GPUs with two different GPT-2 workflow'
pubDate: 2022-01-25
author: 'Zehao Lu, Prasenjit Chakraborty'
image: '/blog/intro-to-llm-profiling/nvidia_captain.avif'
tags: [GPU]
---
![Alt text](/blog/intro-to-llm-profiling/nvidia_captain.avif)

# Introduction

# The Computational Core of Modern LLMs
A prerequisite for effective performance analysis is a foundational understanding of the target software's core working mechanisms. To establish a solid baseline for our performance expectations, this section will detail the internal workings of the GPT-2 architecture as implemented in our two C++ codebases: the Eigen-Optimized Kernel [LLM-Eigen](https://github.com/zhangpiu/llm.cpp.git) and the CCCL-Accelerated Engine [LLM-CCCL](https://github.com/gevtushenko/llm.c.git) both originated from legendary Andrej Karpathy’s [llm.c](https://github.com/karpathy/llm.c.git). We will illustrate how architectural and implementation choices shape the runtime behavior and performance characteristics of transformer-based models. This foundational understanding will help establish clear performance expectations and guide the deeper profiling and analysis discussed in later sections. Readers already familiar with GPT-2 internals and model execution pipelines may choose to skim this section and proceed directly to the detailed performance investigation.
To begin, let’s focus on the training forward pass while ignoring batching for simplicity, as depicted in the first layer of the GPT-2 architecture diagram.
![First layer of GPT2](/blog/intro-to-llm-profiling/gpt2-layer.png)

The model begins by converting the raw input text into a format suitable for computation, by breaking down into discrete tokens. These tokens are then mapped to continuous, high-dimensional vectors known as embeddings. This process, often handled via a lookup table, transforms sparse token IDs into dense, meaningful input vectors. The final input vector that enters the main transformer block combines these token embeddings with positional encodings. For both implementations, LLM-Eigen and LLM-CCCL, the main entry point is the **train_gpt2.cu** file. The training loop orchestrates the overall training process. In each iteration of this loop, a training sample and its corresponding target are loaded from the dataset, after which the forward pass of the model is executed. This forward pass forms the computational backbone of the training process, where most of the profiling and performance analysis will be focused in subsequent sections.

The code begins by allocating memory for activations, the intermediate tensors produced during forward and backward passes. The two implementations differ significantly in how they handle these allocations. The LLM-CCCL version allocates all activations in a single contiguous memory block, which reduces fragmentation and allows for efficient GPU memory access and management. In contrast, the LLM-Eigen implementation uses lazy allocation through the Eigen library’s LazyAllocate mechanism, which allocates memory on demand as tensors are created. While lazy allocation provides greater flexibility when working with varying tensor shapes and sizes, for example, during inference with dynamic sequence lengths, it may not yield the most optimized implementation for fixed-size training, where a bulk contiguous buffer is typically more efficient.

The training cycle begins with the entry into the Model's Forward Function. As illustrated in the architecture diagram , the initial processing steps are consistent: the input sequence is first tokenized, and the resulting token embeddings are summed with the positional encodings. This resultant vector, forms the initial input vector for the first transformer layer. The model then iterates through all transformer layers in sequence. Except for the first layer, the output of the preceding layer serves as the input to the next layer in the sequence.

A subtle but important difference lies in how the core operations within each transformer layer are structured and executed. In the LLM-Eigen version, each layer is encapsulated as a C++ class, with its constituent blocks implemented as member functions or objects of that class. In contrast, the CCCL version utilizes a flat, procedural execution of the blocks in every layer. Another key distinction lies in the implementation of sub-blocks within each transformer layer. The LLM-Eigen version adopts a mathematical operation-by-operation approach. Each core step of the calculation is mapped directly to an Eigen operation which, in turn, results in multiple, distinct kernel launches on the device. The LLM-CCCL codebase employs a more traditional high-performance strategy, where a single, fused kernel performs the entire computation of a sub-block. All these implementation choices have a direct impact on performance, and their effects will be analyzed in detail in the performance profiling section later in this blog.

The table below highlights the specific code sections corresponding to each implementation choice in both versions, providing an easy reference for readers to correlate source code with the architectural flow shown in the accompanying figure. Together, the table and figure serve as a practical guide to understanding how a modern LLM is structured and executed at a low level.

This reference will also be valuable in the next section, where we develop the roofline performance model for each of these blocks to quantify and compare their computational efficiency and memory behavior.

| Sub Blocks | LLM-Eigen | LLM-CCCL |
|------------|-----------|----------|
|Model Forward Function|ForwardGPU @ gpt.hpp|gpt2_forward @ train_gpt2.cu|
|Token Embedding & Positional Embedding|__Forward @ gpt.hpp|encoder_forward @ train_gpt2.cu|
|LLM Layer|Block::Forward @ gpt.hpp|encoder_forward @ train_gpt2.cu|
|LayerNorm|nn::LayerNorm::Forward @ nn.hpp|layernorm_forward @ train_gpt2.cu(layernorm_forward_kernel3)|
|QKV Linear Projection|CausalSelfAttention::Forward @ gpt.hpp(c_attn_->Forward)|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|Self Attention: QKT|nn::MatMul::Forward @ gpt.hpp|attention_forward @ train_gpt2.cu(cublasSgemmStridedBatched)|
|Self Attention: Softmax|nn::Softmax::Forward @ gpt.hpp|softmax_forward_kernel5 @ train_gpt2.cu|
|Self Attention: Value Matmul|nn::Softmax::Forward @ gpt.hpp|attention_forward @ train_gpt2.cu(cublasSgemmStridedBatched)|
|O Linear Projection|nn::MatMul::Forward @ gpt.hpp|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|Residual|nn::Residual::Forward @ gpt.hpp|residual_forward @ train_gpt2.cu|
|FeedForward: MLP1 & MLP2|MLP::Forward @ gpt.hpp nn::Linear::Forward|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|FeedForward: GeLU|nn::NewGELU::Forward @ gpt.hpp|gelu_forward @ train_gpt2.cu|

# Basics of GPU

# CUPTI Introduction

# Performance Analysis

