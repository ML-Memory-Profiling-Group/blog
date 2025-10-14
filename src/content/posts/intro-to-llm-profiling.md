---
title: 'Introduction to LLM Profiling'
description: 'A profiling tutorial for Nvidia GPUs with two different GPT-2 workflow'
pubDate: 2025-10-14
author: 'Zehao Lu, Prasenjit Chakraborty'
tags: [GPU]
---

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

# CUPTI

CUPTI is a set of API that enables developers to both retrieve hardware counters from NVidia GPUs and trace the host-side activities on CUDA. It serves as the foundation of NSight Compute, the official GPU profiler provided by NVidia. With CUPTI, independent developers can develop customized profilers that leverage the same sets of metrics and derive their own specialized insights through custom data processing

In the big picture, CUPTI has two key functionalities:

* Tracing: collecting host-side activities, like kernel launches and memset, etc.  
* Profiling: collecting hardware counters and other derived metrics like throughput.

It can also be divided into multiple sets by the way it collects data, including

* the Activity API,  
* the Callback API,  
* the Host Profiling API,  
* the Range Profiling API,  
* the PC Sampling API,  
* the SASS Metric API,  
* the PM Sampling API,  
* the Checkpoint API,  
* the Profiling API,

For this blog, we built a GPU profiler on top of the Activities API and Range Profiling API to trace and profile. We won’t talk about the details about how our profiler is built, but rather we will focus on the performance data collected. In case you are interested, the profiler is available here(github\_link) and the corresponding tutorial in detail is also available here(blog\_link). Simply speaking, our profiler is able to separate code into logical blocks called “range” by wrapping the code with push and pop range functions, which defines the range we are interested in and would like to collect data from. Here is a sample code of how we wrapped and timed the range:

```cpp
#ifdef TRAINING_FORWARD 

      if(curr_step == 1 && curr_fwd_block == 1)

        GmpProfiler::getInstance()->pushRange("MLP", GmpProfileType::CONCURRENT_KERNEL);

#endif

      GMP_TIMED("MLP", mlp_->Forward(ln2_y_2d_const, mlp_y_2d));

#ifdef TRAINING_FORWARD

      if(curr_step == 1 && curr_fwd_block == 1)

        GmpProfiler::getInstance()->popRange("MLP", GmpProfileType::CONCURRENT_KERNEL);

#endif
```

GmpProfiler::getInstance()-\>pushRange/popRange is the API of our profiler that collects both traces and metrics and defines the range. GMP\_TIMED is simply a macro to use C++ chrono to get the CPU time spent by the wrapped portion of the code and this is where the wall-clock time comes from.

All the activities records and metrics collected will be grouped by range name and accumulated or averaged among all the kernels’ data within the range, so that we can understand how each phase of the LLM performs.

# Performance Analysis
## Kernel Invocations

Kernel launch is where CUDA assigns computation tasks to the GPU. The number of kernels and the size of blocks and grids can produce profound impact on system performance. Ideally, each kernel should have enough blocks and threads so that it doesn’t under utilize the compute resources. On the other hand, too many blocks, threads or kernel launches themselves will accumulate overheads and severely hurt the overall performance. In this section, we will see how the two implementations differ and why they differ. In later sections, we will discuss how these differences impact the performance
![][kernel-num]  

If we compare the two implementations in forward process, it is obvious that the Eigen version launches more kernels, especially in the attention. In the cccl version, it is implemented in a way that all the computation is fused in one global kernel. Only big ranges, like mlp and attention, will employ some helper device kernels. Whereas the kernel launches in Eigen are not explicit. In other words, the number of kernels and their grid/block sizes are dependent on the implementation of Eigen. It is not guaranteed that each assignment of eigen will generate only one kernel, and that’s why generally the Eigen version launches more kernels than the cccl version. This is the reason why most ranges of Eigen llm.cpp launch more kernels than the cccl one.

However, this doesn’t explain why the Eigen llm.cpp has a huge attention range with 440 kernel launches, which far exceed other ranges. The main reason for it is the improper use of for loops in the code. Here is a sample pseudo code of how the attention in eigen llm.cpp is implemented:

```cpp
for (int b = 0; b < B; ++b)  
{  
    for (int h = 0; h < NH; ++h)  
    {  
        // Calculate Q K V

        // Calculate QK^T

        // softmax

        // att * V  
    }  
}  
```

This piece of code is looping over batch size and number of heads. There are two matrix multiplications and softmax in each iteration, which will produce quite a lot of kernels. With B=4 and NH=12, all these kernels are repeated 48 times, so no surprise so many kernels are launched. This exemplifies a pitfall of GPU programming. It is common and fine to use for loops when we write programs for CPU, but the misuse of for loops on GPU programs can heavily downgrade the performance. We will discuss the performance drop in the next section.

|  | eigen forward | cccl forward |
| :---- | :---- | :---- |
| min | 1 | 16 |
| max | 3144 | 1536 |
| mean | 2.741594966 | 158.9488133 |
| median | 4 | 320 |
| avg warp/block | 23.51575188 | 8.282119391 |

Another important thing to note about kernels is the grid size and block size. Here we have provided the stats of all implementations. The data shows that both directions of eigen implementation launches most of the kernels with merely single-digit blocks, whereas cccl version launches with about 160 blocks on average. It indicates that most of the kernels launched by the eigen llm.cpp are heavily underutilizing the compute resources of the GPU. The device we are testing on is A100 with 108 SMs. Ignoring the stalls caused by data dependencies and assuming that there is only one stream on the GPU, given that one block can not reside across multiple SMs, we need at least 108 blocks to ensure that each SM has been assigned at least 1 block. Since the eigen forward kernels only have an average grid size of 2.7, there is only around 2% of SMs utilized, which is surprisingly low. Note that here we are assuming blocks will be assigned to inactive SMs first instead of sharing SM with other blocks because this policy can maximize the SM utilization. The exact block scheduling policy isn’t publicly specified, so we inferred the behavior empirically and speculated the least blocks requirement to fully utilized SMs. The actual number of blocks needed should be greater or equal 108, so it won’t change our conclusion.

## Wall Clock Time and GPU Time

Time consumption is one of the key metrics of any type of program. Here we will analyze the wall clock time and the GPU time of the two implementations. GPU time is simply the time GPU takes to finish the calculation of all threads. It is collected from hardware counters per kernel through CUPTI. To get the GPU time of a range, we add the gpu\_\_time\_duration.max of all the kernels in that range. On the other hand, the wall clock time is the duration between the start timestamp to the end timestamp. We used C++ std::chrono::high\_resolution\_clock::now() to wrap the code snippet and get the duration through subtracting the two timestamps. This time is measured on the CPU side, so it will include all the time spent by the code wrapped, including GPU execution time, launch overhead, any housekeeping operations like makespan, makeMatrix, synchronized cuda memory copy/memset, etc. Note that by default the kernel launch is asynchronous, so launching a kernel will only push it into a queue, so this method won’t capture the GPU time. With CUPTI range profiling enabled, all kernels will be executed synchronously so that only after the GPU finishes its job will the kernel return. That’s why our wall clock time includes the GPU time.

We should notice that both times will rise significantly if CUPTI is involved. We have verified it through comparing the wall clock time without profiling and GPU time with profiling. If the GPU time we measured doesn’t contain the overhead, the profiled GPU time should not exceed the non-profiled wall clock time because the wall clock time includes more items including overheads and bookkeeping than GPU time. The data shows that for some ranges do have a higher profiled GPU time than the non-profiled wall clock time. Therefore all the data we have shown will include CUPTI overheads.

![][forward-wallclock-time]![][forward-wallclock-time-ratio]![][kernel-num-ratio]![][forward-gpu-time]![][forward-gpu-time-ratio]  
The above charts are the wall clock time(in microseconds) and GPU time(in nanoseconds) of all the ranges and their ratio. It should be very clear that there is a huge overall performance gap between eigen and cccl implementation if we compare the wall clock time. Many factors contribute to these gaps. 

Let’s start from the GPU time. Even without involving any factors on the CPU side, the gap between the two versions is still quite large. The biggest contributors to the GPU time are the attention, lm\_head and softmax\_cross\_entropy. This makes sense because all of them are both compute and memory heavy. The reason for attention to take so much time is because it has large matrix multiplication to get Q, K, V and the correlations. Additionally, softmax is also an expensive operation because it requires the sum and max of each token. Similarly, softmax\_cross\_entropy also calculates softmax, and in addition to that, to get the cross entropy, it needs to calculate the max of all the logits. Although the lm\_head is essentially a matrix multiplication, it operates over the vocabulary dimension rather than the embedding dimension. This makes the weight matrix extremely large, resulting in significantly higher computation time and memory usage. Consequently, these ranges are the most resource-intensive among all and take most of the time.

If we compare the ratio of GPU time, we can find that the attention increased from about 10% to 35%, the biggest jump among all the ranges. This suggests that it is the main contributor to the gap of GPU time. Given that eigen llm.cpp launches 440 kernels for attention, it has a huge number of metrics to collect because all metrics are collected for every kernel. With that said, it’s fair to believe that the big jump of the GPU time for attention is due to CUPTI overhead. Layer norms also take more GPU time relative to the cccl version. That’s probably because the kernel of the layer norm only has one block. The low SM utilization and dram throughput can cause the GPU execution time to increase significantly.

On the other side, the gap of the wall clock time enlarges, suggesting that there are factors other than the CUPTI and GPU usage that further drops down the overall performance. We suggest that the additional dropdown is probably caused by launch overhead. To demonstrate this, we present a ratio chart of Eigen’s kernel launches. By comparing the kernel launch ratio with the wall-clock time ratio, a clear linear relationship emerges, strongly indicating that launch overhead is probably the primary contributor to the wall-clock time in eigen llm.cpp.

To further explain the difference, we conducted an experiment using a simple CUDA program and did some calculations. In this program, we launched 440 kernels with minimum operations. The average wall clock time we got is around 130 microseconds per kernel, which should mostly be launch overhead.   **The number is not matching\! The data we got is around 100k microsecond, not the same magnitude. My previous calculation matches 100k, and I copied the logs to the excel. However, I can no longer reproduce that number.. Figuring out what is causing the discrepency...**

Finally, the developer of the cccl llm.cpp implementation applies several optimizations to improve cache efficiency — for instance, using cache streaming to allow one-time data to bypass the cache, and employing reverse iteration to increase cache hits at the tail of arrays. In contrast, the Eigen version lacks such low-level optimizations, at least from the user side. As a result, the cccl version achieves higher cache hit rates and fewer dram accesses, which directly contributes to its shorter execution time.

## SASS Instructions

SASS (Streaming Assembler) is the low-level assembly language executed by NVIDIA GPUs.  
It’s the final compiled form of CUDA kernels. CUPTI allows us to collect all kinds of SASS, but in this blog, we will focus on the global load/store instructions and the bytes it reads/writes. Here is the sample SASS instruction data of the residual range:

| metrics | eigen | cccl |
| :---- | :---- | :---- |
| smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum | 1572864 | 1572864 |
| smsp\_\_sass\_data\_bytes\_mem\_global\_op\_st.sum | 786432 | 786432 |
| smsp\_\_sass\_inst\_executed\_op\_global\_ld.sum | 3072 | 12288 |
| smsp\_\_sass\_inst\_executed\_op\_global\_st.sum | 1536 | 6144 |

* smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum is the total number of global load warp instructions issued. Note that this doesn’t include atomic or shared loads,which are collected in other metrics. 
* smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum represents the actual data loaded by the SASS instructions. 

Store instructions are similar to the load instructions. We choose residual as an example because it is relatively straightforward. Residual only contains an element-wise add operation, like C=A+B. The GPU should load two input matrices and store the output matrix. That’s why there are 2x load instructions and bytes compared to stores. You may also notice that even though both implementations load/store the same amount of data, the eigen version executed only ¼ instructions of the cccl version. This is because the eigen version uses vectorized loads for contiguous elements. Each global load will load 128 bits of data, i.e. 4 floats, whereas the usual global load will only load 32 bits, so one float each time. We can verify this by dividing bytes by instructions, and then divide by 32, to get the average bytes loaded per thread. The result of the Eigen version is 16 bytes per instruction, i.e. 128 bit per load or store. For the cccl version, this number is reduced to 32bit/inst, indicating that each instruction only loads one float. So this vectorization is one optimization Eigen implicitly does for loads and stores automatically, which can reduce redundant instructions and issue overheads.

## L1, L2 and dram accesses

When SASS loads and stores are executed in the thread, they will be coalesced with other instructions executed by other threads within the warp and be sent to L1. If it is missed,L1 will forward requests further to L2. If still missed, L2 will send requests to the dram in sectors. Here are the metrics we are interested in. We will show the residual range as an example:  
![][residual-accesses]

* L1tex\_\_t\_requests\_pipe\_lsu\_mem\_global\_op\_ld.sum: approximately the global load requests L1 cache received from the warps. The “lsu” implies that the requests are from the load store unit. The “approximately” means there might be requests other than global loads, like LDSTS instructions, but this is not so important and is beyond the scope of this blog. In most cases, you can find that it ballparkly matches the number of SASS load requests.  
* L1tex\_\_t\_sectors\_pipe\_lsu\_mem\_global\_op\_ld.sum is the number of sectors accessed by the requests received by L1 cache. In general this metric should be greater or equal to smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum/32. Since the warps access contiguous and 32-aligned addresses in residual range, it exactly matches that result.   
* Lts\_\_t\_requests\_srcunit\_tex\_op\_read.sum: the requests L2 cache received from L1.  
* Lts\_\_t\_sectors\_srcunit\_tex\_op\_read.sum: the sectors accessed by the L2 requests from L1. Each request can contain 1\~4 sectors. This metric also represents how many sectors L1 missed.  
* Dram\_\_sectors\_read.sum: sectors requested from L2 because of L2 misses.Note that though dram sends the data in bursts, the unit of these metrics is 32 byte sectors, so these metrics should be the actual bytes loaded divided by 32\.

In general, from L1 to L2 to dram, the sector metrics should gradually reduce. The higher the hit rates, the more they reduce. Here we can see L1 sector loads and L2 sector loads are the same. This is because all the addresses in residual will only be accessed once, so the hit rate is 0%. All the sectors being accessed in L1 are forwarded to L2.  Previously we mentioned that the eigen llm.cpp is utilizing vectorized load, and that’s why the L1 requests of eigen are relatively low compared to the cccl version. There are also different dram sector reads between two implementations. This is probably because of L2 partitions or the activations that remain in L2 since L2 will not be flushed between kernels.

## Dram throughput

Finally, after requests have been filtered through L1 and L2, they reach dram, whose bandwidth greatly affects the overall performance of the system. CUPTI provides dram\_\_throughput.avg.pct\_of\_peak\_sustained\_elapsed, a percentage showing how much of theoretical sustained peak throughput one kernel can use, but this metrics only measures per kernel throughput. If we calculate the average throughput through adding all the metrics in range and divide by number of kernels in range, in some extreme cases, it may show wrong throughput because it loses the information of time. For example, if we have 1 kernel that heavily utilizes 100% throughput for an hour and 99 kernels use 0% in just 1 second, we will get an average usage of 1%, which looks pretty off. So instead of directly averaging the throughput metrics provided by CUPTI, we calculate the overall throughput by doing total\_accessed\_sectors\*32/sum\_of\_gpu\_time. More specifically, (dram\_\_sectors\_read.sum \+ dram\_\_sectors\_write.sum) \* 32 / gpu\_\_time\_duration.max. Here is the data we produced:

![][dram-throughput]  
From the chart, we can find that generally the cccl version consumes more dram throughput than the eigen one. Previously we talked about the low grid size of the eigen version. If the kernel is short of blocks, it will use few SMs, and the number of warp instructions issued per cycle will be limited because most SMs are not active. Remember the average grid size for eigen implementation is 2.7. This makes most of the SM inactive and not being able to issue store or load commands and leave the remaining throughput wasted. Another reason might be  the GPU time of the range. If we refer back to the prior section, we can find that the cccl version takes less than 1/10 GPU time of the eigen ones. Our equation indicates that the denominator is the GPU time. With the same amount of dram loads and stores, the bandwidth will be multiple times higher if the time is as short as that. The reduced time of the cccl llm.cpp indicates a better usage of dram bandwidth over leaving the bandwidth wasted for a long time.

Furthermore, we can see that in both implementations, the layer norms barely accessed the dram. This is expected because the calculation of the norms doesn’t involve any parameters. All it needs is to load the previous activations and store the result norm. As L2 will not be flushed across kernels, the activation produced by the previous range should still reside in the L2. Therefore even if there will be SASS loads and L1 requests, these accesses will be filtered out by L2 and keep the dram intact. That’s another reason to explain layer norms use such a little throughput in both implementations other than the grid size.  

[kernel-num]: </blog/intro-to-llm-profiling/kernel-num.png>
[forward-wallclock-time]: </blog/intro-to-llm-profiling/forward-wallclock-time.png>
[forward-wallclock-time-ratio]: </blog/intro-to-llm-profiling/forward-wallclock-time-ratio.png>
[kernel-num-ratio]: </blog/intro-to-llm-profiling/kernel-num-ratio.png>
[forward-gpu-time]: </blog/intro-to-llm-profiling/forward-gpu-time.png>
[forward-gpu-time-ratio]: </blog/intro-to-llm-profiling/forward-gpu-time-ratio.png>
[residual-accesses]: </blog/intro-to-llm-profiling/residual-accesses.png>
[dram-throughput]: </blog/intro-to-llm-profiling/dram-throughput.png>