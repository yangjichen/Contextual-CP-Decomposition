# Contextual-CP-Decomposition

We propose a new decomposition framework called Contextual CP decomposition (CCP) model, which incorporate the additional information to estimate the underlying structures that are more accurate and more interpretable. Here we present code logic and data acquisition methods.



### Overview

Tensor decomposition is among the most important tool for low rank structures extraction in tensor data analysis. In many scenarios, informative covariates from various domains are available, which can potentially drive the underlying structures of the observed data. Classical CANDECOMP/PARAFAC (CP) decomposition method only focuses on the relational information among objects, and cannot make use of such additional information. In this article, we propose a new decomposition framework called Contextual CP decomposition (CCP) model, which incorporate the additional information to estimate the underlying structures that are more accurate and more interpretable. The framework is closely related to several existing methods and make sense in various situations. Moreover, we develop an efficient algorithm for parameter estimation. We demonstrate the advantages of our proposal with comprehensive simulations and real data analysis on GDELT political events study. The empirical results show that our method offers improved modeling strategy as well as efficient computation.



### Prepare the environment

To run this code, you need to install the dependencies as follows,

```pyhton
numpy==1.16.4
tensorly==0.5.0
pandas==0.23.4
gdelt==0.1.10.6
matplotlib==3.0.2
```



### Simulation 

Contains 5 parts：

~~`simulation_v1.py`: Toy model~~

`simulation_v2.py`: Investige the effect of increased dimensions and various SNRs.

`simulation_v3.py`: Compare the elapsed time for **CCP** and **CPALS**.

`simulation_v4.py`: Explore the accuracy of covariate-relevant terms estimation.

`simulation_v5.py`: Examine the robustness of **CCP** to heavy-tailed noise.

`simulation_v6.py`: We investigate the special case that auxiliary information is unavailable, and provide the numerical evidence that CP decomposition is the degenerate case of the **CCP** model. 



### Real Dataset Study

Contains 3 steps：

`GDELT1.py`: Download a subset of GDELT dataset which covers the period from January, 2018 to December, 2018.

`GDELT2.py`: Calculate the number of diplomatic events between the above 20 countries, and reorder the dataset as a tensor.

`GDELT3.py`: Explore the relationship between the country’s geopolitical power and the frequency of diplomatic events by using our algorithms.



### Reference: 

For details of the algorithm and implementation, see our paper:

[1] Jichen yang, Nan Zhang. "Contextual Tensor Decomposition by Projected Alternating Least Squares". Submitted. 

