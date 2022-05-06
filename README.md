
# STA496H1 Summer 2022

## Bayesian Analysis of Normalizing Flow Models using Importance Sampling

Scott Schwartz

| Research/Reading | Auxiliary/Deferred |
|-|-|
| Yichen Ji   | Neal Jin |
| Eric Jiang  | Jianing Zhang |
| Haining Tan | Xin Peng |
| Ryan Wang   | Eric Zhu |

Following on the rapid advancements in deep learning using neural networks, an extremely interesting area of research addresses the quantification of uncertainty in the context of neural networks.  This topic has seen exciting (and sometimes contentiously debated) success under the label of Bayesian Deep Learning, with examples such as "Bayes by Backprop" (Blundell et al.) and "Gaussian Process Approximation using Dropout" (Gal et al.) demonstrating the great potential of applying Bayesian considerations in the context of neural networks. This research project will briefly review existing Bayesian Deep Learning ecosystem and the key methodologies upon which it is based, e.g., Variational Inference. Then, in a meeting of "classic" with "new", the research will consider how a Bayesian Analysis might proceed for the Normalizing Flow (i.e., bijection) class of neural network models based on Importance Sampling.

Normalizing Flows are a widely used methodology which can approximate an arbitrarily complex data distribution by applying a sequence of invertible smooth (change of variables) transformations to a multivariate normal distribution. Bayesian posterior analysis can proceed with Importance Sampling using the prior as the proposal function and the likelihood evaluation as the importance weights. By employing a Bayesian posterior analysis in this context we intend to capture epistemic uncertainty inherent in model fitting, and by using a Normalizing Flow as a likelihood we intend to capture aleatoric uncertainty inherent in data generating mechanisms (as well as produce the required importance weights). The crux of the issue then is the obligatory (and sometimes adversarial) question of "What is your prior?" that Bayesians everywhere are more than familiar with; however, here the answer also crucially drives the computational cost of the proposed Importance Sampling method. It is expected that the research will begin by looking into the suitability of the SWAG posterior approximation methodology (Wilson et al.) as an "empirical Bayes prior" which can then be fine-tuned on the basis of Importance Sampling. The work will be done in Python using TensorFlow and TensorFlow Probability neural networks.

### Required Student Prerequisites:

Students must have experience working in TensorFlow (or PyTorch, etc.) as evidenced by course or portfolio work.  Students must have a solid understanding of Bayesian Analysis and familiarity with applied computation, i.e., as should generally be the case for students who have taken the appropriate advanced coursework.  Strong interest in Bayesian, neural network, and computational methodology is of course preferred. And enthusiasm for and comfort with working through challenging problems in new areas is of course very beneficial for research work.

# Roadmap

0. [Importance Sampling (IS) (Givens/Hoeting Chapter 6.4.1)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Some initial orienting questions and partial responses for your review](files/BayesImportanceSampling.ipynb)

1. Empirical Bayes (EB) ([Commentary from Haining Tan](files/Empirical_Bayes.pdf))
   - OPEN QUESTION: what impact (if any) do Empirical Bayes piror specifications have on estimation based on importance sampling?

2. [Variational Inference (VI) (Schwartz STA410 3.0.2)](https://colab.research.google.com/drive/1bFm8kKsFjsVITAScCQeSh2Tn59uk9yGr#cell-opt-VI) <details><summary>[Click] To make the link above work...</summary> Remove the (annoyingly) appended "=" at the end of the address and you'll link directly to the intended section</details>

   1. [ Introduction to VI in TensorFlow](files/DenseVariational.ipynb) (based on [this TensorFlow "article"](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html))

   2. Landmark paper: [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) (and perhaps see also [The Local Reparameterization Trick](https://arxiv.org/abs/1506.02557))

    - Another VI example
    - Bayesian Posterior Analysis 
    - Markov Chain Monte Carlo (MCMC): 
      - Metropolis-Hastings, Gibbs Sampling, and Hamiltonian MC
    
   3. Review Paper: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

      - This may or may not be a useful resource at this stage

   4. Landmark paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

     - Autoencoders are a seminal application of VI; however, they are not focussed on uncertainty charcterization
     - Thus, while they serve as a "proof of understanding" exercise of VI, they are tangential to our own efforts
     - So skip this for now, but if you wish to return to it later see the [Keras Documentation](https://keras.io/examples/generative/vae/),
       and other open source resources, e.g., for [MNIST](https://danijar.com/building-variational-auto-encoders-in-tensorflow/)
       and [Fashion MNIST](https://learnopencv.com/variational-autoencoder-in-tensorflow/)

3. Dropout as Bayes

  0. Gaussian Processes (GP)
  1. Dropout as GP approximation is Bayes: 
     1. Dropout can model Bayes, but Bayes cannot model Dropout
     2. Other possible limitations and noise
  2. Dropout IS NOT VI Bayes

4. Normalizing Flows (NF)
  1. MADE autoregressive structure
     - conditional parameter outputs and the chain rule
  2. MAF and IAF and RealNVP
  3. Hamiltonian NF

5. SWAG

6. Stochastic Normalizing Flows


  


  
