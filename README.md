
# STA496H1 Summer 2022

## Bayesian Analysis of Normalizing Flow Models using Importance Sampling

- Scott Schwartz
- Yichen Ji [accepted]
- Haining Tan [accepted]
- Ryan Wang [accepted]
- Annie Lu [offered]
- Eric Jiang [accepted]
- Neal Jin?
- Rose Xiaoxuan?
- Eric Zhu [deferred]
- Xin Peng [deferred]



Following on the rapid advancements in deep learning using neural networks, an extremely interesting area of research addresses the quantification of uncertainty in the context of neural networks.  This topic has seen exciting (and sometimes contentiously debated) success under the label of Bayesian Deep Learning, with examples such as "Bayes by Backprop" (Blundell et al.) and "Gaussian Process Approximation using Dropout" (Gal et al.) demonstrating the great potential of applying Bayesian considerations in the context of neural networks. This research project will briefly review existing Bayesian Deep Learning ecosystem and the key methodologies upon which it is based, e.g., Variational Inference. Then, in a meeting of "classic" with "new", the research will consider how a Bayesian Analysis might proceed for the Normalizing Flow (i.e., bijection) class of neural network models based on Importance Sampling.

Normalizing Flows are a widely used methodology which can approximate an arbitrarily complex data distribution by applying a sequence of invertible smooth (change of variables) transformations to a multivariate normal distribution. Bayesian posterior analysis can proceed with Importance Sampling using the prior as the proposal function and the likelihood evaluation as the importance weights. By employing a Bayesian posterior analysis in this context we intend to capture epistemic uncertainty inherent in model fitting, and by using a Normalizing Flow as a likelihood we intend to capture aleatoric uncertainty inherent in data generating mechanisms (as well as produce the required importance weights). The crux of the issue then is the obligatory (and sometimes adversarial) question of "What is your prior?" that Bayesians everywhere are more than familiar with; however, here the answer also crucially drives the computational cost of the proposed Importance Sampling method. It is expected that the research will begin by looking into the suitability of the SWAG posterior approximation methodology (Wilson et al.) as an "empirical Bayes prior" which can then be fine-tuned on the basis of Importance Sampling. The work will be done in Python using TensorFlow and TensorFlow Probability neural networks.

### Required Student Prerequisites:

Students must have experience working in TensorFlow (or PyTorch, etc.) as evidenced by course or portfolio work.  Students must have a solid understanding of Bayesian Analysis and familiarity with applied computation, i.e., as should generally be the case for students who have taken the appropriate advanced coursework.  Strong interest in Bayesian, neural network, and computational methodology is of course preferred. And enthusiasm for and comfort with working through challenging problems in new areas is of course very beneficial for research work.

# Roadmap

0. Importance Sampling (IS)
   - [An initial orienting question](BayesImportanceSampling.ipynb)
   - [A quite nice answer to review](Importance_Sampling.pdf)
   - [Relevant reading Givens/Hoeting Chapter 6.4.1](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)

     <details><summary>[Click] Points of Consideration</summary>
     <br><br>
     1. Importance Sampling is often presented as a variance reduction technique.  Since our desire is to produce a (weighted) sample representation of the posterior, we do not immediately have an estimates for which variance reduction might apply; however, Bayesian credible intervals are based on percentile ranks within posterior samples, so might we expect variance reduction with respect to our estimates of Bayesian credible intervals?
     <br><br>
     2. Generally speaking, what is the difference between unnormalized and normalized importance weights? What are the strengths of each of them that might make them better for a given application context?  Hint: consider the bias characterized in eq. 6.42 in the provided reading.
     <br><br>
     3. Specifically for our Bayesian context, what is the computational difference between the mathematical forms of the unnormalized and normalized importance weights? Stating this question more directly, what can be said about the marginal likelihood with respect to unnormalized and normalized importance weights and what does that mean, computationally? Hint: "Unfortunately, marginal likelihoods are generally difficult to compute" (https://en.wikipedia.org/wiki/Marginal_likelihood) <br>
     </details>

1. Variational Inference (VI)

   - Course Notes [Section 3.0.2 (Bayesian) Variational Inference](https://colab.research.google.com/drive/1bFm8kKsFjsVITAScCQeSh2Tn59uk9yGr#cell-opt-VI)

     <details><summary>[Click] To make the link above work...</summary> Remove the (annoyingly) appended "=" at the end of the address and you'll link directly to the intended section</details>
   - Review Paper: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

   1. Landmark paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
   
     - [Keras Documentation](https://keras.io/examples/generative/vae/)
     - [Open Source MNIST](https://danijar.com/building-variational-auto-encoders-in-tensorflow/)
     - [Open Source Fashion MNIST](https://learnopencv.com/variational-autoencoder-in-tensorflow/)
     - Etc.
     <br><br>

   2. Landmark paper: [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) (and perhaps see also [The Local Reparameterization Trick](https://arxiv.org/abs/1506.02557))

    - a
    - b
    - c

2. VI in TensorFlow

   - AutoEncoders
     
