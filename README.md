
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

0. Importance Sampling (IS) [(Givens/Hoeting Chapter 6.4.1)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Some initial orienting questions and partial responses for your review](files/BayesImportanceSampling.ipynb)

1. Empirical Bayes (EB) ([Introduction and Commentary from Haining Tan](files/Empirical_Bayes.pdf))
   - OPEN QUESTION: what impact (if any) do Empirical Bayes piror specifications have on estimation based on importance sampling?

2. Variational Inference (VI) [(Schwartz STA410 3.0.2)](https://colab.research.google.com/drive/1bFm8kKsFjsVITAScCQeSh2Tn59uk9yGr#cell-opt-VI) <details><summary>[Click] To make the link above work...</summary> Remove the (annoyingly) appended "=" at the end of the address and you'll link directly to the intended section</details>

   1. [Introduction to VI in TensorFlow](files/DenseVariational.ipynb) (based on [this TensorFlow "article"](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html))

   2. [Bayesian Neural Networks (BNN) / Bayes by Backprop (BBB)](files/BayesByBackprop.ipynb) relative to Bayesian (posterior) analysis and Markov chain Monte Carlo (MCMC) with PyMC3
      - Landmark paper: [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) (and perhaps see also [The Local Reparameterization Trick](https://arxiv.org/abs/1506.02557))
    
   3. Review Paper: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) 
    
        > Hopefully the preceeding materials have been sufficient and this is just a reference at this point.

   4. Landmark paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

        > Autoencoders are a seminal application of VI; however, they are not focussed on uncertainty charcterization.
        > Thus, while they serve as a "proof of understanding" exercise of VI, they are tangential to our own efforts.
        > So skip this for now, but if you wish to return to it later see the [Keras Documentation](https://keras.io/examples/generative/vae/),
        > and other open source resources, e.g., for [MNIST](https://danijar.com/building-variational-auto-encoders-in-tensorflow/)
        > and [Fashion MNIST](https://learnopencv.com/variational-autoencoder-in-tensorflow/)

3. Gaussian Processes ([Introductory Lecture](https://www.youtube.com/watch?v=4vGiHC35j9s))
    1. [The Gaussian Process (GP)](files/GaussianProcesses.ipynb) and Stochastic Processes 
    3. [MC-Dropout Approximates a GP](files/DropoutBayes.ipynb), including
        1. Dropout is not VI Bayes [[HMG2017](https://arxiv.org/abs/1711.02989) rejects [KSW2015](https://arxiv.org/abs/1506.02557)]
        2. MC-Dropout is Bayes (when it approximates a posterior which is a GP)
        
           > From the ***tour de force*** [Thesis](https://t.co/YXw7UX7I9P?amp=1) and resulting ***landmark*** [Manuscript](https://arxiv.org/abs/1506.02142) and [Appendix](https://arxiv.org/abs/1506.02157)
        4. Concerns regarding MC-Dropout [from Ian Osband ([1](https://mobile.twitter.com/ianosband/status/1014466510885216256?lang=en), [2](https://www.reddit.com/r/MachineLearning/comments/8w0v9m/d_ian_osband_dropout_posteriors_give_bad/), [3](https://www.reddit.com/r/MachineLearning/comments/emt4ke/discussion_research_variational_bayesian/), [4](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf)) and [HMG2018](https://arxiv.org/abs/1807.01969)]
        5. [Dropout isn't Bayesian](https://discourse.pymc.io/t/frequency-of-missing-value-imputation/8809) (and MCMC with Metropolis-Hastings, Gibbs Sampling, and Hamiltonian MC)
    4. [The GP4ML Textbook](http://gaussianprocess.org/gpml/) 
     
        > Hopefully the preceeding materials have been sufficient and this is just a reference at this point.

    <details><summary>[Click] At this stage...</summary> 
   We've seen BNN/BBB and MC-Dropout as characterizing uncertainty in the NN context.<br>
   And we've also seen more traditional Bayesian analysis with MCMC using PyMC.<br>
   Can we add something to the Bayesian Deep Learning (BDL) domain?</details>

4. Normalizing Flows (NF)
    1. MADE autoregressive structure
        - conditional parameter outputs and the chain rule
    2. MAF and IAF and RealNVP
    3. Hamiltonian NF

5. [SWAG](https://www.google.com/search?channel=trow5&client=firefox-b-d&q=swag+bayes+approximation) / [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.06707) / [Stochastic gradient Langevin dynamics](https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics) / [Hamiltonian Flows?](https://arxiv.org/abs/2203.05723) Etc.
    - where "Etc." means identifying and gathering together relevant BDL literature that might be helpful to us
