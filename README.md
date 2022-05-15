
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

# Setup

0. Importance Sampling (IS) [(Givens/Hoeting Chapter 6.4.1)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Some initial orienting questions and partial responses for your review](files/BayesImportanceSampling.ipynb)

      <details><summary>Scott's summary of the 4PM conversation on Monday, May 9.</summary> <br>
      The second half of the meeting was <a href="https://utoronto.zoom.us/rec/share/KiJbYUB1mhXAXn27CzDCbxLfhb-7vWHJlToWB5bkyQd4WdHOHCAZKcAKyakvLnop.d3mAcWux8Evw8Cuy">recorded</a> (and has passcode Sc#1wsPr9#).<br><br>
      In the first half of the meeting Ryan addressed the final question of the notebook regarding distribution quantile estimation and we discussed empirical CDFs and Gentle's view on rank order statistics as the fundamental information contained in a data sample. Thanks to Haining's considerations of what it would mean to integrate an inverse quantile function it became clear to me that quantile estimation cannot be formulated as an integration problem, and so quantile estimation (i.e., analyzing rank order statistics) is something different from MC-integration (which seems quite interesting, but I think for now we'll have to put a pin this topic for a later time).<br><br>
      I do not believe the middle two questions addressed in the notebook (regarding helpful attributes of proposal distributions and the computational distinction between unnormalized and normalized importance weights) were systematically address in our question, but they were each tangentially touched upon to some degree. I.e., respectively, see (b) below, and note that normalized IS weights do not require a (generally very hard) marginal likelihood computation to find the normalizing constant of the posterior, but instead can just be based upon normalized likelihood computations as weights.<br><br>
      The first question of the notebook was addressed on the basis of Eric's questions, from which we were able to discuss (a) Monte Carlo Integration as simply being epectation (an integral) estimation done on the basis of standard statistical (CLT) analysis, (b) that IS affords estimator variance reduction for well-chosen proposal distributions which produce large IS weights only for very small g(θ), and following up on that (c) that IS allows us to have control over what sampling distribution to use which can allow us to avoid "expensive to sample" distributions.<br><br> 
      Other discussion focussed on (1) the "big picture" of the proposed Normalizing Flow BDL method and why it was cute, I mean, chosen. And (2) the history of Bayesian Deep Learning (BDL) and why it exists. If I've missed anything else of note please let me know and I will add it here!
</details>
      
1. Empirical Bayes (EB) ([Introduction and Commentary from Haining Tan](files/Empirical_Bayes.pdf))
   - OPEN QUESTION: what impact (if any) do Empirical Bayes piror specifications have on estimation based on importance sampling?

2. Variational Inference (VI) [(Schwartz STA410 3.0.2)](https://colab.research.google.com/drive/1bFm8kKsFjsVITAScCQeSh2Tn59uk9yGr#cell-opt-VI) <details><summary>To make the link above work...</summary> Remove the (annoyingly) appended "=" at the end of the address and you'll link directly to the intended section</details> <details><summary>[Click] Recording of the 10:00AM-12:15PM conversation on Thursday, May 12.</summary><a href="https://utoronto.zoom.us/rec/share/rERCzi2oDfhvi6UfqWpcpwu_z_whv_vmLbH5A1lilvXC0OPydRbWK56MqTmvsvu7.sAnPk0ZsRVRudHEY">recording</a> (+Cg2&AgvP6)</details>

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
        3. Concerns regarding MC-Dropout [from Ian Osband ([1](https://mobile.twitter.com/ianosband/status/1014466510885216256?lang=en), [2](https://www.reddit.com/r/MachineLearning/comments/8w0v9m/d_ian_osband_dropout_posteriors_give_bad/), [3](https://www.reddit.com/r/MachineLearning/comments/emt4ke/discussion_research_variational_bayesian/), [4](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf)) and [HMG2018](https://arxiv.org/abs/1807.01969)]
        4. [Dropout isn't Bayesian](https://discourse.pymc.io/t/frequency-of-missing-value-imputation/8809) (and MCMC using "Metropolis within Gibbs", Hamiltonian MC, etc.)
    4. [The GP4ML Textbook](http://gaussianprocess.org/gpml/) 
     
        > Hopefully the preceeding materials have been sufficient and this is just a reference at this point.

    <details><summary>At this stage...</summary> 
   We've seen BNN/BBB and MC-Dropout as characterizing uncertainty in the NN context.<br>
   And we've also seen more traditional Bayesian analysis with MCMC using PyMC.<br>
   Can we add something to the Bayesian Deep Learning (BDL) domain?</details>

4. [Normalizing Flows (NF)](files/NormalizingFlows.ipynb), covering:
    1. Change of Variables, Jacobians, Determinants, and Eigenthings
    2. Computation versus Transformation Tradeoff Motivating NF
    3. MADE autoregressive structure, conditional parameter outputs, and the chain rule
    4. Masked/Inverse Autoregressive Flows (MAF/IAF), but not RealNVP or Hamiltonian Flows
   <br><br>
   <details><summary>Are we leaving TF for PyTorch?</summary><img src="files/images/therewillBB.gif"></details>

5. Parallelize 
   
   <details><summary>Our powers combinded</summary><img src="files/images/capn.gif"></details>
      
   - Haining/Eric will create a presentation of the [SWAG](https://www.google.com/search?channel=trow5&client=firefox-b-d&q=swag+bayes+approximation) manuscript including all of it's introductory and contextual material.
      - It seems the SGLD citation [59] may be a key reference (highlighted also in manuscript footnote 3)
      - References [45] and [39] appear of possible interest
      - Fun fact: ["Neal [49]"](https://www.cs.toronto.edu/~radford/homepage.html)
   - Ryan/Yichen will create a presentation on [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.06707) which will include general introducions the following general topics and then explain their specific applications in the manuscript.
      - Optimizing NF and SNF
        <details><summary>Concepts</summary>
        JKL = E_{μZ (z)Pf (z→x)} [− log w(z → x)] = KL (μZ (z)Pf (z → x)||μX (x)Pb(x → z)) + constant<br>
        JML = E_{μX (x)Pb (x→z)} [− log w(x → z)] = KL (μX (x)Pb(x → z)||μZ (z)Pf (z → x)) + constant<br>
        KL (pX (x) ‖ μX (x)) ≤ KL (μZ (z)Pf (z → x) ‖ μX (x)Pb(x → z))<br> 
        </details>
      - [Langevin dynamics](https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics): is "t" in this section is different than "t" in other parts of the of paper?
      - [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing): what is the basis idea and how does this manifest in the SNF architecture?
      - Neural MCMC: you will need to see what this is through the references
        <details><summary>Questions</summary><br>
        Is "t" in the MCMC subsection is different than "t" in other parts of the of paper?<br><br>
        How does (log) path probability in the MCMC subsection (Langevin dynamics subsections) fit into things?<br><br>
        Is the basic idea just to LD + MCMC perturb intermittently between flows to create the noise bypasses topological constraints?<br><br>
        If the prior is the base distribution and SNF fit on data is interpreted as "transforming the prior into the posterior", then repeated stochastic realizations of the SNF are samples with importance weights which are the ratio of the SNF "output distirubtion" relative to the base distribution, so repeated stochastic realizations are importance weighted representations of the postrior.  How are realizations created and stored?  How are uncertainty characterizations presented?<br><br>
        On page 7 the paper says: "Note that neural spline flows perform better than RealNVP without reweighting, but significantly worse with reweighting - presumably because the sharper features representable by splines can be detrimental for reweighting weights."  I think this is saying that the NSF is not sufficiently heavy tailed to be a good importance sampling proposal. What do you think?
        </details>

   - Etc., where "Etc." means identifying and gathering together relevant BDL literature that might be helpful to us
      - [Hamiltonian Flows?](https://arxiv.org/abs/2203.05723)  

# Roadmap

| Week of | Days | Topics | Deliverable | Target | 
|-|-|-|-|-|
|May 9| 4 | IS, EB, VI, BBB | | |
|May 16| 4 | GP, MC-Dropout, NF | | |
|May 23| 4 | SWAG, SNF | Slides Presentation I: IS through NF | May 26 | 
|May 30| 4 | SWAG, SNF | Slides Presentation II: SWAG and SNF | June 2 |
|June 6| 4 | Coding / Writing | Outline Planned Paper | June 9 |
|June 13| 3 | Coding / Writing | Finalize Analysis Examples | June 16 |
|June 20| 3 | Coding / Writing | Draft Manuscript Version I | June 23 |
|June 27| 3 | Coding / Writing | Draft Manuscript Version II | June 30 |
|July 4| 3 | | Final Manuscript Submission | ASAP |
|July 11| 3 | | final poster or slides presentation | TBA |

| Deadline | Event | Special Note | Dates |
|-|-|-|-|
| *Will ask Radu about poster crash | ISBA, Montreal | First time ever hosted in Canada | June 26 - July 1 |
| Paper May 16/19 Poster October 12 | NeurIPS, New Orleans | Bayesian Deep Learning Workshop | Nov 28 - Dec 9 | 
