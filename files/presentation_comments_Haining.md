

# Haining: Bayesian Methods

1. 4:09-4:46 (47 minutes) for 10 slides with interactive conversations

   Not a criticism: just feedback about how much time it can take to present material
   
2. Page 2: the content of the slide is good but, but I'm not sure I like the bullet point formatting.

   I find "the Bayesian version of Importance Sampling" to be a good turn of phrase.
   Good job creating this useful addition to the nomenclature for us
   
   I think saying "IS is for variance reduction" is not holistic enough; instead, let's say
   "The flexibility to choose an arbitrary (heavy tailed) proposal distribution is an obvioius attraction of IS; however,
    to realize the variance reduction potential of the choices of proposal distributions may nonetheless be quite restrictive
    in that they must only produce large IS weights when the evaluation of the function f(x) of the expectation is close to 0."
      
   Your explanation during the presentation about how the IS variance reduction is achieved was very good.

   I think the outline of the above sentence and the two mathematical specifications on the slide tell the complete story.
   I also think the information in your text is good; nonetheless, this slide still feels like it's at a rough draft stage to me.
   Can the bullet points be removed, the main message presented more efficently, and then special side comments perhaps included
   as informational note boxes down below? 

3. Page 3: This slide is not currently helpful enough.

   How about making this two slides? The first slide is the visual of the NF that you have, and then the second slide could show
   SWAG (VI) approximation of posterior, which is then carried forward as the prior used for importance sampling.
   
   A visual/discussion of the 80/20 SWAG training/IS fine tuning could also be helpful here; and, I'd like us to start considering
   "core sets" (https://arxiv.org/abs/2011.09384) as a way to reduce the sample size burden for IS likelihood weight evaluation.
   (And, I really like that you're already thinking about this with your 80/20 notion -- it's a very sensible proposal).

   Let's use math symbols insted of writing out "theta"; and, a visual of the affine transformation (e.g., MAF) in terms of NNs
   might be helpful to clarify what "theta" actually looks like. This might benefit from some of Ryan's visuals; and, if you've
   got something like this, you could discuss what it means to "put a prior" on the parameters of a NN, or what the SWAG approximation
   of the posterior of the parameters of the NN actually looks like. Such visuals/discussions would go a long way in answer the following.

   **What is a NN used for in an NF? Why do we need a NN? What does it do?**

   I think the following additional questions also present opportunities for improved clarity and explanation.

   - **How are IS weights and samples stored and used?**
   - **Are we training NF? Are we training a NN? What is the loss function?** (-- Ryan)

   Regarding the loss function of a NF, there was some suggestion that this included both reconstruction AND KL loss.
   I may be wrong, but I don't think this is true.  I think JKL (forward) and the JMK (backward) discussed in the SNF
   paper might be helpful for you, and to start with I would refer you to my second point in my email to everyone
   titled "Thursday 4pm follow up", where I give my initial thoughts on these two NF optimization objectives.
   Also, note that the SNF objectives are a little more complex because they include transition distributions as well
   (for MH) which are not generally present in vanilla NF contexts (but of course are present as the S in SNF).
   
   Note: "Normalizing" in NF just means the base distirbution is Gaussian; unless Yichen can show us a citation for his claim that
   the "normalizing" term comes from the Jacobian adjustment term which maintains the (area 1) "normalization" of the transform.

4. Pages 4-9: I think all of these can be collapsed down to a single page; however, I know that your intention was to
   spend time introducing the details of these various methods, so this is not a criticism but just a matter of the choice
   of focus moving forward.  We want to introduce these methods in a way that gives an air of complete confidence and
   understanding, as if we are completely comfortable with them (including their strengths and limitations) and just
   assume they're available however needed relative to our own work. For me, the highlights are

   - Markov Chain Monte Carlo (MCMC) constructs Markov dependent samples whose stationary distributions are posteriors of interest
     and can thus be used for Monte Carlo estimation of posterior expectations.
   - High autocorrelation can limit MCMC applicability, especially in high dimensional domains where perturbation or Gibbs sampling
     based transition kernels do not efficiently traverse the sampling space.
   - Metropolis Hastings (MH) propose-accept-reject MCMC increasingly incorporates physical dynamics proposal systems, e.g., as in
     Hamitonian Monte Carlo (HMC) and Stochastic Gradient Langevin Dynamics (SGLD).
   - HMC is notable for its inclusion of auxilliary parameters which embed a target distribution into an energy system which can be
     evolved to produce efficient MH proposals.
   - SGLD is notable in that it fulfills the hybrid roles of simultaneously optimizing an objective function, while at the same time
     producing MH-based MCMC sampling.
   - An alternative to MCMC sampling of target distributions is to approximate the target distribution through a variational inference
     specification, thus reformulating the task in the form of an optimization objective.

   While that high level is what I think our ostensible operations (as regards MCMC, etc.) should be presented as, of course
   there are many implementation details to work out behind the scenes.  More in that vein, I have the following questions:

   - **Does the SGLD step factor epsilon matching the brownian pertibation epsilon really make MH acceptance probability 1?**
   - **If this is so, is it because of the transition probabilities relative to the gradient directions of the transitions?**
   - **In your own words, what would you say is the purpose of the auxiliary energy parameters in HMC?**
   - **Do you agree with the second point above: "High autocorrelation can limit... Gibbs sampling...do not efficiently traverse"?**

   By the way, these questions are a result of how good your general overview of SGLD was; and, how good your answers to the questions
   asked during your presentation were.  I did not know what SGLD was, but based on what you said I know "feel" I do. Actually,
   I now even more think of SGD as an approximation of SGLD; or, SGLD with epsilon=0 is SGD, with t indexing small SGD-batches.
   **I think this connection should be made at some point, no?**

   One other thing I noticed was that your SGLD specification was given in terms of the posterior, and I guess this is an obvious
   and natural choice; however, I thought it was a very good and provacative choice, nonetheless.  It reminds me that we will want
   to be similarly intentional (and consistent) with our other notational representation choices.
   
5. Page 9: comments on VI

   I think you said, "expectation of log likelihood" but perhaps more clearly you could have added "under variational distribution".
   Your mean-field explanation was good.  Perhaps you could also add the other common choice of full-rank, and give examples of both?

6. Page 10: comments on VAE

   Your summary and explanation of VAE was quite good. You said that what the VAE does is "model uncertainty of laten variable",
   but this is different than modeling the uncertainty of the encoder and decoder themselves.

7. Page 11: comments on BBB

   I liked your expectation/gradient interchange notation. At first I thought it should go inside the brakets, but changed my mind.
   I did not follow or understand the visual about backbrop, but reviewing the slides now it's excellent.

I am extremely pleased with the consolidated understanding demonstrated through your talk, Haining.  Your ever increasing foundation
of diverse and advanced statistical concepts is a pleasure to witness. I look forward to seeing you (implicitly) improve "slide 3"
through the presentation you'll provide us on SWAG. And, for the moment, I encourage you to put away any stress or concern you
have about computation resulting from large parameter spaces.  We'll start with small toy problems as our examples, and we just
need proof of concept at this stage.  One we've got that working then we can then have a clear discussion about our computational
limitations and what we can overcome and what seems intractable.


