Ryan: 4:56 start (page 18) 5:55 end (28). Amazing.
You had the right number of slides for about 20 min and covered them very
efficiently and directly when speaking, so you managed to elicit about twice
as much question and comment from the audience as your actual material...
We all clearly appreciated benefiting from your presentation.

So, yes, what should be said is that your presentation was outstanding,
and VASTLY exceeded my expectations and hopes regarding your progress.
You continue to impress with your amazing work ethic and productivity, Ryan.
And, I (and I expect your peers as well), personally thank you for increasing
our awareness and understanding of these advanced topics. I myself learned
many, many new things and developed a substantially increased understanding
of these topics on the basis of your presentation.  I am not paying you
enough, Ryan, it seems... I hope you will nonetheless remain motivated.
And, I hope you are finding your learning to be sufficiently rewarding.

Let's see what comments I can make beyond that...

0. Let's say posteriors "concentrate" with increasing data, rather than
   "collapse".  I think "collapse" is better reserved for the phenomenon
   like that observed in the VAE context (which I knew I had heard before)
   where the KL penalty and reconstruction loss can become pathelogical: 
   https://datascience.stackexchange.com/questions/48962/what-is-posterior-collapse-phenomenon
   
1. The "fl" character styling in "flow" is awesome.
2. Your generalization of the NF in terms of "conditioners" and
   "transformers"/"couplers" makes the varients very approachable.
3. My current feeling is that the primary criticisms of our "NF as
   a likelihood with a prior on its parameters" idea will be the
   intertwined issues that (a) there are a lot of knobs in to
   choose for a NN, and (b) our computations can be demanding.
   My sense is that we may find the answer to these issues to be
   one and the same.
   - Originally, you called this "NN pruning", but now I wonder
     if it might be characterized as well in terms of (1) and (4)
     on your "slide 19". Perhaps I am wrong, but I want to somehow
     "integrate over" the considerations of how to specify a NN
     and arrive at some general purpose specification that is
     both lightweight and generally sufficient for our intended
     application.  E.g., can our concerns regarding NN specifications
     go away if we just abstract them as optimal transport SDEs?
     [Sorry, I know that's not what (1)/(4) on "slide 19" refer
      to, but perhaps it conveys my intuition about how to "avoid"
      the criticism that there are too many design options for NNs].
4. Your insight that MADE parallelizes the chain rule was outstanding.
5. I have seen the graphic on your "slide 21" many times, but never
   found it helpful until it was presented with the "conditioner" and
   "transformer"/"coupler" abstratction.  It was very fun to finally
   "get it" (i.e., visually "see" how the forward/backward bottlenecks
   can form); and, your "slide 22" was perfectly designed to
   immediately reinforce and support this understanding.  This
   section of your presentation was extremely well designed, Ryan.
   - Actually, I would say that following up on all this with your
     "slide 23" made for an absolutley phenomenal MAF/IAF presentation.
     Far and away VASTLY superior to any explanation of these topics
     I myself have ever given.  If your sense is that you are just
     following along with another outline or conceptualization of
     NF methods, my sense is that you are actually doing much, much
     more than that.  Crafting a great and illuminating presentation
     IS NOT as easy as it seems it should be... I think you have a
     very real talent for explanation.  I am extremely impressed, Ryan.
6. Regarding your "slide 24", it was very interesting to see you
   bring up batch norm. I'm not at all surprised by this: batch norm
   is perhaps the primary reason NN became DNN. Still, this is the
   first time I've seen this clearly highlighted in the NF context,
   as my previous sense of recommendations regarding good handling of
   NF methods was that the permutations aspects of NF specifiations
   were what was emphasized as containing the key choies regarding
   representational power and sufficient transformational flexibility.
7. The details of "1x1 convolutions generalize permutations" (on
   your "slide 26") are not clear to me.  Perhaps at some point
   you might make this specification more explicit?
8. Since "Residual Flows" address (1)+(4)/(a)+(d) on your "slide 19"
   they strike me as interesting in the same manner that a SDE NF
   is interesting.  I.e., would this approach allow the specification
   of a NF to be given in terms of its transformational power as
   defined by the expressiveness of the Jacobian of the transformation
   rather than the details of the NN specification? Or do you suspect
   the specification of "Residual Flows" still comes down to many
   choices regarding architecture, tuning, and other parameter
   specifications?
   - Is your comment regarding computation that "Residual Flows"
     remain computationally intractible, relatively speaking?
9. Regarding your "slide 27", my current (perhaps misguided)
   interest in cont. inf. flows boils down to the question:
   - Do these remove burdens of architecture specification?
   - My current focus is that we need to be able to presribe
     a general purpose and robust "NF specification recipe",
     and I'm wondering if any of the options you've laid out
     would more readily provide such a thing for us.
   - Is my intution that thinking in terms of ODE/SDE NNs
     with black box solves somehow absolves us from detailed
     consideration of architecture specifications?
   - I.e., are these ODE/SDE things provide off-the-shelf
     out-of-the-box performance? Or is are still a lot of
     dials here?  Perhaps this a question for David Davenaud.
   - Great point noting the forward/inverse computational
     cost similarity.  This seems extremely interesting
     relative to MAF/IAF/RealNVP.
     - And, actually, I didn't realize RealNVP tended to
       provide computational benefits over MAF/IAF, so great
       job emphasizing that was well in your presentation.
10. Well done prounoucning "Papamakarios". I think I've gotten
    "Duvenaud" down, but I'm still working on "Erdogdu".
11. I really liked your characterization that there is an
    "emerging field coming out of infinitesimal flows" using
    NN as ODEs/SDEs and refocussing computation on black-box
    integration solvers (if I understood that correctly).
    - This feels like how VI sidesteps MCMC by changing the
    type of problem being pursued to provide Bayes inference.

With those comments and questions made, here are the remaining
questions I wrote down during your presentation, and you may
have already answered them all.

- NeuralMCMC is the idea to use a NN to learn a transition kernel; but,
  it does not address uncertainty in the parameters of a NN, correct?
- "Bayesian IS" (as Haining phrased it) is based on IS weighting of
  independent proposals; so, the use of a SWAG (approximate posterior)
  fit as a prior for "Bayesian IS" is an independence sampler based
  alternative to NeuralMCMC that avoids focus on a construction
  kernel and thus avoids (MCMC) concerns regarding autocorrelation, yes?
- Müller 2019 is using IS to produce samples from a target distribution
  approximating a posterior; so, this work along with the stochastic
  NF (SNF) are not direct competitors to our work since they are
  uninterested in characterizing the uncertainty of the NF itself, yes?
- Does the "choice of determent" fully characterize classes of NF
  representational capability, do other aspects of the specification,
  e.g., implementation and algorithmic bestow meaningful delineations
  between different NF methodologies?
- Have I understood you to endorse MAF at this stage?  Or RealNVP?
- Do you suspect (as I do) that our "simple toy statistical problems"
  task will likley not require a large number of parameters?
- Did you comment that "multi-scale architecture" increases computation
  but provides improved representation?  And did this comment somehow
  have something to do with permutations?  I can't quite make sense of
  the notes I took, but all of these concepts seem to appear together
  in what I've written...
- Will we need batch norm?
- Will we need linear flows as permutation matrix building blocks?
- How do invertible 1x1 convolutions learn permutations??
- You mentioned that an entire NN may be made contractive by stacking
  contractive flows, and that this has has "lots of success"; however,
  you also note that this can be overly limiting. Does this mean that
  these "Residual Flows" are perhaps "overly hyped"?
- How reated to "Residual Flows" are "Matrix Determinant Lemma Flows"?
- Do you have more to say about "Flow Pruning" at this stage?
