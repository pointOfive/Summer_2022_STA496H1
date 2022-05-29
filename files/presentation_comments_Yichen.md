Part I: 4:46 start (page 12) 4:56 end (page 17). About two minutes per slide. Brilliant.
This was exactly the level of review presentation I was looking for, Yichen.
And you moved over the topics smoothely and comfortably, just as should be done for such talks.

Page 12: I like the graphic
- but maybe z0, z1, ..., zk are enough?
- and there seems to be room to address the Jacobian piece of the transformation as well; i.e.,
  the pdf for z1 and zk could be given in terms of z0 and the Jacobians... might be nice

Page 13: Wonderful contextualization; however, there are very real distinctions to highlight.
q1: How is an NN helpful to make a NF?  How *exactly* is a NN used in the context of a NF?
q2: What's the difference between a generative model and a probabilistic model, if anything?
q3: What is the BIGGEST difference between a NF and a GAN; and, between a NF and a VAE?
q4: Why do all these different methods exist? Are there pros/cons or specialized uses for each?

Page 14: Your description of flows was very good; but I beg to quibble two lesser points.
q5: I have never seen this notation for a Jacobian: what is your reference here?
    The z_k is bolded implying dim(z_k)>1, so I don't think "partial" is the right symbol?
    Same for "partial" z_{k-1}... I've never seen that.  Isn't "partial" a univariate notation?
    Since we're dealing with a multivariate outcome with respect to each multivariate input
    shouldn't we be using Jacobian notation?    
q6: I think "normalzing" refers to the "normal" base distribution:
    do you have a reference for it refering to the "normalizing" of the Jacobian?
    I.e., if it's a "density" then by definition it must be "normalized" (with area 1).

Page 15: Please cite the references you rely upon! It's fine for a rough draft...
- But in general anything outside of a "home group presentation" needs immaculate citations.
- Page 18 is an example of how citations should be included for used figures, etc.

Page 16: Same comment regarding "partial" notation in place of traditional "Jacobian" notation.
- The fact that d is the same in R^d -> R^d is very crucial... the most crucial piece, actually.

Page 17: Great job addressing computation -- this is a key component of our NF conversation
- Wonderful that you concluded on this and then passed off to Ryan at this stage: well planned.

I have a few further questions about specific things you've said, but I can't remember
which slide they were made on or what pieces of your presentation they specifically referenced.
q7: Related to you mentioning "transformation shapes" and "geometric interpretation/intution",
    Does the value of the Jacobian change at different points in the domain?
q8: What visuals or even mathematical statements that could provide helpful intuition relative
    to the answer of the previous question? Or, just more generally regarding inflation and
    contraction behavior of transformations and their Jacobian corrections? 
q9: At some point in your presentation I believe you said "In our case using BBB". I believe
    this was perhaps in reference to a casual comment about characterizing uncertainty in
    NN architectures? Does your statement mean that there is no other way uncertainty in
    NN parameters?  
    
    

Part II: 5:55 start (page 29) 6:35 end (page 38).  Questions and interruptions were completely
undisciplined at this stage, and I think it was mostly me doing so: sorry about that, Yichen!

Page 29: Plot would be sexy but I don't think the white plot on the gray slide looks so good.
Page 30: Perhaps highlight the key phrases you want to focus on with so much text?
- And who is BDL 2019? Wilson?  Your reference listing is improving! But needs a bit more!
- The Blundell quote nicely encapsulates our conversations about BMA/stacking/etc.

Page 31: I really like this slide, but I think it could use it's space a little better?
- Can bullet points be fit onto a single line rather than two lines so it looks cleaner?
- Your total variation equation is great, but you could label aleatoric and epistemic
  with \underbrace{...}_{aleatoric} + \underbrace{...}_{epistemic}
 
q10: You've said "aleatoric variability is encoded in the likelihood and it doesn't decreased
     with more data"; however, we don't always know the likelihood. How should I interpret
     this under competing models? E.g., what if I add more features? Or what if I change the 
     flexibility of the model by changing between, say, linear and quadratic specifications?
     Is aleatoric variability different depending upon the model specification in use?

- I would center the total variation equation (with clear labels) for this slide; then, I
  would emphasize your (exctremely excellent) characterization of the likelihood (aleatoric)
  versus the posterior (epistemic). And then I would make sure the bullet points appeared
  in terms of relevance and were said as simply as possible (on only a single line) so the
  layout looks extremely simple and approachable -- right now it's hard to want to read it.
- Yay a reference!

Page 32: I like that you use two column slides, as I think that's space effective; however...
- The areas labeled for their aleatoric uncertainty also have "low epistemic uncertainty"
- This figure could probably be remade pretty easily as a PyMC3/GP to improve formatting.
  The page seems very "cut and paste" rough draft. Not a problem for our purposes, so far...

q11: How is the degree of uncertainty within extrapolation regions determined?  There's no
     data at all, so how would one know what to do here?  The truth is that such a thing
     needs to be so-called "calibrated", which means that the eventual "uncertainty" in
     extrapolation regions will actually be controlled by, e.g., the choice fo the kernel
     function in the context of a GP. So when "uncertainty" is controlled by "tuning
     parameters", we say uncertainty needs to be "calibrated".

Page 33: I think this slide is perhaps more introductory and should be earlier in the deck.
- **It might be good in the context of describing an NF NN?**
- Contradicts "Modeled by placing a prior distribution over the weights of NNs" on slide 31.
- Good job with references here.  Is "Bayesian training approach" from that reference?
  I think we decided we prefer the phrasing "Bayesian perspective or interpretations"?

Page 34: I like the concept and content of this slide.  Very good contextualization.

- BMA appears under "ensemble learning" on wikipedia, which is telling:
  https://en.wikipedia.org/wiki/Ensemble_learning
- As I mentioned, the math expression you've used here doesn't actually acount for BMA.
  I haven't been at all specific about this, but to correct that now please see section
  10.8 here for the necessary extended notational specifications ignored so far:
  https://jrnold.github.io/bayesian_notes/model-comparison.html#bayesian-model-averaging
- Nonetheless, what you have is the "posterior predictive distribution"
  https://en.wikipedia.org/wiki/Posterior_predictive_distribution
  and this is an key topic to emphasize as well, especially relative to BMA

- Wilson's point is that posteriors concentrates with increasing data in Bayesian analysis
  so methods that don't do that can only be viewed as being perhaps Bayesian inspired.
- Posterior "concentration" is the right nomenclature as compared to posterior "collapse":
  https://datascience.stackexchange.com/questions/48962/what-is-posterior-collapse-phenomenon
- I don't like saying that "the Bayesian model average converges to the MLE solution"...
  this seems to wrongly suggest that the MLE is the objective of a Bayesian analysis...
  What I'd say is "the uncertainty in the BMA goes to 0 as data nize n goes to infinity"

- I greatly like that your slides are centered around specific mathematical expressions,
  but I think the expressions need to be more prominant and the text should be supportive
  rather than as central as it is. Make text minimal with the fewest lines possible!

Page 35: I didn't find this slide useful during the presentation as it's too busy!
- Looking at it now I still don't find it so helpful; and, it is so BBB-centric, no?
- Also, VI and SGD etc are not analog terminologies... SGD optimizes VI...
- I'm not a fan of this slide, but perhaps you can still convince me of it's use yet?

Page 36: I not a big fan of this slide, either; but, it is a little bit more informative
- **I wonder if we should look into KFAC?**
- **Deep Ensembles** here indeed suggests it has a Bayesian approximation interpretation.**
- I think your highlighting exercise was good, but otherwise I don't see the value here.

Page 37: I find this slide extremely informative
- **I wonder if we should look into Laplace approximation?**
- **Why does this table leave out SWAG?**

Page 38:
- One of the talks I'm most looking forward to at ISBA/Montreal is about priors for NNs.
- I don't think we will be too concerned about the usual "what is your prior?" critique
  since our prior will just be a SWAG posterior which already prescribes a choice
  of prior; however, we will certainly have to address
  1. the "uncertainty only for a single posterior mode" critique, and 
  2. the "heavier computational cost" critique, and, you've not mentioend this, but
  3. the "NNs have too many tuning parameters" is a critique that is absolutely going to
    come and one which will require a "standard recipe" or "prescription" on our end
    for guiding practitioners as to how to quickly whip up a usable NF suiting our method.
- I think the critique that regularization strategies don't often benefit much from a
  Bayesian perspective is fair.  I think most DNN regularization is inspired by
  numeric and algorithmic considerations as opposed to Bayesian considerations; so, I
  think I agree with Wilson that the argument is not that Bayes provides improved
  regularizers; but, that the BMA framework should theoretically be the be choice around.
- That said, here's a NIPS 2019 "Tutorial" that argues that Bayesian characterizations
  of optimization procedures (GD, SGD, RMSprop, Adam, etc.) could lead to improved
  optimization strategies: https://slideslive.com/38923183/deep-learning-with-bayesian-principles

Yichen -- your contribution to our group is wonderful. Your lines of inquiry are helping to
contextualize and justify our work, and will help us present a strongly defensible argument
for the methodologies we propose. PLEASE MAKE SURE YOU ARE KEEPING TRACK OF ALL THE
REFERENCES YOU ARE PUTTING TOGETHER IN YOUR EXPLORATION. They will make for a very rich
and expansive bibliography for our manuscript!
