## _Generative Adversarial Nets_ and _Variational Auto-Encoders_



## What are GANs?

Generative Adversarial Networks (GAN) is a deep learning model and one of the most promising methods for unsupervised learning on complex distributions in recent years. The model produces a fairly good output through (at least) two modules in the framework: Generative Model and Discriminative Model.

Take a simple example,

Generative Model: produce new fake currency

Discriminative Model: Given a currency, detect whether it is fake or not.

Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine
articles.

### Work Principle

The adversarial modeling framework is most straightforward to apply when the models are both
multilayer perceptrons. To learn the generator’s distribution pg over data x, we define a prior on
input noise variables pz(z), then represent a mapping to data space as G(z; θg), where G is a
differentiable function represented by a multilayer perceptron with parameters θg. We also define a
second multilayer perceptron D(x; θd) that outputs a single scalar. D(x) represents the probability
that x came from the data rather than pg. We train D to maximize the probability of assigning the
correct label to both training examples and samples from G. We simultaneously train G to minimize
log(1 − D(G(z))):

![image](formula1.png)

## Intro to VAEs

Variational Encoder VAE (Variational Auto-encoder), like GAN, has become the most popular method for unsupervised learning of complex probability distributions.
Generally, 

###



**References**

1.Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron
Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2014.

2.Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014

