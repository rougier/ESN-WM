# A Robust Model of Gated Working memory

**Abstract**: We introduce a robust yet simple reservoir model of gated working
memory with instantaneous update that is able to store an arbitrary value at
random time over an extended period of time. The dynamic of the model is not
attractor based but exploits reentry and a non-linearity that is learned during
the training phase using only a few values. Further study of the model shows
that there is actually an unusual large range of hyper-parameters for which the
results hold. Any large enough population (n > 100), mixing excitatory and
inhibitory neurons can quickly learn and realize such gated working memory
function with good precision.


# Introduction
# Model
# Results
# Analysis
# Discussion


This suggests that such gated working memory might be a structural property of
any random neural population. quasi independently of weight initialization,
inner and outer connectivity, spectral radius or leak. The model is fed with a
stream of uniform random values and a gate signal. When the gate is open, the
current value is stored and when the gate is closed, the last stored value is
maintained.
