# A Robust Model of Gated Graded Working Memory

**Abstract**: We introduce a robust yet simple reservoir model of gated graded
working memory with instantaneous update that is able to store an arbitrary
value at random time over an extended period of time. The dynamic of the model
is not attractor based but exploits reentry and a non-linearity that is learned
during the training phase using only a few representative values. Further study
of the model shows that there is actually an unusual large range of
hyper-parameters for which the results hold (spectral radius, leak, number of
neurons, sparsity). Any large enough population (n > 100), mixing excitatory
and inhibitory neurons can quickly learn and realize such gated graded working
memory function with good precision maintained over an extended period of time.


# Introduction

* define working memory and gated (graded) working memory (in biology)
* remind the main three theories
* review computational models and their potential limitation
* plan of the paper

# Methods

* Model presentation
  (re-use IJCCN figure)
* Taks presentation (single channel, multi-channel)
  (re-use IJCCN figure)
* Training / Testing protocol

# Results

* Perfomance on a smooth signal, single channel
  + error
  + most correlated
  + less correlated
* Perfomance on a smooth signal, three channels
  + error
* Influence of hyper-parameters
  1. number of neurons
  2. spectral radius
  3. sparsity
  4. leak
* Robustness to noise
* Influence of the number of channels
* Influence of the number of learned values
  
# Analysis

* Minimal model (3 or 4 neurons)
* Dynamic, drift, attractor state ?
* Problem of simultaneous ticks
* Explain why the system cannot be robust to internal / output noise

# Discussion

* Rewriting leads to no feedback and no sustained activity
* Any neural population will do (robustness)
* Online / Offline training






# Gated graded memory property of a reservoir

# Introduction

# Methods

## The gating task
## The multiple gating task
## A reservoir model

# Results

## The ability to learn to perform the gating task seems to be a generic property of a big enough group of neurons

- Illustration with smooth signal of how perform the reservoir on the gating task and how evolves the activities
- Illustration showing the influence of the hyperparameters (spectral radius, sparsity of the reservoir and number of neurons)

## This property seems robust against different degradation
- Illustration showing the degradation of performance against noise added in the reservoir
- Illustration showing the degradation of performance against noise added in the output
- Illustration showing the degradation of performance against removal of units (with or without adaptation after removal)
- Illustration showing the degradation of performance against removal of weights (with or without adaptation after removal)
- Illustration showing the degradation of performance against inner weight fluctuation (with or without adaptation after the change)

## There can be more than one items memorised in a single reservoir

- Illustration of how perform the reservoir on the multiple gating task for different number of value to be meomrized.
- Illustration showing the degradation of the quality of the memorisation against the number of items that should be memorized.
- Illustration of potential problems when two memory must be changed simultaneously

## Is the memory directly in the neurons activities ? Not really (Activity maintained or not maintained debate)

- There is variation of activities while maintaining (similar to activity during the delay)
	- Illustration showing that there is activities while the information must be maintained (the mean activity against time)
- Even if it can be read out from the units, the information of the memory is not really directly in any units inside the reservoir
	- Illustration showing how the activity correlates with the memory maintained
	- Can we readout the memory from the activity if we exclude the most correlated units ?
- But the memory is maintained in the output right ? Not really.
	- Explanation of the rewriting and how to remove this output unit while conserving the same activities.
	- Study of the recurrent matrices thus obtained

## Is the memory inside attractors of the dynamics (line attractor/continous attractor) ? Not really

- Ilustration of the drift of the memory for very long runs and various conditions




This suggests that such gated working memory might be a structural property of
any random neural population. quasi independently of weight initialization,
inner and outer connectivity, spectral radius or leak. The model is fed with a
stream of uniform random values and a gate signal. When the gate is open, the
current value is stored and when the gate is closed, the last stored value is
maintained.
