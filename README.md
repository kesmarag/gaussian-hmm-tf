<h1 id="kesmarag.ghmm.GaussianHMM">GaussianHMM</h1>

```python
GaussianHMM(self, num_states, data_dim)
```
A Hidden Markov Models class with Gaussians emission distributions.

<h2 id="kesmarag.ghmm.GaussianHMM.fit">fit</h2>

```python
GaussianHMM.fit(self, data, max_steps=100, batch_size=None, TOL=0.01, min_var=0.1, num_runs=1)
```
Implements the Baum-Welch algorithm.

    Args:
      data: A numpy array with rank two or three.
      max_steps: The maximum number of steps.
      batch_size: None or the number of batch size.
      TOL: The tolerance for stoping training process.

    Returns:
      True if converged, False otherwise.


<h2 id="kesmarag.ghmm.GaussianHMM.posterior">posterior</h2>

```python
GaussianHMM.posterior(self, data)
```
Runs the forward-backward algorithm in order to calculate
       the log-scale posterior probabilities.

    Args:
      data: A numpy array with rank two or three.

    Returns:
      A numpy array that contains the log-scale posterior probabilities of
      each time serie in data.


<h2 id="kesmarag.ghmm.GaussianHMM.run_viterbi">run_viterbi</h2>

```python
GaussianHMM.run_viterbi(self, data)
```
Implements the viterbi algorithm. 
    (I am not sure that it works properly)

    Args:
      data: A numpy array with rank two or three.

    Returns:
      The most probable state path.


<h2 id="kesmarag.ghmm.GaussianHMM.generate">generate</h2>

```python
GaussianHMM.generate(self, num_samples)
```
Generate simulated data from the model.

    Args:
      num_samples: The number of samples of the generated data.

    Returns:
      The generated data.

## instalation using pip:

pip install https://github.com/kesmarag/gaussian-hmm-tf/tree/master/dist/kesmarag-gaussian-hmm-tf-0.1.1.tar.gz

## usage
```python
import numpy as np
from kesmarag.ghmm import GaussianHMM

# create a random data set with 3 time series.
data = np.random.randn(3, 100, 2)

# create a model with 10 hidden states.
model = GaussianHMM(10, 2)

# fit the model
model.fit(data)

# print the trained model
print(model)

# Good luck
```