# Bi-Tempered Logistic Loss

This is not an officially supported Google product.

Overview of the method is here: [Google AI Blogpost](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)

Also, explore the interactive [visualization](https://google.github.io/bi-tempered-loss/) that demonstrates the practical properties of the Bi-Tempered logistic loss.


## TensorFlow

A replacement for standard logistic loss function: ```tf.losses.softmax_cross_entropy``` is available [here](https://github.com/google/bi-tempered-loss/blob/master/tensorflow/loss.py#L161)


```python
def bi_tempered_logistic_loss(activations,
                              labels,
                              t1,
                              t2,
                              label_smoothing=0.0,
                              num_iters=5):
  """Bi-Tempered Logistic Loss with custom gradient.
  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
  Returns:
    A loss tensor.
  """
```

Replacements are also available for the transfer functions:

Tempered version of tf.nn.sigmoid:

```python
def tempered_sigmoid(activations, t, num_iters=5):
  """Tempered sigmoid function.
  Args:
    activations: Activations for the positive class for binary classification.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
  Returns:
    A probabilities tensor.
  """
```

Tempered version of tf.nn.softmax:
```python
def tempered_softmax(activations, t, num_iters=5):
  """Tempered softmax function.
  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
  Returns:
    A probabilities tensor.
  """
```


## Citation

When referencing Bi-Tempered loss, cite this [paper](https://arxiv.org/pdf/1906.03361.pdf):


```
@article{bitemperedloss,
  author    = {Ehsan Amid and
               Manfred K. Warmuth and
               Rohan Anil and
               Tomer Koren},
  title     = {Robust Bi-Tempered Logistic Loss Based on Bregman Divergences},
  journal   = {CoRR},
  volume    = {abs/1906.03361},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.03361},
}
```

## Contributions
We are eager to collaborate with you too! Please send us a pull request or open a github issue. Please see this doc for further [details](https://github.com/google/bi-tempered-loss/blob/master/CONTRIBUTING.md)
