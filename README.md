# Bi-Tempered Logistic Loss

This is not an officially supported Google product.

Overview of the method is here: [Google AI Blogpost](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)

Also, explore the interactive [visualization](https://google.github.io/bi-tempered-loss/) that demonstrates the practical properties of the Bi-Tempered logistic loss.

Bi-Tempered logistic loss is a generalized softmax cross-entropy loss function with bounded loss value per sample and a heavy-tail softmax probability function.

Bi-tempered loss generalizes (with a bias correction term):

- Zhang & Sabuncu. "Generalized cross entropy loss for training deep neural networks with noisy labels." In NeurIPS 2018.

which is recovered when 0.0 <= t1 <= 1.0 and t2 = 1.0. It also includes:

- Ding & Vishwanathan. "t-Logistic regression." In NeurIPS 2010.

for t1 = 1.0 and t2 >= 1.0.

Bi-tempered loss is equal to the softmax cross entropy loss when t1 = t2 = 1.0. For 0.0 <= t1 < 1.0 and t2 > 1.0, bi-tempered loss provides a more robust alternative to the cross entropy loss for handling label noise and outliers.

## TensorFlow and JAX

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

Tempered version of tf.nn.sigmoid and jax.nn.sigmoid:

```python
def tempered_sigmoid(activations, t, num_iters=5):
  """Tempered sigmoid function.
  Args:
    activations: Activations for the positive class for binary classification.
    t: Temperature > 0.0.
    num_iters: Number of iterations to run the method.
  Returns:
    A probabilities tensor.
  """
```

Tempered version of tf.nn.softmax and jax.nn.softmax:
```python
def tempered_softmax(activations, t, num_iters=5):
  """Tempered softmax function.
  Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature > 0.0.
    num_iters: Number of iterations to run the method.
  Returns:
    A probabilities tensor.
  """
```


## Citation

When referencing Bi-Tempered loss, cite this [paper](https://arxiv.org/pdf/1906.03361.pdf):


```
@inproceedings{amid2019robust,
  title={Robust bi-tempered logistic loss based on bregman divergences},
  author={Amid, Ehsan and Warmuth, Manfred KK and Anil, Rohan and Koren, Tomer},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15013--15022},
  year={2019}
}
```

## Contributions
We are eager to collaborate with you too! Please send us a pull request or open a github issue. Please see this doc for further [details](https://github.com/google/bi-tempered-loss/blob/master/CONTRIBUTING.md)
