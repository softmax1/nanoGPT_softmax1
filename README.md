# nanoGPT @ softmax1

🚧🚧🚧
> Notice: This page is under construction ... some of the links below wont work and the text may change before I begin the experiment!

🚧🚧🚧

The goal is to run an experiment to see if there is evidence [softmax1](https://www.evanmiller.org/attention-is-off-by-one.html), also known as "Quiet Attention" can perform better than regular old transformers.

This is a humble experiment, on a fairly small model, so I am not expecting any big conclusions. It would be suprising if I found a solid result to be honest. However this experiemnt (and people's feedback) may convince me or someone else to run bigger experiments.

# The experiment

It is good to define what the experiment is before I do it and how I will evaluate the result. This will avoid temptations to change the experiment as I go to "fix" things.

## Model Choice

I will use a model that trains fairly quick and cheaply. This is the nanoGPT model. I will use this specific commit so you can reproduce this at home: [eba36e84649f3c6d840a93092cb779a260544d08](https://github.com/karpathy/nanoGPT/tree/eba36e84649f3c6d840a93092cb779a260544d08)

## Data Preparation

I will run the data preparation here: [prepare.py](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/data/shakespeare/prepare.py), but instead of Shakespeare I will use text from Reddit. The reason is I want to still have a look at what is produced and judge it by eye and that will be easier in modern everyday English.

This preparation script uses `tiktoken` encoding from OpenAI, with the `gpt2` setting.

The script to obtain the text and the text itself can be found in the [obtaintext.py](./stc/patch/obtaintext.py) folder of this repo.

You can download the text [from here](https://q1r1.c19.e2-5.dev/models/reddit_sydney_text_sample.tgz) (sah256 b5d4fe2899431bed3b9f16c1a4773b5088094c5e7c03e77431affffd869e2d3c), at the moment anyway!

## Model Parameters

I will use the [train_shakespeare_char.py](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/config/train_shakespeare_char.py) as the basis for the parameters, which defines a baby GPT model. I will double the size of the `n_embd` to 768 since we are now using bigger tokens and not just characters.

I will increase `max_iters` to 100000 to get more training done on both models. And increase `lr_decay_iters` to be consistent too.

In summary

| Parameter                   | Value    | Meaning                                                               |
| --------------------------- | -------- | --------------------------------------------------------------------- |
| gradient_accumulation_steps | 1        | Number of  forward passes per iteration to perform                    |
| batch_size                  | 64       | Number of examples to use to simulataneously train the network        |
| block_size                  | 256      | Number of tokens entering the model                                   |
| n_layer                     | 6        | Number of layers of transformer blocks                                |
| n_head                      | 6        | Number of attention heads                                             |
| n_embd                      | 768      | Size of trained vector representing each token                        |
| dropout                     | 0.2      | Proportion of randomly selected parameters to not update on each step |
| learning_rate               | 1e-3     | Control how big the updates are                                       |
| max_iters                   | 1000000  | Number of iterations                                                  |
| decay_lr                    | True     | whether to decay learning rate                                        |
| warmup_iters                | 100      | During warm up iterations it ramps up from 0 to min_lr linearly       |
| lr_decay_iters              | 1000000  | After this number of iters, stop cosine decay and  stick to min_lr    |
| min_lr                      | 1e-4     | Minimum learning rate                                                 |
| beta1                       | 0.9      | beta1 for AdamW                                                       |
| beta2                       | 0.99     | beta2 for AdamW                                                       |
| bias                        | False    | do we use bias inside LayerNorm and Linear layers?                    |
| grad_clip                   | 1.0      | clip gradients at this value                                          |
| backend                     | nccl     | torch backend type; nccl is best for gpu                              |
| device                      | cuda     | device to run calculations on                                         |
| dtype                       | bfloat16 | datatype to use in tensors                                            |
| compile                     | true     | torch performanc optimization                                         |

## Code changes to implement softmax1

I have reviewed he following code, taken from https://github.com/softmax1/EsperBERTo/blob/7d2d5ed8695b95ade6bcbe21b7ce981b3c9394d7/src/functional.py#L7, and I will probably use this (subject to license being published).

```python
def softmax_n_shifted_zeros(input: Tensor, n: int) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=-1, keepdim=True).values
    # shift the input to prevent overflow (and underflow in the denominator)
    shifted_inputs = subtract(input, input_maxes)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = exp(shifted_inputs)
    original_denominator = numerator.sum(dim=-1, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_zeros = multiply(input_maxes, -1)
    # and then add this contribution to the denominator
    denominator = add(original_denominator, multiply(exp(shifted_zeros), n))
    return divide(numerator, denominator)
```

Todo: It might be worth figuring out the most efficient backprop for this? 

Todo: Describe the changes to the model.py

## Execution Environment

I will be using [modal.com](https://modal.com) to train and evaluate the model. 

The script [experiment.py](/experiment.py) uses modal to perform the init and training steps on a cloud gpu.

Usage:

```bash
modal token new
modal run experiment.py:init
modal run experiment.py:train
```

## Evaluation

I won't use perplexity to compare the models as we don't expect it to change. However it will be recorded and might be interesting.

I will use the method here to measure kurtosis: https://github.com/softmax1/softmax1

```python
for name, param in model.named_parameters():
    mean = param.data.mean()
    diffs = param.data - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
```

I will then, time permitting, see what compressability looks like, i.e. what if we quantize the model?