# cs224n-win18-squad
Code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.

# History

## v0.0 baseline
Original code

```
Epoch 275, Iter 236500, dev loss: 5.271603
Calculating F1/EM for 1000 examples in train set...
Calculating F1/EM for 1000 examples in train set took 7.54 seconds
Epoch 275, Iter 236500, Train F1 score: 0.807397, Train EM score: 0.718000
Calculating F1/EM for all examples in dev set...
Calculating F1/EM for 10391 examples in dev set took 45.10 seconds
Epoch 275, Iter 236500, Dev F1 score: 0.402082, Dev EM score: 0.293908
```

Sanity {"f1": 38.252188868583765, "exact_match": 32.22222222222222}
Dev    {"f1": 43.64367121372662, "exact_match": 34.418164616840116}
#3 on dev leaderboard

## v1.0 GRU to LSTM
Switched RNN cell from GRU cell to LSTM cell

```
INFO:root:Epoch 20, Iter 16500, dev loss: 4.628717
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 7.34 seconds
INFO:root:Epoch 20, Iter 16500, Train F1 score: 0.655934, Train EM score: 0.555000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 45.06 seconds
INFO:root:Epoch 20, Iter 16500, Dev F1 score: 0.423220, Dev EM score: 0.312097
```
Sanity {"f1": 43.07304663783283, "exact_match": 36.17283950617284}
Dev    {"f1": 45.601230787738146, "exact_match": 36.53736991485336
#8 on dev leaderboard

## v2.0 Switch basic attention to bilinear form
Switch basic attention to bilinear form

```
INFO:root:Epoch 23, Iter 19000, dev loss: 4.779728
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 7.48 seconds
INFO:root:Epoch 23, Iter 19000, Train F1 score: 0.689376, Train EM score: 0.585000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 45.79 seconds
INFO:root:Epoch 23, Iter 19000, Dev F1 score: 0.412955, Dev EM score: 0.302377
```

Sanity {"f1": 41.83363823183292, "exact_match": 33.58024691358025}
Dev    {"f1": 44.586709889036435, "exact_match": 35.08987701040681}

## v2.1 Switch basic attention to bilinear form
Switch basic attention to bilinear form with identity initialization
instead of xavier.

Note: may be worth retrying with regularization

```
INFO:root:Epoch 23, Iter 19000, dev loss: 4.625487
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 7.29 seconds
INFO:root:Epoch 23, Iter 19000, Train F1 score: 0.682001, Train EM score: 0.568000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 46.32 seconds
INFO:root:Epoch 23, Iter 19000, Dev F1 score: 0.415693, Dev EM score: 0.306515
```

Sanity {"f1": 41.89903350556816, "exact_match": 35.18518518518518}
Dev    {"f1": 45.30173689133834, "exact_match": 36.338694418164614}

## v3 Stack LSTM cells
Initially tested with two stacked cells.

```
INFO:root:Epoch 22, Iter 18500, dev loss: 4.735123
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 11.04 seconds
INFO:root:Epoch 22, Iter 18500, Train F1 score: 0.751977, Train EM score: 0.657000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 75.79 seconds
INFO:root:Epoch 22, Iter 18500, Dev F1 score: 0.444390, Dev EM score: 0.333077
```

Sanity {"f1": 44.43220007758217, "exact_match": 38.148148148148145}
Dev    {"f1": 48.24849816947582, "exact_match": 39.195837275307476}
#16 on dev leaderboard

## Backlog

# more stacking for rnn cells

# feed forward before softmax

# residuals via short circuits and layernorm

# multiheaded attention

# revisit attention bilinear form with regularization
