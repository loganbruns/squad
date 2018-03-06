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

## v3_1 Stack LSTM cells
Stacked with four cells.

```
INFO:root:Epoch 22, Iter 18500, dev loss: 4.877050
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.21 seconds
INFO:root:Epoch 22, Iter 18500, Train F1 score: 0.703295, Train EM score: 0.604000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 144.13 seconds
INFO:root:Epoch 22, Iter 18500, Dev F1 score: 0.426893, Dev EM score: 0.317775
```

Sanity {"f1": 41.11472393130989, "exact_match": 34.44444444444444}
Dev    {"f1": 46.13626539394965, "exact_match": 37.13339640491959}

## v3_2 Stack LSTM cells
Stacked with two cells and add layer norm.

Stopped due to TOO slow

```
INFO:root:Epoch 5, Iter 4000, dev loss: 5.002152
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 25.32 seconds
INFO:root:Epoch 5, Iter 4000, Train F1 score: 0.425096, Train EM score: 0.321000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 230.70 seconds
INFO:root:Epoch 5, Iter 4000, Dev F1 score: 0.352479, Dev EM score: 0.254643
```

## v3_3 Stack LSTM cells
Stacked with two cells and LSTMBlockCell.

Stopped due to bad dev loss curve

```
INFO:root:Epoch 8, Iter 6500, dev loss: 4.821969
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 9.08 seconds
INFO:root:Epoch 8, Iter 6500, Train F1 score: 0.634083, Train EM score: 0.527000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 54.79 seconds
INFO:root:Epoch 8, Iter 6500, Dev F1 score: 0.414401, Dev EM score: 0.305168
```

## v3_4 Stack LSTM cells
Stacked with three layers

```
INFO:root:Epoch 3, Iter 18500, dev loss: 4.647619
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 15.64 seconds
INFO:root:Epoch 3, Iter 18500, Train F1 score: 0.813153, Train EM score: 0.724000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 124.08 seconds
INFO:root:Epoch 3, Iter 18500, Dev F1 score: 0.450833, Dev EM score: 0.337504
```

Sanity {"f1": 42.26598375022376, "exact_match": 36.41975308641975}
Dev    {"f1": 48.77692235146288, "exact_match": 39.593188268684955}

## v4 Multiheaded Attention
Multiheaded Attention.

Note: below is on branch relative to v1
```
INFO:root:Epoch 22, Iter 18500, dev loss: 4.657432
INFO:root:Calculating F1/EM for 1000 examples in train set...
Refilling batches...
Refilling batches took 3.59 seconds
INFO:root:Calculating F1/EM for 1000 examples in train set took 8.06 seconds
INFO:root:Epoch 22, Iter 18500, Train F1 score: 0.736833, Train EM score: 0.636000
INFO:root:Calculating F1/EM for all examples in dev set...
Refilling batches...
Refilling batches took 2.22 seconds
Refilling batches...
Refilling batches took 0.00 seconds
INFO:root:Calculating F1/EM for 10391 examples in dev set took 48.10 seconds
INFO:root:Epoch 22, Iter 18500, Dev F1 score: 0.425627, Dev EM score: 0.315658
```

Sanity {"f1": 44.43563356665186, "exact_match": 38.641975308641975}
Dev    {"f1": 46.22796345168189, "exact_match": 37.13339640491959}

## v4_1 Multiheaded Attention
Multiheaded Attention updated to fix weighted sum to be full size values.

Note: below is on branch relative to v1

```
INFO:root:Epoch 20, Iter 16500, dev loss: 4.769416
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 9.67 seconds
INFO:root:Epoch 20, Iter 16500, Train F1 score: 0.702132, Train EM score: 0.606000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 60.55 seconds
INFO:root:Epoch 20, Iter 16500, Dev F1 score: 0.412239, Dev EM score: 0.302666
```

Sanity {"f1": 40.591542777041866, "exact_match": 33.7037037037037}
Dev    {"f1": 44.708251993879955, "exact_match": 35.496688741721854}

## v4_2 Multiheaded Attention
Multiheaded Attention updated to add a single layer fully connected
instead of average of heads

Note: below is on branch relative to v1

```
INFO:root:Epoch 18, Iter 15500, dev loss: 4.713960
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 11.14 seconds
INFO:root:Epoch 18, Iter 15500, Train F1 score: 0.685318, Train EM score: 0.576000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 74.70 seconds
INFO:root:Epoch 18, Iter 15500, Dev F1 score: 0.428223, Dev EM score: 0.315465
```

Sanity {"f1": 42.69393224007684, "exact_match": 37.03703703703704}
Dev    {"f1": 46.39409992966636, "exact_match": 37.21854304635762}

## v4_3 Multiheaded Attention
Multiheaded Attention plus basic attention between embeddings in
addition to hidden states

Note: below is on branch relative to v1

```
INFO:root:Epoch 17, Iter 14000, dev loss: 4.706721
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 10.08 seconds
INFO:root:Epoch 17, Iter 14000, Train F1 score: 0.753629, Train EM score: 0.664000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 63.55 seconds
INFO:root:Epoch 17, Iter 14000, Dev F1 score: 0.420370, Dev EM score: 0.311135
```

Sanity {"f1": 41.73292612623567, "exact_match": 36.17283950617284}
Dev    {"f1": 45.75936458729861, "exact_match": 36.565752128666034}

## v4_4 BiDAF attention
Bi-directional multiheaded attention.

Note: below is on branch relative to v1

## Backlog

# similarity head from bidaf for multiheaded

# bidaf

# more stacking for rnn cells

# feed forward before softmax

# residuals via short circuits and layernorm

# multiheaded attention

# revisit attention bilinear form with regularization
