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

```
INFO:root:Epoch 10, Iter 8500, dev loss: 4.266326
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 7.65 seconds
INFO:root:Epoch 10, Iter 8500, Train F1 score: 0.643647, Train EM score: 0.523000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 48.73 seconds
INFO:root:Epoch 10, Iter 8500, Dev F1 score: 0.464632, Dev EM score: 0.347320
```

Sanity {"f1": 45.857403383413896, "exact_match": 39.25925925925926}
Dev    {"f1": 50.24618423841851, "exact_match": 40.74739829706717}

## v5 BiDAF attention with two layers
Bi-directional multiheaded attention with two layers

```
INFO:root:Epoch 12, Iter 10000, dev loss: 4.248739
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 10.70 seconds
INFO:root:Epoch 12, Iter 10000, Train F1 score: 0.709331, Train EM score: 0.595000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 80.51 seconds
INFO:root:Epoch 12, Iter 10000, Dev F1 score: 0.485221, Dev EM score: 0.366086
```

Sanity {"f1": 45.74603664669224, "exact_match": 38.641975308641975}
Dev    {"f1": 52.43855839392188, "exact_match": 42.904446546830656}
#34 on dev leaderboard

## v5_1 BiDAF attention with two layers and two more fully connected output layers
BiDAF attention with two layers and two more fully connected output layers

```
INFO:root:Epoch 12, Iter 10000, dev loss: 4.334206
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 11.10 seconds
INFO:root:Epoch 12, Iter 10000, Train F1 score: 0.752329, Train EM score: 0.656000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 86.02 seconds
INFO:root:Epoch 12, Iter 10000, Dev F1 score: 0.484270, Dev EM score: 0.365701
```

Sanity {"f1": 47.61124700453312, "exact_match": 40.864197530864196}
Dev    {"f1": 52.53578071744936, "exact_match": 43.0558183538316}

## v5_2 Retrain embeddings

Crazy overfitting

```
INFO:root:Epoch 6, Iter 4500, dev loss: 4.646455
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 10.93 seconds
INFO:root:Epoch 6, Iter 4500, Train F1 score: 0.774622, Train EM score: 0.668000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 82.49 seconds
INFO:root:Epoch 6, Iter 4500, Dev F1 score: 0.448740, Dev EM score: 0.336253
```

Sanity {"f1": 42.4202848180741, "exact_match": 34.44444444444444}
Dev    {"f1": 48.58739310895522, "exact_match": 39.25260170293283}

## v5_3 hidden state of 300 dimensions

```
INFO:root:Epoch 10, Iter 8000, dev loss: 4.491363
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 11.83 seconds
INFO:root:Epoch 10, Iter 8000, Train F1 score: 0.830607, Train EM score: 0.746000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 91.22 seconds
INFO:root:Epoch 10, Iter 8000, Dev F1 score: 0.485656, Dev EM score: 0.368588
```

Sanity {"f1": 47.55317742191654, "exact_match": 40.98765432098765}
Dev    {"f1": 52.619511182947925, "exact_match": 43.0558183538316}


## v6 Multi-headed BiDAF attention

```
INFO:root:Epoch 13, Iter 11000, dev loss: 4.530684
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.05 seconds
INFO:root:Epoch 13, Iter 11000, Train F1 score: 0.676184, Train EM score: 0.566000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 126.78 seconds
INFO:root:Epoch 13, Iter 11000, Dev F1 score: 0.457048, Dev EM score: 0.342700
```

Sanity {"f1": 44.64501791033862, "exact_match": 38.51851851851852}
Dev    {"f1": 49.46892884195423, "exact_match": 40.21759697256386}

## v6_1 Multi-headed BiDAF attention
Removed fully connected output from attention layer

Stopped early since it looked exactly the same as v6

## v6_2 BiDAF modeling layer
Added bi-LSTM after BiDAF attention

```
INFO:root:Epoch 13, Iter 11000, dev loss: 3.247179
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.84 seconds
INFO:root:Epoch 13, Iter 11000, Train F1 score: 0.859833, Train EM score: 0.746000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 160.10 seconds
INFO:root:Epoch 13, Iter 11000, Dev F1 score: 0.669573, Dev EM score: 0.527091
```

Sanity {"f1": 64.2309441309658, "exact_match": 58.39506172839506}
Dev    {"f1": 72.21678027292796, "exact_match": 62.42194891201514}
#20 on dev leaderboard

## v6_3 BiDAF addl bi-LSTM for end output
Add concat for states and extra separate bi-LSTM for end output

```
INFO:root:Epoch 1, Iter 6500, dev loss: 3.038269
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 34.46 seconds
INFO:root:Epoch 1, Iter 6500, Train F1 score: 0.770729, Train EM score: 0.627000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 312.84 seconds
INFO:root:Epoch 1, Iter 6500, Dev F1 score: 0.650111, Dev EM score: 0.503801
```

Sanity {"f1": 60.849127264978705, "exact_match": 53.58024691358025}
Dev    {"f1": 70.15586507668179, "exact_match": 59.517502365184484}


## v6_4 Multi-headed BiDAF attention with modeling layer
Try multi-headed with BiDAF that has modeling layer per head

TOO SLOW!!!

```
INFO:root:Epoch 4, Iter 4500, dev loss: 3.435196
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 87.11 seconds
INFO:root:Epoch 4, Iter 4500, Train F1 score: 0.642663, Train EM score: 0.496000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 833.91 seconds
INFO:root:Epoch 4, Iter 4500, Dev F1 score: 0.585153, Dev EM score: 0.436147
```

## v6_5 BiDAF only single bi-LSTM for end output and concat

Aborted since it looked worse than v6_2

IMPORTANT: unlike bidaf including context hiddens with others seems important

## v6_6 BiDAF single bi-LSTM for end output with multi-headed and context hidden

```
INFO:root:Epoch 10, Iter 11000, dev loss: 2.989576
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 30.13 seconds
INFO:root:Epoch 10, Iter 11000, Train F1 score: 0.798626, Train EM score: 0.676000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 275.35 seconds
INFO:root:Epoch 10, Iter 11000, Dev F1 score: 0.653341, Dev EM score: 0.510442
```

Sanity {"f1": 63.14716189455512, "exact_match": 56.17283950617284}
Dev    {"f1": 70.46553459499877, "exact_match": 60.24597918637654}

continuing with  --learning_rate=0.0005 --max_gradient_norm=3.5

```
INFO:root:Epoch 1, Iter 13500, dev loss: 3.000100
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 29.79 seconds
INFO:root:Epoch 1, Iter 13500, Train F1 score: 0.855376, Train EM score: 0.729000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 272.91 seconds
INFO:root:Epoch 1, Iter 13500, Dev F1 score: 0.668812, Dev EM score: 0.523434
```

Sanity {"f1": 63.99229259782295, "exact_match": 56.41975308641975}
Dev    {"f1": 71.92831101447436, "exact_match": 61.485335856196784}

## v6_7 BiDAF only single bi-LSTM for end output and add context hidden

```
INFO:root:Epoch 12, Iter 9500, dev loss: 3.099524
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.40 seconds
INFO:root:Epoch 12, Iter 9500, Train F1 score: 0.824909, Train EM score: 0.703000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 158.76 seconds
INFO:root:Epoch 12, Iter 9500, Dev F1 score: 0.664953, Dev EM score: 0.519199
```

Sanity {"f1": 63.60864993886001, "exact_match": 55.67901234567901}
Dev    {"f1": 72.17569729920082, "exact_match": 61.91106906338695}

# v7 merge v6_7 and add two more fully connected output layers and update test span prediction

```
INFO:root:Epoch 9, Iter 7000, dev loss: 3.072619
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 19.11 seconds
INFO:root:Epoch 9, Iter 7000, Train F1 score: 0.796002, Train EM score: 0.668000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 163.21 seconds
INFO:root:Epoch 9, Iter 7000, Dev F1 score: 0.660192, Dev EM score: 0.513040
```

Sanity {"f1": 63.84285273800261, "exact_match": 58.02469135802469}
Dev    {"f1": 71.46719122614957, "exact_match": 60.946073793755914}

Continuing with --learning_rate=0.0005 --max_gradient_norm=3.5

```
INFO:root:Epoch 2, Iter 9500, dev loss: 3.201796
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 19.42 seconds
INFO:root:Epoch 2, Iter 9500, Train F1 score: 0.880579, Train EM score: 0.780000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 161.50 seconds
INFO:root:Epoch 2, Iter 9500, Dev F1 score: 0.669300, Dev EM score: 0.523434
```

Sanity {"f1": 66.5627248127454, "exact_match": 60.24691358024691}
Dev    {"f1": 72.22309247029055, "exact_match": 62.06244087038789}

With test span prediction changes

Sanity {"f1": 67.76893773453804, "exact_match": 61.358024691358025}
Dev    {"f1": 73.99727522796228, "exact_match": 63.33017975402081}

# v7_1 remove attn_output from final concat and reduce final hidden layers by one

```
INFO:root:Epoch 3, Iter 19500, dev loss: 3.300917
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 32.96 seconds
INFO:root:Epoch 3, Iter 19500, Train F1 score: 0.912404, Train EM score: 0.806000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10390 examples in dev set took 323.23 seconds
INFO:root:Epoch 3, Iter 19500, Dev F1 score: 0.685307, Dev EM score: 0.534264
```

Sanity {"f1": 66.14765865732052, "exact_match": 58.39506172839506}
Dev    {"f1": 74.16875542104182, "exact_match": 63.87890255439925}

(Span <= 15)

# v7_2 reduce context_len to 400 from 600
*Faster*

(Results relative to v7 not v7_1.)
```
INFO:root:Epoch 12, Iter 10000, dev loss: 3.227865
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 13.58 seconds
INFO:root:Epoch 12, Iter 10000, Train F1 score: 0.861214, Train EM score: 0.744000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 114.51 seconds
INFO:root:Epoch 12, Iter 10000, Dev F1 score: 0.674911, Dev EM score: 0.520450
```

Sanity {"f1": 66.31548093541456, "exact_match": 59.01234567901235}
Dev    {"f1": 72.96486753246917, "exact_match": 61.97729422894986}

# v7_3 merge v7_1 and v7_2

```
INFO:root:Epoch 20, Iter 16500, dev loss: 3.642466
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 13.76 seconds
INFO:root:Epoch 20, Iter 16500, Train F1 score: 0.945380, Train EM score: 0.876000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 109.51 seconds
INFO:root:Epoch 20, Iter 16500, Dev F1 score: 0.674633, Dev EM score: 0.525743
```

Sanity {"f1": 68.83982837808644, "exact_match": 62.22222222222222}
Dev    {"f1": 74.08840183449259, "exact_match": 63.822138126773886}
#11 on dev leaderboard

# v7_3a change to span <= 15

Sanity {"f1": 69.05652619581767, "exact_match": 62.592592592592595}
Dev    {"f1": 74.23802196743264, "exact_match": 64.24787133396404}
#11 on dev leaderboard

# v7_4 hidden size of 250

```
INFO:root:Epoch 6, Iter 13000, dev loss: 3.131100
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 22.91 seconds
INFO:root:Epoch 6, Iter 13000, Train F1 score: 0.860003, Train EM score: 0.735000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 204.20 seconds
INFO:root:Epoch 6, Iter 13000, Dev F1 score: 0.680004, Dev EM score: 0.532673
```

Sanity {"f1": 67.9107345206716, "exact_match": 60.617283950617285}
Dev    {"f1": 73.42958530887074, "exact_match": 63.074739829706715}

# v7_5 hidden size of 300
Killed since it was slow

# v8 separate bi-LSTM for end output
Switched part way to --learning_rate=0.0005 --max_gradient_norm=3.5

```
INFO:root:Epoch 2, Iter 6000, dev loss: 3.101754
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 37.47 seconds
INFO:root:Epoch 2, Iter 6000, Train F1 score: 0.732899, Train EM score: 0.598000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10390 examples in dev set took 364.49 seconds
INFO:root:Epoch 2, Iter 6000, Dev F1 score: 0.635522, Dev EM score: 0.488354
```

Continuing with  --learning_rate=0.0001 --max_gradient_norm=2.5

```
INFO:root:Epoch 2, Iter 7500, dev loss: 2.989900
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 37.06 seconds
INFO:root:Epoch 2, Iter 7500, Train F1 score: 0.764038, Train EM score: 0.622000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10390 examples in dev set took 364.48 seconds
INFO:root:Epoch 2, Iter 7500, Dev F1 score: 0.649537, Dev EM score: 0.501540
```

Continuing with  --learning_rate=0.00005 --max_gradient_norm=2.25

```
INFO:root:Epoch 4, Iter 13000, dev loss: 2.981783
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 38.49 seconds
INFO:root:Epoch 4, Iter 13000, Train F1 score: 0.763715, Train EM score: 0.625000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10390 examples in dev set took 371.89 seconds
INFO:root:Epoch 4, Iter 13000, Dev F1 score: 0.654272, Dev EM score: 0.508181
```

Sanity {"f1": 62.62193004114403, "exact_match": 55.67901234567901}
Dev    {"f1": 70.80925181075419, "exact_match": 60.55818353831599}

# v9 multi-headed with merge of v7* and full size states for concat

```
INFO:root:Epoch 15, Iter 12500, dev loss: 3.025343
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.25 seconds
INFO:root:Epoch 15, Iter 12500, Train F1 score: 0.861873, Train EM score: 0.753000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 154.67 seconds
INFO:root:Epoch 15, Iter 12500, Dev F1 score: 0.675407, Dev EM score: 0.528438
```

Sanity {"f1": 65.17359804016536, "exact_match": 57.407407407407405}
Dev    {"f1": 72.78536982973944, "exact_match": 62.03405865657521}

Continuing with train-slow

```
INFO:root:Epoch 2, Iter 16000, dev loss: 3.216027
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.39 seconds
INFO:root:Epoch 2, Iter 16000, Train F1 score: 0.869798, Train EM score: 0.756000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 156.24 seconds
INFO:root:Epoch 2, Iter 16000, Dev F1 score: 0.676911, Dev EM score: 0.532095
```

Sanity {"f1": 62.943974860844435, "exact_match": 56.41975308641975}
Dev    {"f1": 72.83323025933947, "exact_match": 62.29895931882687}

# v10 self-attention
Self attention of bidaf attention projected to 2*hidden

```
INFO:root:Epoch 2, Iter 4500, dev loss: 5.543233
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 49.78 seconds
INFO:root:Epoch 2, Iter 4500, Train F1 score: 0.408550, Train EM score: 0.303000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10389 examples in dev set took 509.50 seconds
INFO:root:Epoch 2, Iter 4500, Dev F1 score: 0.352584, Dev EM score: 0.239773
```

Aborted due to slow learning

# v10_1 self-attention of bidaf attention with no projection
Removed fully connected projection layer

Aborted to restart with Cudnn

# v10_2 self-attention of bidaf attention with no projection
Removed fully connected projection layer and switched to CudnnCompatible*

# v11 multi-headed with separate projection matrices

## Backlog

# train half of embeddings

# adjust dropout

# adadelta optimizer

# n-grams

# positional features

# similarity head from bidaf for multiheaded

# bidaf

# more stacking for rnn cells

# feed forward before softmax

# residuals via short circuits and layernorm

# multiheaded attention

# revisit attention bilinear form with regularization
