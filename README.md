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

## v7 merge v6_7 and add two more fully connected output layers and update test span prediction

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

## v7_1 remove attn_output from final concat and reduce final hidden layers by one

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

## v7_2 reduce context_len to 400 from 600
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

## v7_3 merge v7_1 and v7_2

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

## v7_3a change to span <= 15

Sanity {"f1": 69.05652619581767, "exact_match": 62.592592592592595}
Dev    {"f1": 74.23802196743264, "exact_match": 64.24787133396404}
#11 on dev leaderboard

## v7_4 hidden size of 250

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

## v7_5 hidden size of 300
Killed since it was slow

## v8 separate bi-LSTM for end output
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

## v9 multi-headed with merge of v7* and full size states for concat

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

## v10 self-attention
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

## v10_1 self-attention of bidaf attention with no projection
Removed fully connected projection layer

Aborted to restart with Cudnn

## v10_2 self-attention of bidaf attention with no projection
Removed fully connected projection layer and switched to CudnnCompatible*

```
INFO:root:Epoch 5, Iter 19000, dev loss: 3.043008
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 45.63 seconds
INFO:root:Epoch 5, Iter 19000, Train F1 score: 0.819805, Train EM score: 0.689000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10389 examples in dev set took 469.45 seconds
INFO:root:Epoch 5, Iter 19000, Dev F1 score: 0.668080, Dev EM score: 0.522379
```

Sanity {"f1": 64.06971901700273, "exact_match": 55.55555555555556}
Dev    {"f1": 72.27979367413685, "exact_match": 61.79754020813623}

Note: very slow due to small batch size of 25

## v10_3 self-attention of bidaf attention with attn_size=75

```
INFO:root:Epoch 5, Iter 8000, dev loss: 3.197439
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 13.75 seconds
INFO:root:Epoch 5, Iter 8000, Train F1 score: 0.867845, Train EM score: 0.730000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 110.98 seconds
INFO:root:Epoch 5, Iter 8000, Dev F1 score: 0.679789, Dev EM score: 0.532480
```

Sanity {"f1": 68.24940412398973, "exact_match": 61.23456790123457}
Dev    {"f1": 73.57862724757865, "exact_match": 62.980132450331126}

## v11 multi-headed with separate projection matrices

Aborted to restart with Cudnn

## v11_1 multi-headed with separate projection matrices

```
INFO:root:Epoch 7, Iter 7500, dev loss: 3.002624
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.98 seconds
INFO:root:Epoch 7, Iter 7500, Train F1 score: 0.815278, Train EM score: 0.686000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 143.95 seconds
INFO:root:Epoch 7, Iter 7500, Dev F1 score: 0.668733, Dev EM score: 0.524107
```

Sanity {"f1": 65.71772165330948, "exact_match": 58.641975308641975}
Dev    {"f1": 72.46684999452987, "exact_match": 62.37464522232734}

## v11_2 multi-headed with separate projection matrices and shared W per head
Share the same weight matrix per head such qn and contexts are more easily comparable

```
INFO:root:Epoch 6, Iter 10000, dev loss: 3.160727
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.71 seconds
INFO:root:Epoch 6, Iter 10000, Train F1 score: 0.874699, Train EM score: 0.748000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 142.10 seconds
INFO:root:Epoch 6, Iter 10000, Dev F1 score: 0.680009, Dev EM score: 0.531325
```

Sanity {"f1": 65.66194861169762, "exact_match": 58.51851851851852}
Dev    {"f1": 73.71308043170023, "exact_match": 63.41532639545885}

## v12 new baseline with CudnnCompatible* and swap_memory=True

```
INFO:root:Epoch 7, Iter 8500, dev loss: 3.027820
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 14.96 seconds
INFO:root:Epoch 7, Iter 8500, Train F1 score: 0.836449, Train EM score: 0.703000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 120.45 seconds
INFO:root:Epoch 7, Iter 8500, Dev F1 score: 0.679710, Dev EM score: 0.532480
```

Sanity {"f1": 65.93260656263881, "exact_match": 59.25925925925926}
Dev    {"f1": 73.58635037870457, "exact_match": 63.10312204351939}

## v12_1 try RNN layers = 3

Appears to be converging to v12 (only slower)

```
INFO:root:Epoch 8, Iter 6500, dev loss: 3.002166
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.78 seconds
INFO:root:Epoch 8, Iter 6500, Train F1 score: 0.782680, Train EM score: 0.660000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 163.69 seconds
INFO:root:Epoch 8, Iter 6500, Dev F1 score: 0.667698, Dev EM score: 0.517660
```

Sanity {"f1": 64.62205634468117, "exact_match": 57.28395061728395}
Dev    {"f1": 72.23655320270973, "exact_match": 61.258278145695364}

## v12_2 try hidden size = 250

```
Calculating dev loss...
Epoch 10, Iter 8000, dev loss: 3.181835
Epoch 10, Iter 8000, Train F1 score: 0.916957, Train EM score: 0.824000
Epoch 10, Iter 8000, Dev F1 score: 0.681718, Dev EM score: 0.533635
```

Sanity {"f1": 64.27701969408896, "exact_match": 57.901234567901234}
Dev    {"f1": 73.74353716544717, "exact_match": 63.614001892147584}

## v12_3 add layer_norm at end of rnn output

Does not seem to improve curve.

```
INFO:root:Epoch 7, Iter 5500, dev loss: 3.315401
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 15.30 seconds
INFO:root:Epoch 7, Iter 5500, Train F1 score: 0.670365, Train EM score: 0.535000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 126.28 seconds
INFO:root:Epoch 7, Iter 5500, Dev F1 score: 0.622911, Dev EM score: 0.470215
```

## v12_4 use adadelta optimizer with lr=.5

Aborted due to slow learning. Will retry with lr=1

```
INFO:root:Epoch 2, Iter 1500, dev loss: 6.709846
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 11.49 seconds
INFO:root:Epoch 2, Iter 1500, Train F1 score: 0.214812, Train EM score: 0.146000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 85.94 seconds
INFO:root:Epoch 2, Iter 1500, Dev F1 score: 0.212121, Dev EM score: 0.140987
INFO:root:Saving to ./experiments/v12_4/best_checkpoint/qa_best.ckpt...
```

## v12_5 use adadelta optimizer with lr=1

Aborted due to slow learning.

## v12_6 stack RNN layers (2x2)

```
INFO:root:Epoch 14, Iter 16500, dev loss: 3.356169
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 29.45 seconds
INFO:root:Epoch 14, Iter 16500, Train F1 score: 0.882800, Train EM score: 0.775000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 272.67 seconds
INFO:root:Epoch 14, Iter 16500, Dev F1 score: 0.673327, Dev EM score: 0.528823
```

Sanity {"f1": 63.0581082093823, "exact_match": 56.17283950617284}
Dev    {"f1": 72.83562431371074, "exact_match": 62.69631031220435}

## v12_7 stack RNN layers (2x1)

```
INFO:root:Epoch 7, Iter 6500, dev loss: 2.985729
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.28 seconds
INFO:root:Epoch 7, Iter 6500, Train F1 score: 0.872066, Train EM score: 0.762000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 135.64 seconds
INFO:root:Epoch 7, Iter 6500, Dev F1 score: 0.687000, Dev EM score: 0.536041
```

Sanity {"f1": 67.59769937539241, "exact_match": 60.370370370370374}
Dev    {"f1": 74.34160453547857, "exact_match": 63.746452223273415}

## v12_8 stack RNN layers (3x1)

```
INFO:root:Epoch 8, Iter 9000, dev loss: 3.056645
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 26.76 seconds
INFO:root:Epoch 8, Iter 9000, Train F1 score: 0.858259, Train EM score: 0.729000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 244.81 seconds
INFO:root:Epoch 8, Iter 9000, Dev F1 score: 0.690127, Dev EM score: 0.540949
INFO:root:Saving to ./experiments/v12_8/best_checkpoint/qa_best.ckpt...
```

Sanity {"f1": 66.34450981713464, "exact_match": 58.888888888888886}
Dev    {"f1": 74.49557453257329, "exact_match": 64.24787133396404}

## v13 add | c - a | style features to BiDaf

```
INFO:root:Epoch 14, Iter 13000, dev loss: 3.442659
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.43 seconds
INFO:root:Epoch 14, Iter 13000, Train F1 score: 0.841963, Train EM score: 0.711000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 137.45 seconds
INFO:root:Epoch 14, Iter 13000, Dev F1 score: 0.665741, Dev EM score: 0.518814
```

Sanity {"f1": 66.34345805682364, "exact_match": 59.75308641975309}
Dev    {"f1": 71.91682622965291, "exact_match": 61.49479659413434}

## v14 make embeddings trainable regularized with layer_norm

```
INFO:root:Epoch 3, Iter 4000, dev loss: 3.070076
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 15.07 seconds
INFO:root:Epoch 3, Iter 4000, Train F1 score: 0.834436, Train EM score: 0.712000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 120.78 seconds
INFO:root:Epoch 3, Iter 4000, Dev F1 score: 0.668219, Dev EM score: 0.522183
```

Sanity {"f1": 64.33516079852636, "exact_match": 56.79012345679013}
Dev    {"f1": 72.5009416362869, "exact_match": 62.27057710501419}

## v14_1 make embeddings trainable with a loss term relative to originals
(relative to v12_7)

```
INFO:root:Epoch 6, Iter 4500, dev loss: 3.086039
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.03 seconds
INFO:root:Epoch 6, Iter 4500, Train F1 score: 0.887037, Train EM score: 0.772000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 139.59 seconds
INFO:root:Epoch 6, Iter 4500, Dev F1 score: 0.690741, Dev EM score: 0.541238
```

Sanity {"f1": 65.89297986560469, "exact_match": 60.0}
Dev    {"f1": 74.68961207605716, "exact_match": 64.54115421002838}
#33 on dev leaderboard

## v15 self-attention of bidaf attention with attn_size=50
(relative to v12_7)

```
INFO:root:Epoch 12, Iter 9500, dev loss: 3.247487
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 14.57 seconds
INFO:root:Epoch 12, Iter 9500, Train F1 score: 0.890551, Train EM score: 0.771000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 120.68 seconds
INFO:root:Epoch 12, Iter 9500, Dev F1 score: 0.693712, Dev EM score: 0.547878
```

Sanity {"f1": 68.42045069814374, "exact_match": 61.60493827160494}
Dev    {"f1": 74.84457410851721, "exact_match": 64.98580889309366}

## v15_1 self-attention of bidaf attention with cross attention feature

```
INFO:root:Epoch 6, Iter 6500, dev loss: 2.880845
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 18.66 seconds
INFO:root:Epoch 6, Iter 6500, Train F1 score: 0.792174, Train EM score: 0.662000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 162.84 seconds
INFO:root:Epoch 6, Iter 6500, Dev F1 score: 0.689395, Dev EM score: 0.543259
```

Sanity {"f1": 67.5689438193464, "exact_match": 60.370370370370374}
Dev    {"f1": 74.71791270873666, "exact_match": 64.720908230842}

## v15_2 self-attention of bidaf attention with attn_size=100

```
INFO:root:Epoch 5, Iter 5500, dev loss: 2.899911
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.62 seconds
INFO:root:Epoch 5, Iter 5500, Train F1 score: 0.786296, Train EM score: 0.646000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 150.73 seconds
INFO:root:Epoch 5, Iter 5500, Dev F1 score: 0.684654, Dev EM score: 0.536714
```

Sanity {"f1": 65.3878289988554, "exact_match": 56.17283950617284}
Dev    {"f1": 74.02533719751729, "exact_match": 63.73699148533586}

## v15_3 self-attention of bidaf attention and orthogonality loss

```
INFO:root:Epoch 7, Iter 5500, dev loss: 81.656080
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 14.56 seconds
INFO:root:Epoch 7, Iter 5500, Train F1 score: 0.853902, Train EM score: 0.738000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 121.70 seconds
INFO:root:Epoch 7, Iter 5500, Dev F1 score: 0.686598, Dev EM score: 0.536522
```

Sanity {"f1": 68.63707060098082, "exact_match": 62.46913580246913}
Dev    {"f1": 74.3349248055598, "exact_match": 63.99243140964995}

## v15_4 self-attention of bidaf attention and diversity loss

```
INFO:root:Epoch 2, Iter 10500, dev loss: 3.511083
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 15.09 seconds
INFO:root:Epoch 2, Iter 10500, Train F1 score: 0.955555, Train EM score: 0.890000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 120.52 seconds
INFO:root:Epoch 2, Iter 10500, Dev F1 score: 0.686599, Dev EM score: 0.535271
```

Sanity {"f1": 66.76979829852336, "exact_match": 59.876543209876544}
Dev    {"f1": 73.96536586259387, "exact_match": 63.24503311258278}

## v16 multi-headed with 4 heads instead of 8
(relative to v12_7)

```
INFO:root:Epoch 14, Iter 11500, dev loss: 3.194167
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.04 seconds
INFO:root:Epoch 14, Iter 11500, Train F1 score: 0.896673, Train EM score: 0.782000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 145.44 seconds
INFO:root:Epoch 14, Iter 11500, Dev F1 score: 0.683118, Dev EM score: 0.536714
```

Sanity {"f1": 66.03049427163761, "exact_match": 58.76543209876543}
Dev    {"f1": 73.81476469731143, "exact_match": 63.661305581835386}

## v16_1 multi-headed with original basic attention

```
INFO:root:Epoch 8, Iter 9000, dev loss: 2.941654
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 24.79 seconds
INFO:root:Epoch 8, Iter 9000, Train F1 score: 0.791471, Train EM score: 0.650000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 218.68 seconds
INFO:root:Epoch 8, Iter 9000, Dev F1 score: 0.684368, Dev EM score: 0.532576
```

Sanity {"f1": 67.1344705624331, "exact_match": 60.123456790123456}
Dev    {"f1": 74.00610636156621, "exact_match": 63.528855250709555}

after train-slow

Dev    {"f1": 74.66017833088468, "exact_match": 64.30463576158941}

## v16_2 multi-headed with bidaf and diversity loss addition

```
INFO:root:Epoch 2, Iter 10000, dev loss: 3.193834
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.35 seconds
INFO:root:Epoch 2, Iter 10000, Train F1 score: 0.861112, Train EM score: 0.745000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 146.29 seconds
INFO:root:Epoch 2, Iter 10000, Dev F1 score: 0.671152, Dev EM score: 0.527668
```

Sanity {"f1": 63.80541345396422, "exact_match": 57.407407407407405}
Dev    {"f1": 72.44432041308498, "exact_match": 62.34626300851466}

## v16_3 multi-headed with original basic attention and diversity loss addition

```
INFO:root:Epoch 10, Iter 11000, dev loss: 3.153353
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 24.38 seconds
INFO:root:Epoch 10, Iter 11000, Train F1 score: 0.869879, Train EM score: 0.748000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 212.09 seconds
INFO:root:Epoch 10, Iter 11000, Dev F1 score: 0.686256, Dev EM score: 0.535752
```

Sanity {"f1": 66.32659804873553, "exact_match": 58.888888888888886}
Dev    {"f1": 74.09163479498308, "exact_match": 63.69914853358562}

train-slow

Sanity {"f1": 68.78652232691461, "exact_match": 60.370370370370374}
Dev    {"f1": 74.66231751774502, "exact_match": 63.964049195837276}

## v16_4 multi-headed with original basic attention, diversity loss addition, layer_norm

```
INFO:root:Epoch 17, Iter 19500, dev loss: 3.398669
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 24.25 seconds
INFO:root:Epoch 17, Iter 19500, Train F1 score: 0.769239, Train EM score: 0.636000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 220.18 seconds
INFO:root:Epoch 17, Iter 19500, Dev F1 score: 0.671809, Dev EM score: 0.521028
```

Sanity {"f1": 65.37629663547116, "exact_match": 57.53086419753087}
Dev    {"f1": 72.8397258062988, "exact_match": 62.02459791863765}

## v16_5 multi-headed with original basic attention, diversity loss addition, eight heads
Very slow learning and training rates and appears to be converging to the same place. 

## v17 200 dimensional embeddings

```
INFO:root:Epoch 10, Iter 8000, dev loss: 3.269544
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.49 seconds
INFO:root:Epoch 10, Iter 8000, Train F1 score: 0.921450, Train EM score: 0.822000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 134.29 seconds
INFO:root:Epoch 10, Iter 8000, Dev F1 score: 0.691624, Dev EM score: 0.540083
```

Sanity {"f1": 69.6038133767179, "exact_match": 61.60493827160494}
Dev    {"f1": 74.54686193313371, "exact_match": 63.79375591296121}

## v17_1 300 dimensional embeddings

```
INFO:root:Epoch 6, Iter 4500, dev loss: 3.065319
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.13 seconds
INFO:root:Epoch 6, Iter 4500, Train F1 score: 0.876436, Train EM score: 0.749000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 135.43 seconds
INFO:root:Epoch 6, Iter 4500, Dev F1 score: 0.682618, Dev EM score: 0.531518
```

Sanity {"f1": 63.986630042588196, "exact_match": 56.54320987654321}
Dev    {"f1": 73.91397185393312, "exact_match": 63.292336802270576}

## v17_2 adadelta with .5

```
INFO:root:Epoch 19, Iter 15500, dev loss: 3.275553
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.48 seconds
INFO:root:Epoch 19, Iter 15500, Train F1 score: 0.771683, Train EM score: 0.639000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 137.75 seconds
INFO:root:Epoch 19, Iter 15500, Dev F1 score: 0.642913, Dev EM score: 0.497738
```

Aborted due to slow progress

## v18 merge v14_1 and increase embeddings to 200

```
INFO:root:Epoch 6, Iter 6000, dev loss: 3.041146
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 20.43 seconds
INFO:root:Epoch 6, Iter 6000, Train F1 score: 0.912042, Train EM score: 0.811000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 176.20 seconds
INFO:root:Epoch 6, Iter 6000, Dev F1 score: 0.685901, Dev EM score: 0.534790
```

Sanity {"f1": 66.75444367113904, "exact_match": 58.02469135802469}
Dev    {"f1": 74.17111684576615, "exact_match": 63.35856196783349}

## v18_1 merge self attention

```
INFO:root:Epoch 8, Iter 9000, dev loss: 3.529354
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 17.81 seconds
INFO:root:Epoch 8, Iter 9000, Train F1 score: 0.908688, Train EM score: 0.814000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 155.85 seconds
INFO:root:Epoch 8, Iter 9000, Dev F1 score: 0.681106, Dev EM score: 0.532865
```

Sanity {"f1": 67.78147102212704, "exact_match": 58.76543209876543}
Dev    {"f1": 73.58827318534182, "exact_match": 63.18826868495743}

## v18_2 rerun v18

```
INFO:root:Epoch 5, Iter 5000, dev loss: 3.022299
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 20.89 seconds
INFO:root:Epoch 5, Iter 5000, Train F1 score: 0.870375, Train EM score: 0.747000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 179.86 seconds
INFO:root:Epoch 5, Iter 5000, Dev F1 score: 0.684122, Dev EM score: 0.529593
```

Sanity {"f1": 64.5926390316668, "exact_match": 55.80246913580247}
Dev    {"f1": 73.89543442321445, "exact_match": 62.8949858088931}

More time

```
Epoch 1, Iter 9500, dev loss: 3.513365
Calculating F1/EM for 1000 examples in train set...
Calculating F1/EM for 1000 examples in train set took 20.28 seconds
Epoch 1, Iter 9500, Train F1 score: 0.933472, Train EM score: 0.844000
Calculating F1/EM for all examples in dev set...
Calculating F1/EM for 10391 examples in dev set took 179.49 seconds
Epoch 1, Iter 9500, Dev F1 score: 0.688439, Dev EM score: 0.536714
```

Sanity {"f1": 66.55425488314621, "exact_match": 58.51851851851852}
Dev    {"f1": 74.42028860761968, "exact_match": 63.7275307473983}

## v18_3 rerun v18

```
INFO:root:Epoch 5, Iter 5500, dev loss: 3.191027
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 20.49 seconds
INFO:root:Epoch 5, Iter 5500, Train F1 score: 0.839143, Train EM score: 0.719000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 177.12 seconds
INFO:root:Epoch 5, Iter 5500, Dev F1 score: 0.683081, Dev EM score: 0.531518
```

Sanity {"f1": 66.00362835533815, "exact_match": 56.91358024691358}
Dev    {"f1": 74.00090238072025, "exact_match": 63.48155156102176}

## v18_4 rerun v18

```
INFO:root:Epoch 4, Iter 7500, dev loss: 3.135208
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 21.28 seconds
INFO:root:Epoch 4, Iter 7500, Train F1 score: 0.885602, Train EM score: 0.791000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 183.33 seconds
INFO:root:Epoch 4, Iter 7500, Dev F1 score: 0.680744, Dev EM score: 0.531614
```

Sanity {"f1": 64.65204144874038, "exact_match": 57.28395061728395}
Dev    {"f1": 73.67780459564382, "exact_match": 63.33964049195837}

## v18_5 rerun v18

## v19 add word dropout on questions

```
INFO:root:Epoch 1, Iter 10000, dev loss: 3.392499
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 16.01 seconds
INFO:root:Epoch 1, Iter 10000, Train F1 score: 0.857063, Train EM score: 0.750000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 134.73 seconds
INFO:root:Epoch 1, Iter 10000, Dev F1 score: 0.654213, Dev EM score: 0.510538
```

Sanity {"f1": 64.27570619277547, "exact_match": 56.41975308641975}
Dev    {"f1": 70.51730121697514, "exact_match": 60.122989593188265}

## v19_1 add word dropout on contexts

```
INFO:root:Epoch 5, Iter 5500, dev loss: 3.105663
INFO:root:Calculating F1/EM for 1000 examples in train set...
INFO:root:Calculating F1/EM for 1000 examples in train set took 15.64 seconds
INFO:root:Epoch 5, Iter 5500, Train F1 score: 0.852134, Train EM score: 0.739000
INFO:root:Calculating F1/EM for all examples in dev set...
INFO:root:Calculating F1/EM for 10391 examples in dev set took 132.68 seconds
INFO:root:Epoch 5, Iter 5500, Dev F1 score: 0.682386, Dev EM score: 0.530748
```

Sanity {"f1": 64.62890904342704, "exact_match": 55.80246913580247}
Dev    {"f1": 73.7616142761364, "exact_match": 63.065279091769156}

## v20 back to 100 dimensional embeddings

```
Epoch 6, Iter 5000, dev loss: 3.034356
Epoch 6, Iter 5000, Train F1 score: 0.848582, Train EM score: 0.724000
Epoch 6, Iter 5000, Dev F1 score: 0.687549, Dev EM score: 0.540853
```

Sanity {"f1": 66.48869964650964, "exact_match": 59.50617283950617}
Dev    {"f1": 74.15823742078166, "exact_match": 63.982970671712394}

## v20_1 rerun v20

```
Epoch 6, Iter 4500, dev loss: 3.047778
Epoch 6, Iter 4500, Train F1 score: 0.882214, Train EM score: 0.756000
Epoch 6, Iter 4500, Dev F1 score: 0.688294, Dev EM score: 0.538158
```

Sanity {"f1": 65.12398620757757, "exact_match": 56.54320987654321}
Dev    {"f1": 74.53615096243219, "exact_match": 64.30463576158941}

## v20_2 rerun v20

```
Epoch 3, Iter 5500, dev loss: 3.073231
Epoch 3, Iter 5500, Train F1 score: 0.905314, Train EM score: 0.791000
Epoch 3, Iter 5500, Dev F1 score: 0.686208, Dev EM score: 0.542681
```

Sanity {"f1": 65.66281188543671, "exact_match": 59.135802469135804}
Dev    {"f1": 74.1132853060881, "exact_match": 63.99243140964995}

## v21 .01 for lambda for embeddings loss (kept)

```
INFO:root:Epoch 8, Iter 6500, dev loss: 3.059520
INFO:root:Epoch 8, Iter 6500, Train F1 score: 0.894753, Train EM score: 0.788000
INFO:root:Epoch 8, Iter 6500, Dev F1 score: 0.694040, Dev EM score: 0.547493
```

Sanity {"f1": 65.29490751704502, "exact_match": 57.03703703703704}
Dev    {"f1": 74.97793044159951, "exact_match": 64.8155156102176}

## v21_1 .1 for lambda for embeddings loss

```
INFO:root:Epoch 6, Iter 4500, dev loss: 3.063017
INFO:root:Epoch 6, Iter 4500, Train F1 score: 0.883729, Train EM score: 0.755000
INFO:root:Epoch 6, Iter 4500, Dev F1 score: 0.692043, Dev EM score: 0.541526
```

Sanity {"f1": 65.08686656875066, "exact_match": 56.666666666666664}
Dev    {"f1": 74.8216120376559, "exact_match": 64.78713339640493}

## v22 context word dropout (kept)

```
Epoch 3, Iter 13000, dev loss: 3.216774
Epoch 3, Iter 13000, Train F1 score: 0.932283, Train EM score: 0.834000
Epoch 3, Iter 13000, Dev F1 score: 0.697966, Dev EM score: 0.549322
```

Sanity {"f1": 66.83285682073218, "exact_match": 59.25925925925926}
Dev    {"f1": 75.5688252347527, "exact_match": 65.3926206244087}

# Backlog

- layer norm at end of attentions before concat

- train half of embeddings

- adjust dropout

- adadelta optimizer

- n-grams

- positional features

- similarity head from bidaf for multiheaded

- bidaf

- more stacking for rnn cells

- feed forward before softmax

- residuals via short circuits and layernorm

- multiheaded attention

- revisit attention bilinear form with regularization
