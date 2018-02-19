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

