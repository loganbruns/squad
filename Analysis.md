# Observations

## Longest span

```
$ cat data/train.span | awk '{print $2-$1}' | sort -rn | head -1
45
$ cat data/dev.span | awk '{print $2-$1}' | sort -rn | head -1
36
```

## Reversed start and end

As much as 11% in some batches during testing

{"f1": 71.46719122614957, "exact_match": 60.946073793755914} to {"f1": 73.10965589246446, "exact_match": 62.090823084200565} by changing prediction logic to [start_pos, start_pos+50]

## Longest sentence

```
$ cat data/train.context | awk '{print NF}' | sort -rn | head
766
```

```
$ cat data/train.question | awk '{print NF}' | sort -rn | head
60
```

```
$ cat data/train.context | awk '{print NF}' | sort -rn > train.context_len

>>> context_lens = np.loadtxt('train.context_len')
>>> len(context_lens)
86326
>>> np.mean(context_lens)
137.90535875634222
>>> np.std(context_lens)
56.88903485667903
>>> len(filter(lambda x: x > 200, context_lens))
10532
>>> len(context_lens)
86326
>>> np.std(context_lens) + 3 * np.std(context_lens)
227.55613942671613
>>> len(filter(lambda x: x > 250, context_lens))
3758
>>> len(filter(lambda x: x > 300, context_lens))
1428
>>> len(filter(lambda x: x > 400, context_lens))
182

$ cat data/train.span | awk '{print $2}' | sort -rn > train.span_ends

>>> span_ends = np.loadtxt('train.span_ends')
>>> span_ends 
array([605., 599., 598., ...,   0.,   0.,   0.])
>>> len(filter(lambda x: x > 400, span_ends))
15
>>> len(span_ends)
86326

*Seems like context_len = 400 may be fine.*

```

```
$ cat data/train.question | awk '{print NF}' | sort -rn > train.question_len

>>> question_lens = np.loadtxt('train.question_len')
>>> len(filter(lambda x: x > 30, question_lens))
>>> len(question_lens)
86326
```

# Span distribution

```
cat data/train.span | awk '{print $2-$1}' | sort -rn > train.spans

>>> span_lens = np.loadtxt('train.spans')
>>> h, x1 = np.histogram(span_lens, bins=50)
>>> dx = x1[1] - x1[0]
>>> ((np.cumsum(h) * dx) / len(span_lens)) / sum((np.cumsum(h) * dx) / len(span_lens))
array([0.00685571, 0.01226396, 0.01533507, 0.01695067, 0.0179483 ,
       0.01860014, 0.01908117, 0.01944693, 0.01970743, 0.01970743,
       0.01991504, 0.02008099, 0.02022306, 0.02034514, 0.02044505,
       0.02053545, 0.02060393, 0.02066387, 0.02072552, 0.02072552,
       0.02077913, 0.02082178, 0.0208615 , 0.0208905 , 0.02092169,
       0.02094484, 0.02097262, 0.02098992, 0.02100308, 0.02100308,
       0.0210155 , 0.02102208, 0.02102623, 0.02102915, 0.02103183,
       0.02103329, 0.02103378, 0.02103427, 0.02103476, 0.02103476,
       0.021035  , 0.02103549, 0.02103549, 0.02103549, 0.02103549,
       0.02103573, 0.02103573, 0.02103573, 0.02103573, 0.02103597])
>>> h
array([28134, 22194, 12603,  6630,  4094,  2675,  1974,  1501,  1069,
           0,   852,   681,   583,   501,   410,   371,   281,   246,
         253,     0,   220,   175,   163,   119,   128,    95,   114,
          71,    54,     0,    51,    27,    17,    12,    11,     6,
           2,     2,     2,     0,     1,     2,     0,     0,     0,
           1,     0,     0,     0,     1])
>>> len(span_lens)
86326
>>> len(filter(lambda x: x > 15, span_lens))
1773
```

# Characters

```
$ cat data/train.context data/train.question | tr -d ' ' | sed 's/\(.\)/\1\n/g' | sort | uniq > data/characters.txt
$ wc -l data/characters.txt 
932 data/characters.txt
```