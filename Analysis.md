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

