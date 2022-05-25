# newa clausal complementation

## Annotated dataset

All data is in [one set](https://github.com/boruizhang/newa/blob/master/for_ml/k_folds/all.txt).
The embedded clauses are in brackets.

## Experiments:

[10-folds evaluation](https://github.com/boruizhang/newa/blob/master/for_ml/k_folds/newa_nerda_10folds.ipynb)

[fine-tuning mBert](https://github.com/boruizhang/newa/blob/master/for_ml/newa_nerda_IOBCP.ipynb)

## conlleval.py script

`conlleval.pl` takes the standard CoNLL
format and outputs the performance.  The sentences are in the normal
conll format and the final two columns should be correct and predicted.
