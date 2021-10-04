# newa


## Try conlleval.py script

Hi Bri, I added an classic perl script called `conlleval.pl`.  If you
don't know perl that's okay because it's takes the standard CoNLL
format and outputs the performance.  The sentences are in the normal
conll format and the final two columns should be correct and predicted.

I think using this script may help us make sure that the format is
standard.  I think the format should be I/O/B and then '-LABEL'.  I
forgot the details of the labels, but if we pretend that all the
labels are the same and we just want to find the chunks, then we can
use the following commands to make a test of the evaluation script by
just copying the labels to another column, i.e. imagine if we have
perfect performance:

```
paste total_true_emb <(cut -f 2 total_true_emb)
```

however if we give this output to ./conlleval.pl, it's confused
because it wants the labels to be B-EC, I-EC, B-ECV, B-ECV. Also, I
think it might make sense to just have O for 'Outside', rather than
O-EC and O-ECV.

```
paste total_true_emb  <(cut -f 2 total_true_emb) | perl -pe 's/IEC-?(V)?/I-EC$1/g; s/OEC-?(V)?/O/g; s/BEC-?(V)?/B-EC$1/g;'
```

to get the results, we pipe this output to the `conlleval.pl` script
and use the -d flag to match tabs instead of just spaces:

```
paste <(cut -f 1,2 total_true_emb) <(cut -f 2 total_true_emb) | perl -pe 's/IEC-?(V)?/I-EC$1/g; s/OEC-?(V)?/O/g; s/BEC-?(V)?/B-EC$1/g;'  | ./conlleval.pl -d '\t'
processed 4303 tokens with 705 phrases; found: 705 phrases; correct: 705.
accuracy: 100.00%; precision: 100.00%; recall: 100.00%; FB1: 100.00
               EC: precision: 100.00%; recall: 100.00%; FB1: 100.00  349
              ECV: precision: 100.00%; recall: 100.00%; FB1: 100.00  356
```

So you can see it gives the prediction of two different chunk labels,
EC and ECV.  Because we copied the tags, the true and predicted labels
are the same, so the accuracy is 100%.  So I didn't really do
anything, but it's good to set up the evaluation first!
