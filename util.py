"""
This is a file to import, with helper functions.

It also has the experiments (copied from notebooks)
"""
from collections import defaultdict
import inspect
import itertools
import os
import re
import sqlite3
import subprocess
import sys

import nltk
from nltk.tokenize import WhitespaceTokenizer
import NERDA

def conlleval(toks, trues, preds):
    """takes lists of tokens, true labels, and predictions

    returns dict of num_tokens, num_phrases, num_found, num_correct,
    accuracy, precision, recall, fb1

    This fuction runs the conlleval.py script by opening a subprocess
    and writing/piping three columns of tab separated values: token \t
    true_label \t predicted value

    """
    p1 = subprocess.Popen(["cat"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    p2 = subprocess.Popen(["./conlleval.pl", "-d", "\t"], stdin=p1.stdout, stdout=subprocess.PIPE,
                          cwd=os.path.dirname(os.path.realpath(__file__)))
    for tok, true, pred in zip(toks, trues, preds):
        #print(tok, true, pred, file=p1.stdin, sep="\t")
        p1.stdin.write(("\t".join([tok, true, pred])+"\n").encode("utf-8"))
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    p1.stdin.close() # close the input.
    output = p2.communicate()[0].decode("utf-8")
    print(output)
    pattern = re.compile(r"^processed (\d+) tokens with (\d+) phrases; found (\d) phrases; correct: (\d)\.\n*accuracy: (\d*\.\d*)%; precision: +(\d*\.*\d*); +recall: +(\d*\.\d*); *FB1 *(\d*\.\d*)",
                         re.MULTILINE)
    pattern = re.compile(r"processed (\d+) tokens with (\d+) phrases; *found: (\d*) phrases; *correct: (\d+).\naccuracy: *(\d*\.\d*)%; *precision: *(\d*\.\d*)%; *recall: +(\d*\.\d*)%; *FB1: *(\d*\.\d*)",
                         re.MULTILINE)
    match = re.search(pattern, output)
    results = {"num_tokens": int(match.group(1)),
               "num_phrases": int(match.group(2)),
               "num_found": int(match.group(3)),
               "num_correct": int(match.group(4)),
               "accuracy": float(match.group(5)),
               "precision": float(match.group(6)),
               "recall": float(match.group(7)),
               "fb1": float(match.group(8))}
    return results

def get_labeled_tokens(infilename="just_ec.conll"):
    """ reads the data from conll format, default file just_ec.conll"""
    labeled_tokens = []
    labeled_tokens.append(("<S>", "O")) #beginning of sentence marker
    for line in open(infilename):
        try:
            tok, lab, _ = line.split("\t")
            labeled_tokens.append((tok, lab))
        except ValueError:
            labeled_tokens.append(("</S>", "O")) # end of sentence marker
            labeled_tokens.append(("<S>", "O")) # beginning of sentence marker
    return labeled_tokens

def make_train_test(labeled_tokens):
    """ this picks a somewhat arbitrary point that has an sentence break
    it assumes the data as of 2021-10-10, update as needed!
    """
    return labeled_tokens[:3997], labeled_tokens[3997:]

def featfn_just_word(tok):
    """just get the word as a feature"""
    return {"word": tok}

def flatten_lol(lol):
    """flatten list of lists (lol)"""
    return [item for sublist in lol for item in sublist]


def experiment_001_naive_bayes_unigram():
    """this is a simple experiment that just uses the current word to
    predict the tag"""
    labeled_tokens = get_labeled_tokens()
    train, test = make_train_test(labeled_tokens)
    trainfeat = [({"word": tok}, lab) for (tok, lab) in train]
    testfeat  = [({"word": tok}, lab) for (tok, lab) in test]
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)
    train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    train_toks, train_true = zip(*train)
    test_toks, test_true = zip(*test)
    # considering inspect.currentframe().f_code.co_name as a way to
    # keep track of experiments, when running a battery of experiments
    results = conlleval(test_toks, test_true, test_pred)
    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def featurize_wordandtag_pseudo_bigram(tokens, classify=False):
    """input is a list/iterable of tokens, output/generator is list of dictionary features, like
    [{word_nm1: the, word_n: dog}]
    if tok == <s>, word_nm1 = "</s>" (padding)

    it's called "psuedo" because it doesn't have true joint features,
    just multiple features including previous word

    """
    prev_tok = "</S>" # end of sentence marker
    prev_lab = "O"
    for tok, lab in tokens:
        feature_dict = {}
        feature_dict["word_n"] = tok
        feature_dict["word_nm1"] = prev_tok
        feature_dict["lab_nm1"] = prev_lab
        prev_tok = tok
        if classify: # this is the part that makes it honest fair, see below
            lab = classify(feature_dict)
        prev_lab = lab
        yield feature_dict, lab

def experiment_002_naive_bayes_pseudo_bigram_dishonest():
    """this is dishonest because it uses the true previous tags: it
    should use the predicted previous tags

    """
    labeled_tokens = get_labeled_tokens()
    train, test = make_train_test(labeled_tokens)

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    testfeat = list(featurize_wordandtag_pseudo_bigram(test))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results


def experiment_002_naive_bayes_pseudo_bigram_honest():
    """uses the classify argument for the feature extractor so that tag
    related features are based on predictions

    """
    labeled_tokens = get_labeled_tokens()
    train, test = make_train_test(labeled_tokens)

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #util.conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)

    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results


def experiment_003_maxent_pseudo_bigram():
    """this is a maximum entropy model that uses the current and
    preceding words as separate features (not joint)"""
    labeled_tokens = get_labeled_tokens()
    train, test = make_train_test(labeled_tokens)

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_pseudo_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]

    test_toks, test_true = zip(*test)
    print(test_toks)
    print(test_true)
    print(test_pred)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results



def featurize_wordandtag_bigram(tokens, classify=False):
    """actually the previous was not really bigram, only two word context
    here's a better bigram using joint features

    input is a list/iterable of tokens, output/generator is list of dictionary features, like
    [{word_nm1: the, word_n: dog}]
    if tok == <s>, word_nm1 = "</s>" (padding)

    """
    prev_tok = "</S>" # end of sentence marker
    prev_lab = "O"
    for tok, lab in tokens:
        feature_dict = {}
        feature_dict["word_n"] = tok
        feature_dict["word_n-1"] = prev_tok
        feature_dict["word_n-1,word_n"] = prev_tok + ","  + tok
        feature_dict["lab_n-1"] = prev_lab
        feature_dict["lab_n-1,word_n"] = prev_lab + ","  + tok
        feature_dict["lab_n-1,word_n-1"] = prev_lab + ","  + prev_tok
        feature_dict["lab_n-1,word_n-1,word_n"] = prev_lab + ","  + prev_tok + "," + tok
        prev_tok = tok
        if classify: # this is the part that makes it honest fair, see below
            lab = classify(feature_dict)
        prev_lab = lab
        yield feature_dict, lab

def experiment_004_maxent_bigram():
    """ this is a maxent model using proper bigrams"""

    labeled_tokens = get_labeled_tokens()
    train, test = make_train_test(labeled_tokens)

    trainfeat = list(featurize_wordandtag_bigram(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def nerda_format(data):
    """ from Bri's notebook """
    ner = defaultdict(list)
    for line1,line2 in itertools.zip_longest(*[data]*2):
        sents = []
        tags = []
        tk = WhitespaceTokenizer()
        tokens = iter(tk.tokenize(line1))
        current_tag = "O"
        for token in tokens:
            suffix = ""
            if token in line2:
                suffix = "-V"
                # Good: even if the verb is complex (aka spaced
                # out),suffix can still apply to the entire phrase

                # Bad: false positive -V may occur, if one word that
                # has multiple meanings occurring more than one time
                # in the same sentence
            try:
                if token == "[":
                    current_tag = "I"
                    tags.append("B" + suffix)
                    sents.append(next(tokens))
                elif token == "]":
                    current_tag = "O"
                    #     tags.append(current_tag+suffix)
                    #     sents.append(next(tokens))
                else:
                    tags.append(current_tag+suffix)
                    sents.append(token)
            except StopIteration:
                pass
        ner['sentences'].append(sents)
        ner['tags'].append(tags)
    return ner

def nerda_format_just_iob(data):
    """just IOB"""
    ner = defaultdict(list)
    for line1,line2 in itertools.zip_longest(*[data]*2):
        sents = []
        tags = []
        tk = WhitespaceTokenizer()
        tokens = iter(tk.tokenize(line1))
        current_tag = "O"
        for token in tokens:
            try:
                if token == "[":
                    current_tag = "I"
                    tags.append("B")
                    sents.append(next(tokens))
                elif token == "]":
                    current_tag = "O"
                        #     tags.append(current_tag+suffix)
                        #     sents.append(next(tokens))
                else:
                    tags.append(current_tag)
                    sents.append(token)
            except StopIteration:
                pass
        ner['sentences'].append(sents)
        ner['tags'].append(tags)
    return ner


def make_nerda_train_dev_test(format_fn=nerda_format):
    """ return the 3 data partition in nerda format

    pass in the format function: Bri's original is the default

    to use conlleval, we need nerda_format_just_iob
    """

    with open('for_ml/train.txt') as data:
        train = format_fn(data)
        #  t['sentences'] = ner['sentences']
        #  t['tags'] = ner['tags']

    with open('for_ml/dev.txt') as data:
        dev  = format_fn(data)
        #d['sentences'] = ner['sentences']
        #d['tags'] = ner['tags']

    with open('for_ml/test.txt') as data:
        test = format_fn(data)
        #  test['sentences'] = ner['sentences']
        #  test['tags'] = ner['tags']

    return train, dev, test


def experiment_005_bri_nerda():
    """ uses Bri's model but conll chunk evaluation  """
    train, dev, test = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    tag_scheme = ['B', 'I']

    model = NERDA(
        dataset_training = train,
        dataset_validation = dev,
        tag_scheme = tag_scheme,
        tag_outside = 'O',
        transformer = 'bert-base-multilingual-cased',
        hyperparameters = {'epochs' : 11,
                           'warmup_steps' : 10,
                           'train_batch_size': 5,
                           'learning_rate': 0.0001})
    model.train()
    model.evaluate_performance(test)
    pred_test  = model.predict(test['sentences'])
    results = conlleval(flatten_lol(test['sentences']),
                        flatten_lol(test['tags']),
                        flatten_lol(pred_test))

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def to_padded_tok_tag_tuples_from_nerda(partition):
    """takes the nerda format (dict with "sentences" and "tags" as keys)
    and converts it to a list of (word, tag) tuples, padded with sentence
    begin and end tags)"""

    tokens = []
    tags = []
    for sent in partition['sentences']:
        tokens.append("<S>")
        for tok in sent:
            tokens.append(tok)
        tokens.append("</S>")
    
    for sent in partition['tags']:
        tags.append("O")
        for tag in sent:
            tags.append(tag)
        tags.append("O")
    
    return list(zip(tokens, tags))

def experiment_011_naive_bayes_unigram():
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    
    trainfeat = [({"word": tok}, lab) for (tok, lab) in train]
    testfeat  = [({"word": tok}, lab) for (tok, lab) in test]
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)
    train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    train_toks, train_true = zip(*train)
    test_toks, test_true = zip(*test)
    # considering inspect.currentframe().f_code.co_name as a way to
    # keep track of experiments, when running a battery of experiments
    results = conlleval(test_toks, test_true, test_pred)
    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_011_maxent_unigram():
    
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    
    trainfeat = [({"word": tok}, lab) for (tok, lab) in train]
    testfeat  = [({"word": tok}, lab) for (tok, lab) in test]
    classifier = nltk.MaxentClassifier.train(trainfeat)
    train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    train_toks, train_true = zip(*train)
    test_toks, test_true = zip(*test)
    # considering inspect.currentframe().f_code.co_name as a way to
    # keep track of experiments, when running a battery of experiments
    results = conlleval(test_toks, test_true, test_pred)
    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_012_naive_bayes_pseudo_bigram_dishonest():
    """this is dishonest because it uses the true previous tags: it
    should use the predicted previous tags

    """
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    testfeat = list(featurize_wordandtag_pseudo_bigram(test))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_012_naive_bayes_pseudo_bigram_honest():
    """uses the classify argument for the feature extractor so that tag
    related features are based on predictions

    """
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #util.conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)

    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_013_maxent_pseudo_bigram():
    """this is a maximum entropy model that uses the current and
    preceding words as separate features (not joint)"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_pseudo_bigram(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_pseudo_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]

    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_014_naivebayes_bigram():
    """ this is a bigram model using proper bigrams"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_bigram(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_014_maxent_bigram():
    """ this is a maxent model using proper bigrams"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_bigram(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_bigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def featurize_wordandtag_trigram(tokens, classify=False):
    """trigrams and various permutations of 3 word/2 tag preceding
    context

    input is a list/iterable of tokens, output/generator is list of dictionary features, like
    [{word_nm1: the, word_n: dog}]
    if tok == <s>, word_nm1 = "</s>" (padding)

    """
    prev_tok = ["</S>","</S>"] # end of sentence marker
    prev_lab = ["O", "O"]
    for tok, lab in tokens:
        feature_dict = {}
        # previous words
        feature_dict["word_n"] = tok
        feature_dict["word_n-1"] = prev_tok[-1]
        feature_dict["word_n-2"] = prev_tok[-2]        
        feature_dict["word_n-1,word_n"] = prev_tok[-1] + ","  + tok
        feature_dict["word_n-2,word_n-1,word_n"] = prev_tok[-2] + ","  + prev_tok[-1] + "," + tok
        
        # prev labels
        feature_dict["lab_n-1"] = prev_lab[-1]
        feature_dict["lab_n-2"] = prev_lab[-2]
        feature_dict["lab_n-2,lab_n-1"] = prev_lab[-2] + "," + prev_lab[-1]
        
        #combined words and labels
        # word_n plus one tag
        feature_dict["lab_n-1,word_n"] = prev_lab[-1] + ","  + tok
        feature_dict["lab_n-2,word_n"] = prev_lab[-2] + ","  + tok
        # word_n-1 plus one tag
        feature_dict["lab_n-1,word_n-1"] = prev_lab[-1] + ","  + prev_tok[-1]
        feature_dict["lab_n-2,word_n-1"] = prev_lab[-2] + ","  + prev_tok[-1]
        # word_n-2 plus one tag
        feature_dict["lab_n-1,word_n-2"] = prev_lab[-1] + ","  + prev_tok[-2]
        feature_dict["lab_n-1,word_n-2"] = prev_lab[-1] + ","  + prev_tok[-2]
        # word_n plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n"] = prev_lab[-2] + "," + prev_lab[-1] + ","  + tok
        # word_n-1 plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n-1"] = prev_lab[-2] + "," + prev_lab[-1] + ","  + prev_tok[-1]
        # word_n-2 plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n-1"] = prev_lab[-2] + "," + prev_lab[-1] + ","  + prev_tok[-2]
        # word_n and n-1 plus one tag
        feature_dict["lab_n-1,word_n-1,word_n"] = prev_lab[-1] + "," + prev_tok[-1] + ","  + tok
        feature_dict["lab_n-2,word_n-1,word_n"] = prev_lab[-2] + "," + prev_tok[-1] + ","  + tok
        # word_n and n-2 plus one tag
        feature_dict["lab_n-1,word_n-1,word_n"] = prev_lab[-1] + "," + prev_tok[-2] + ","  + tok
        feature_dict["lab_n-2,word_n-1,word_n"] = prev_lab[-2] + "," + prev_tok[-2] + ","  + tok
        # word_n-1 and n-2 plus one tag
        feature_dict["lab_n-1,word_n-2,word_n-1"] = prev_lab[-1] + "," + prev_tok[-2] + "," + prev_tok[-1]
        feature_dict["lab_n-2,word_n-2,word_n-1"] = prev_lab[-2] + "," + prev_tok[-2] + "," + prev_tok[-1]
        # word_n and n-1 plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n-1,word_n"] = prev_lab[-2] + "," + prev_lab[-1] + "," + prev_tok[-1] + ","  + tok
        # word_n and n-2 plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n-2,word_n"] = prev_lab[-2] + "," + prev_lab[-1] + "," + prev_tok[-2] + ","  + tok
        # word_n-1 and n-2 plus two tags
        feature_dict["lab_n-2,lab_n-1,word_n-2,word_n-1"] = prev_lab[-2] + "," + prev_lab[-1] + "," + prev_tok[-2] + ","  + prev_tok[-1]
        # all
        feature_dict["lab_n-2,lab_n-1,word_n-2,word_n-1,word_n"] = prev_lab[-2] + ",", prev_lab[-1] + "," + prev_tok[-2] + "," + prev_tok[-1] + "," + tok
        prev_tok[-2] = prev_tok[-1]
        prev_tok[-1] = tok
        if classify: # this is the part that makes it honest fair, see below
            lab = classify(feature_dict)
        prev_lab[-2] = prev_lab[-1]
        prev_lab[-1] = lab
        yield feature_dict, lab

def experiment_016_naivebayes_trigram():
    """ this is a naive bayes model using trigrams"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_trigram(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_trigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_016_maxent_trigram():
    """ this is a maxent model using proper bigrams"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_trigram(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_trigram(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def featurize_wordandtag_pseudo_bigram_backward(tokens, classify=False):
    """input is a list/iterable of tokens, output/generator is list of dictionary features, like
    [{word_nm1: the, word_n: dog}]
    if tok == <s>, word_nm1 = "</s>" (padding)

    it's called "psuedo" because it doesn't have true joint features,
    just multiple features including previous word

    """
    next_tok = "<S>" # start of sentence marker
    next_lab = "O"
    output = []
    for tok, lab in reversed(tokens):
        feature_dict = {}
        feature_dict["word_n"] = tok
        feature_dict["word_n+1"] = next_tok
        feature_dict["lab_n+1"] = next_lab
        next_tok = tok
        if classify: # this is the part that makes it honest fair, see below
            lab = classify(feature_dict)
        next_lab = lab
        output.append((feature_dict, lab))
    return(reversed(output))

def experiment_022_naive_bayes_pseudo_bigram_dishonest_backward():
    """ decode backward -- might help with head final language
        
        this is dishonest because it uses the true n+1 tags: it
    should use the predicted n+1 tags

    """
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)

    trainfeat = list(featurize_wordandtag_pseudo_bigram_backward(train))
    testfeat = list(featurize_wordandtag_pseudo_bigram_backward(test))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    test_pred = list(map(classifier.classify, [tok for tok, lab in testfeat]))
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_022_naive_bayes_pseudo_bigram_honest_backward():
    """uses the classify argument for the feature extractor so that tag
    related features are based on predictions

    """
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_pseudo_bigram_backward(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #util.conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)

    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def experiment_023_maxent_pseudo_bigram_backward():
    """this is a maximum entropy model that uses the current and
    preceding words as separate features (not joint)"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_pseudo_bigram_backward(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_pseudo_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_pseudo_bigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]

    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results

def featurize_wordandtag_bigram_backward(tokens, classify=False):
    """ backwards features
    
    actually the previous was not really bigram, only two word context
    here's a better bigram using joint features

    """
    next_tok = "<S>" # end of sentence marker
    next_lab = "O"
    output = []
    for tok, lab in reversed(tokens):
        feature_dict = {}
        feature_dict["word_n"] = tok
        feature_dict["word_n+1"] = next_tok
        feature_dict["word_n+1,word_n"] = next_tok + ","  + tok
        feature_dict["lab_n+1"] = next_lab
        feature_dict["lab_n+1,word_n"] = next_lab + ","  + tok
        feature_dict["lab_n+1,word_n+1"] = next_lab + ","  + next_tok
        feature_dict["lab_n+1,word_n+1,word_n"] = next_lab + ","  + next_tok + "," + tok
        next_tok = tok
        if classify: # this is the part that makes it honest fair, see below
            lab = classify(feature_dict)
        next_lab = lab
        output.append((feature_dict, lab))
    return(reversed(output))

def experiment_024_naivebayes_bigram_backward():
    """ this is a maxent model using proper bigrams, backward decoding"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_bigram_backward(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_bigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results
    

def experiment_024_maxent_bigram_backward():
    """ this is a maxent model using proper bigrams, backward decoding"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_bigram_backward(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_bigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results
def featurize_wordandtag_trigram_backward(tokens, classify=False):
    """ backwards trigram features: various permutations of 3 word/2 tag following 
    context

    input is a list/iterable of tokens, output/generator is list of dictionary features, like
    [{word_nm1: the, word_n: dog}]
    if tok == <s>, word_nm1 = "</s>" (padding)

    """
    next_tok = ["<S>","<S>"] # beginning of sentence marker
    next_lab = ["O", "O"]
    output = []
    for tok, lab in reversed(tokens):
        feature_dict = {}
        # following words
        feature_dict["word_n"] = tok
        feature_dict["word_n+1"] = next_tok[-1]
        feature_dict["word_n+2"] = next_tok[-2]        
        feature_dict["word_n+1,word_n"] = next_tok[-1] + ","  + tok
        feature_dict["word_n+2,word_n+1,word_n"] = next_tok[-2] + ","  + next_tok[-1] + "," + tok
    
        # following labels
        feature_dict["lab_n+1"] = next_lab[-1]
        feature_dict["lab_n+2"] = next_lab[-2]
        feature_dict["lab_n+2,lab_n+1"] = next_lab[-2] + "," + next_lab[-1]
        
        #combined words and labels
        # word_n plus one tag
        feature_dict["lab_n+1,word_n"] = next_lab[-1] + ","  + tok
        feature_dict["lab_n+2,word_n"] = next_lab[-2] + ","  + tok
        # word_n+1 plus one tag
        feature_dict["lab_n+1,word_n+1"] = next_lab[-1] + ","  + next_tok[-1]
        feature_dict["lab_n+2,word_n+1"] = next_lab[-2] + ","  + next_tok[-1]
        # word_n+2 plus one tag
        feature_dict["lab_n+1,word_n+2"] = next_lab[-1] + ","  + next_tok[-2]
        feature_dict["lab_n+1,word_n+2"] = next_lab[-1] + ","  + next_tok[-2]
        # word_n plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n"] = next_lab[-2] + "," + next_lab[-1] + ","  + tok
        # word_n+1 plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n+1"] = next_lab[-2] + "," + next_lab[-1] + ","  + next_tok[-1]
        # word_n+2 plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n+1"] = next_lab[-2] + "," + next_lab[-1] + ","  + next_tok[-2]
        # word_n and n+1 plus one tag
        feature_dict["lab_n+1,word_n+1,word_n"] = next_lab[-1] + "," + next_tok[-1] + ","  + tok
        feature_dict["lab_n+2,word_n+1,word_n"] = next_lab[-2] + "," + next_tok[-1] + ","  + tok
        # word_n and n+2 plus one tag
        feature_dict["lab_n+1,word_n+1,word_n"] = next_lab[-1] + "," + next_tok[-2] + ","  + tok
        feature_dict["lab_n+2,word_n+1,word_n"] = next_lab[-2] + "," + next_tok[-2] + ","  + tok
        # word_n+1 and n+2 plus one tag
        feature_dict["lab_n+1,word_n+2,word_n+1"] = next_lab[-1] + "," + next_tok[-2] + "," + next_tok[-1]
        feature_dict["lab_n+2,word_n+2,word_n+1"] = next_lab[-2] + "," + next_tok[-2] + "," + next_tok[-1]
        # word_n and n+1 plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n+1,word_n"] = next_lab[-2] + "," + next_lab[-1] + "," + next_tok[-1] + ","  + tok
        # word_n and n-2 plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n+2,word_n"] = next_lab[-2] + "," + next_lab[-1] + "," + next_tok[-2] + ","  + tok
        # word_n-1 and n-2 plus two tags
        feature_dict["lab_n+2,lab_n+1,word_n+2,word_n+1"] = next_lab[-2] + "," + next_lab[-1] + "," + next_tok[-2] + ","  + next_tok[-1]
        # all
        feature_dict["lab_n+2,lab_n+1,word_n+2,word_n+1,word_n"] = next_lab[-2] + ",", next_lab[-1] + "," + next_tok[-2] + "," + next_tok[-1] + "," + tok
        
        next_tok[-2] = next_tok[-1]
        next_tok[-1] = tok
        if classify: # this is the part that makes it honest fair
            lab = classify(feature_dict)
        next_lab[-2] = next_lab[-1]
        next_lab[-1] = lab
        output.append((feature_dict, lab))
    return(reversed(output))

def experiment_025_naivebayes_trigram_backward():
    """ this is a naive bayes model using trigrams of backward context"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_trigram_backward(train))
    classifier = nltk.NaiveBayesClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_trigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results
    
def experiment_025_maxent_trigram_backward():
    """ this is a maxent model using trigrams of backward context"""
    train_nerda, dev_nerda, test_nerda = make_nerda_train_dev_test(format_fn=nerda_format_just_iob)
    train = to_padded_tok_tag_tuples_from_nerda(train_nerda)
    #dev_tup = to_padded_tok_tag_tuples_from_nerda(dev_nerda)
    test = to_padded_tok_tag_tuples_from_nerda(test_nerda)
    

    trainfeat = list(featurize_wordandtag_trigram_backward(train))
    classifier = nltk.MaxentClassifier.train(trainfeat)

    # this way predicts over-optimistically because the preceding tags/labels are known
    #train_pred = list(map(classifier.classify, [tok for tok, lab in trainfeat]))
    #train_toks, train_true = zip(*train)
    #util.conlleval(train_toks, train_true, train_pred)

    # this way should be more fair/honest
    #train_pred = list(featurize_wordandtag_bigram(train, classify=classifier.classify))
    #conlleval(train_toks, train_true, [pred for _, pred in train_pred])

    # fair/honest for test
    testfeat_pred = list(featurize_wordandtag_trigram_backward(test, classify=classifier.classify))
    test_pred = [pred for _, pred in testfeat_pred]
    test_toks, test_true = zip(*test)
    results = conlleval(test_toks, test_true, test_pred)

    experiment_name = inspect.currentframe().f_code.co_name
    results["experiment_name"] = experiment_name
    return results
def main():
    """ runs all the experiment functions """
    # create table to store results
    connection = sqlite3.connect("results.db")
    cursor = connection.cursor()
    cursor.execute('''
    create table if not exists conlleval
    (
      experiment_name text not null,
      num_tokens integer not null,
      num_phrases integer not null,
      num_found integer not null,
      num_correct integer not null,
      accuracy real not null,
      precision real not null,
      recall real not null,
      fb1 real not null,
      constraint conlleval_pk primary key (experiment_name)
    );
    ''')

    experiments = sorted([item for item in globals() if item.startswith("experiment")])
    for experiment in experiments:
        print(experiment)
        cursor.execute("select * from conlleval where experiment_name = ?", (experiment, ))
        results = cursor.fetchone()
        if results:
            print("results cached", results)
        else:
            results = globals()[experiment]()
            cursor.execute("""
            insert into conlleval
            (experiment_name,  num_tokens, num_phrases, num_found, num_correct, accuracy, precision, recall, fb1)
            values (
            :experiment_name,
            :num_tokens,
            :num_phrases,
            :num_found,
            :num_correct,
            :accuracy,
            :precision,
            :recall,
            :fb1
            );
            """, results)
            print(results)
            connection.commit()

if __name__ == "__main__":
    main()

