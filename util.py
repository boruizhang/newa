"""
This is a file to import, with helper functions.

It also has the experiments (copied from notebooks)
"""
import inspect
import os
import re
import subprocess
import sys
import nltk

def conlleval(toks, trues, preds):
    """
    takes lists of tokens, true labels, and predictions

    returns num_tokens, num_phrases, num_found, num_correct, accuracy, precision, recall, fb1

    This fuction runs the conlleval.py script by opening a subprocess and writing/piping
    three columns of tab separated values: token \t true_label \t predicted value
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
    return match.groups()

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
    return conlleval(test_toks, test_true, test_pred), inspect.currentframe().f_code.co_name

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
    return conlleval(test_toks, test_true, test_pred)


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
    test_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_toks, test_true = zip(*test)
    return conlleval(test_toks, test_true, [pred for _, pred in test_pred])

def experiment_003_maxent_pseudo_bigram():
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
    test_pred = list(featurize_wordandtag_pseudo_bigram(test, classify=classifier.classify))
    test_toks, test_true = zip(*test)
    return conlleval(test_toks, test_true, [pred for _, pred in test_pred])


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
    test_pred = list(featurize_wordandtag_bigram(test, classify=classifier.classify))
    test_toks, test_true = zip(*test)
    return conlleval(test_toks, test_true, [pred for _, pred in test_pred])

if __name__ == "__main__":
    """ runs all the experiment functions """
    experiments = sorted([item for item in dir() if item.startswith("experiment")])
    for experiment in experiments:
        print(experiment)
        globals()[experiment]()
