"""
This is a file to import, with helper functions.
"""
import inspect
import re
import subprocess

def conlleval(toks, trues, preds):
    """
    takes lists of tokens, true labels, and predictions

    returns num_tokens, num_phrases, num_found, num_correct, accuracy, precision, recall, fb1

    This fuction runs the conlleval.py script by opening a subprocess and writing/piping
    three columns of tab separated values: token \t true_label \t predicted value
    """
    p1 = subprocess.Popen(["cat"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    p2 = subprocess.Popen(["./conlleval.pl", "-d", "\t"], stdin=p1.stdout, stdout=subprocess.PIPE)
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

def experiment_001_naive_bayes_unigram():
    import nltk
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
