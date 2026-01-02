from collections import defaultdict
import re
import string
import math

def tokenize(text: str):
    """
    Lowercase and split into tokens where punctuation marks become their own tokens.
    Example: "hello, world." -> ["hello", ",", "world", "."]
    """
    text = text.lower()
    # words OR one of these punctuations as tokens
    return re.findall(r"\w+|[{}]".format(re.escape(string.punctuation)), text)

def bigram_mle_prob(prev_word: str, next_word: str, unigram_counts, bigram_counts) -> float:
    denom = unigram_counts[prev_word]
    if denom == 0:
        return 0.0
    return bigram_counts[(prev_word, next_word)] / denom

def trigram_mle_prob(w1: str, w2: str, w3: str, bigram_context_counts, trigram_counts) -> float:
    denom = bigram_context_counts[(w1, w2)]
    if denom == 0:
        return 0.0
    return trigram_counts[(w1, w2, w3)] / denom

def predict_next_bigram(context_word: str, top_k=5, unigram_counts=None, bigram_counts=None):
    # all candidates that ever follow context_word
    candidates = [w2 for (w1, w2) in bigram_counts.keys() if w1 == context_word]
    # unique candidates
    candidates = sorted(set(candidates))
    scored = [(w, bigram_mle_prob(context_word, w, unigram_counts, bigram_counts)) for w in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def predict_next_trigram(w1: str, w2: str, top_k=5, bigram_context_counts=None, trigram_counts=None):
    candidates = [w3 for (a, b, w3) in trigram_counts.keys() if a == w1 and b == w2]
    candidates = sorted(set(candidates))
    scored = [(w, trigram_mle_prob(w1, w2, w, bigram_context_counts, trigram_counts)) for w in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

#===============================
# Exercise 2
#===============================

def sentence_prob_no_smoothing(sentence_tokens, unigram, bigram):
    prob = 1.0
    for w1, w2 in zip(sentence_tokens[:-1], sentence_tokens[1:]):
        prob *= bigram[(w1, w2)] / unigram[w1] if unigram[w1] > 0 else 0
    return prob

def sentence_prob_laplace(sentence_tokens, unigram, bigram, V):
    prob = 1.0
    for w1, w2 in zip(sentence_tokens[:-1], sentence_tokens[1:]):
        prob *= (bigram[(w1, w2)] + 1) / (unigram[w1] + V)
    return prob


#===============================
# Exercise 3
#===============================

def bigram_mle(w1, w2, unigram, bigram):
    if unigram[w1] == 0:
        return 0
    return bigram[(w1, w2)] / unigram[w1]

def bigram_laplace(w1, w2, unigram, bigram, V):
    return (bigram[(w1, w2)] + 1) / (unigram[w1] + V)

def perplexity(test_tokens, prob_func):
    log_prob_sum = 0
    N = 0

    for w1, w2 in zip(test_tokens[:-1], test_tokens[1:]):
        p = prob_func(w1, w2)
        if p == 0:
            return float("inf")  # zero probability → infinite perplexity
        log_prob_sum += math.log(p)
        N += 1

    return math.exp(-log_prob_sum / N)