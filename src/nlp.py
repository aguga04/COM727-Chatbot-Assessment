"""Text preprocessing for the intent classifier.

A typed or clicked question is reduced to a bag of words vector by the same
sequence of steps in training and at run time: tokenise, lowercase, discard
punctuation, drop uninformative words, lemmatise, then count against a fixed
vocabulary.

The vocabulary is built from the training patterns and reused afterwards. This
is the same fit on train and transform both rule that governs the scaler in the
income pipeline, applied to text. A word that never appeared in any pattern has
no position in the vector and is discarded, which is why the coverage of the
patterns in ``data/intents.json`` determines the ceiling on what the classifier
can recognise.

Two deliberate departures from the textbook pipeline are documented below: the
tokeniser is regular expression based rather than the sentence trained one, and
the stopword list retains interrogatives. Both are measured choices rather than
shortcuts.
"""

from functools import lru_cache

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

REQUIRED_CORPORA = {
    "wordnet": "corpora/wordnet.zip",
    "omw-1.4": "corpora/omw-1.4.zip",
    "stopwords": "corpora/stopwords",
}

# Words that carry no intent signal are dropped, but interrogatives and modals
# are kept. Every question in the intent set begins with one, so discarding them
# removes the clearest signal available. Measured on the display strings, the
# full stopword list costs both accuracy and confidence.
RETAINED_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "do", "does", "did", "doing", "is", "are", "was", "were", "be", "been",
    "can", "could", "will", "would", "should", "shall", "may", "might", "must",
    "not", "no", "nor", "against", "more", "most", "than", "same", "own",
    "up", "down", "out", "off", "over", "under", "again", "just", "only",
}

_tokeniser = RegexpTokenizer(r"[a-z]+")


@lru_cache(maxsize=1)
def ensure_corpora():
    """Download the NLTK data files the lemmatiser and stopword list need.

    Called lazily rather than at import time, and cached so the check runs once
    per process. The packaged library does not ship its corpora, so a deployment
    that builds successfully will still fail on the first message unless this
    runs. Missing corpora are fetched quietly; present ones are left alone.
    """
    for package, location in REQUIRED_CORPORA.items():
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(package, quiet=True)

    return True


@lru_cache(maxsize=1)
def _lemmatiser():
    """Return the shared lemmatiser, downloading its corpora on first use."""
    ensure_corpora()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def stopwords():
    """Return the stopword set actually used, with interrogatives retained."""
    ensure_corpora()
    from nltk.corpus import stopwords as nltk_stopwords

    return set(nltk_stopwords.words("english")) - RETAINED_STOPWORDS


@lru_cache(maxsize=4096)
def lemmatise(word):
    """Reduce a word to its base form, trying verb then noun.

    The lemmatiser assumes a noun unless told otherwise, so plurals reduce but
    verb inflections do not. Passing a word through twice catches both, and the
    order matters: noun first turns does into doe and was into wa, because both
    are read as plural nouns before anything else is tried. Verb first resolves
    them to do and be, which matches did and is, and it leaves genuine plurals
    such as records and boundaries unharmed. Results are cached because the same
    handful of words recur constantly.
    """
    lemmatiser = _lemmatiser()
    return lemmatiser.lemmatize(lemmatiser.lemmatize(word, pos="v"), pos="n")


def preprocess(text):
    """Turn a sentence into its list of processed tokens.

    Lowercasing and punctuation removal happen together in the tokeniser, which
    only matches runs of letters. Digits are discarded, which is safe here
    because no intent depends on a number.
    """
    tokens = _tokeniser.tokenize(text.lower())
    discard = stopwords()
    return [lemmatise(t) for t in tokens if t not in discard]


def build_vocabulary(documents):
    """Return the sorted vocabulary of a collection of sentences.

    Called once during training. The order is fixed by sorting so that a saved
    vocabulary always produces vectors in the same column order.
    """
    words = set()
    for document in documents:
        words.update(preprocess(document))
    return sorted(words)


def vectorise(text, vocabulary, binary=True):
    """Return the bag of words vector for one sentence.

    Binary presence is the default rather than raw counts. Questions are short
    enough that a word rarely repeats, and presence measured slightly better on
    this intent set than counts did.
    """
    index = {word: position for position, word in enumerate(vocabulary)}
    vector = np.zeros(len(vocabulary), dtype=np.float64)

    for token in preprocess(text):
        position = index.get(token)
        if position is not None:
            vector[position] = 1.0 if binary else vector[position] + 1.0

    return vector


def vectorise_many(documents, vocabulary, binary=True):
    """Return the bag of words matrix for a collection of sentences."""
    return np.vstack([vectorise(d, vocabulary, binary) for d in documents])


def coverage(text, vocabulary):
    """Return the share of a sentence's tokens that the vocabulary recognises.

    Useful when diagnosing a message the classifier handled badly: a low value
    means the wording fell outside everything the patterns covered.
    """
    tokens = preprocess(text)
    if not tokens:
        return 0.0
    known = set(vocabulary)
    return sum(t in known for t in tokens) / len(tokens)


if __name__ == "__main__":
    ensure_corpora()

    variants = [
        "How much did age matter?",
        "did age matter for this person",
        "Does their age carry weight?",
        "is age important here",
    ]

    print("Same question, four phrasings")
    for phrase in variants:
        print(f"  {phrase:<38} {preprocess(phrase)}")

    sets = [set(preprocess(v)) for v in variants]
    shared = set.intersection(*sets)
    print(f"\n  shared by all four: {sorted(shared)}")

    unrelated = set(preprocess("Where does the data come from?"))
    print(f"  shared with an unrelated question: {sorted(sets[0] & unrelated)}")

    print("\nLemmatisation and stopwords are doing work")
    for pair in [("matters", "matter"), ("working", "work"),
                 ("records", "record"), ("counted", "count")]:
        print(f"  {pair[0]:<10} -> {lemmatise(pair[0]):<10} expected {pair[1]}")

    print(f"\n  stopwords active: {len(stopwords())} words")
    print(f"  interrogatives retained: {sorted(RETAINED_STOPWORDS & set(['what', 'how', 'which', 'does']))}")
    print(f"  'the a of to' -> {preprocess('the a of to')}")