from collections import Counter
from fuzzywuzzy import fuzz
import nltk, re, tqdm

def is_valid_question(q):
    bad_words = set(["name", "their", "they", "you", "she", "he", "we", "ad", "blocker"])
    common_bad_qs = set([q.lower() for q in ["What can be accessed?", "Who has been asked to do this?", "What was the result?", "When was the conversation?", "How long has it been?", "When will it be available?", "How does it work?", "Why did this happen?", "Which must be reported?", "What will continue?", "Who was on the call?"]])
    patterns = [r"(when|where|how) (can|does|will|did) (it|this|that)( all)? happen\?",
                r"(when|where) (was|is) the (event|meeting)\?",
                r"(what) (did|does|must|will) (he|she|the mayor|the president|the office|the advisor|the minister|officials|the official|the us|the company|the ministry|the team) (say|do)\?",
                r"(which) other .*",
                r"(what|which) country .*",
                r"(what|which) president .*"]

    q = q.lower()
    words = nltk.tokenize.word_tokenize(q)
    word_counts = Counter(words)
    bad_starts = set(["which company", "which person", "which year", "what number", "who is the person", "what company", "who said", "what is the date", "what is the range", "what was the length", "what is the length"])
    w_words = set(["what", "who", "when", "where", "why", "how", "which", "whom", "whose", "is", "are"])

    for pattern in patterns:
        if re.search(pattern, q):
            return False

    if q in common_bad_qs:
        return False
    if len(words) < 4 or len(words) > 12:
        return False
    if ("what did" in q or "what was" in q) and len(words) >= 7:
        return False
    if any(w in bad_words for w in words):
        return False
    if any(bs in q for bs in bad_starts):
        return False
    if "twitter account" in q:
        return False
    if "stand for" in q:
        return False
    if words[0] not in w_words:
        return False
    if sum([word_counts[w] for w in w_words]) >= 2:
        return False

    if q.count("?") != 1:
        return False
    return True

def fast_fuzz_match(S, haystack, simi=80):
    max_rat = 2 - simi / 100
    min_rat = simi / 100
    N = len(S)
    for hay in haystack:
        N2 = len(hay)
        if N2 > max_rat*N or N2 < min_rat*N:
            continue
        elif fuzz.ratio(S, hay) > simi:
            return True
    return False

def fast_fuzzy_string_set(Xs, with_key=None, simi=80, progress=False):
    outputs, build_up = [], []
    Xs = Xs if not progress else tqdm.tqdm(Xs)
    for x in Xs:
        S = x if with_key is None else x[with_key]
        if not fast_fuzz_match(S, build_up, simi=simi):
            outputs.append(x)
            build_up.append(S)
    return outputs
