
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import tqdm, torch, os

def is_whitespace(c):
    return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F

MODELS_FOLDER =  os.environ["MODELS_FOLDER"]

def align_answer_text(text):
    if len(text) == 0:
        return text

    text = text.strip()

    if text[-1] in [".", ",", "!", "?", "-", ";", ":"]:
        return align_answer_text(text[:-1]) # , "%"
    if text[0] in ['"']:
        return align_answer_text(text[1:].strip())
    if text[-2:] == "'s":
        return align_answer_text(text[:-2].strip())
    if text[0] in ["'", "("] and text[-1] in ["'", ")"]:
        return align_answer_text(text[1:-1].strip())
    if text[0] == '"' and text[-1] == '"' and text.count('"') == 2:
        return align_answer_text(text[1:-1].strip())
    if text[-2:] == "s'":
        return align_answer_text(text[:-1]) # "Steve Jobs'" >> "Steve Jobs"

    if "US$" in text[:3]:
        return align_answer_text(text[2:])

    if text.count("(") + text.count(")") == 1:
        return align_answer_text(text.replace("(", "").replace(")", ""))

    if text.count('"') == 1:
        return align_answer_text(text.replace('"', ""))

    text = text.replace("[note", "").replace("[n", "").replace("[citation", "") # SQuAD specific
    return text

class QAModel:
    def __init__(self, model_card, max_seq_length=512, device="cuda", scorer_type="overlap", starter_model=None):
        assert scorer_type in ["overlap", "any_answer", "likelihood"], "scorer_type not recognized"

        self.device = device
        self.model_card = model_card
        self.max_seq_length = max_seq_length
        self.max_query_length = 48
        self.scorer_type = scorer_type

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_card).to(self.device)

        tok_args = {"add_prefix_space": True} if "roberta" in self.model_card else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_card, **tok_args)
        self.cls_token, self.sep_token = self.tokenizer.eos_token, self.tokenizer.bos_token
        if self.cls_token is None and self.sep_token is None:
            # EOS and BOS weren't set
            assert len(self.tokenizer.all_special_tokens) >= 2, "There doesn't seem to be enough special tokens in this vocabulary for CLS and SEP"
            self.cls_token = self.tokenizer.all_special_tokens[-1]
            self.sep_token = self.tokenizer.all_special_tokens[-2]

        if starter_model is not None:
            self.reload(starter_model)
        print("Model loaded")

    def reload(self, from_file):
        if not os.path.isfile(from_file):
            from_file = os.path.join(MODELS_FOLDER, from_file)
            if not os.path.isfile(from_file):
                raise FileNotFoundError("Model file %s not found in absolute location or MODELS_FOLDER" % from_file)

        print(self.model.load_state_dict(torch.load(from_file), strict=False))
        torch.cuda.empty_cache()

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def logits2text(self, features, start_position, end_position):
        try:
            mapping = features["token2char_idx"]
            char_start = mapping[start_position]
            if end_position+1 > max(mapping.keys()):
                char_end = len(features["paragraph"])
            else:
                char_end = mapping[end_position+1]
            return align_answer_text(features["paragraph"][char_start:char_end])
        except:
            # For now... should fix this for sure
            return "no answer"

    def tokenize_full(self, paragraph_text, question_text, answer_start=-1, orig_answer_text=None, pad=True):
        # answer_start < 0 --> is_impossible = true
        correct_is_beyond = False

        words, char_to_word_offset = [], []
        prev_is_whitespace = True

        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    words.append(c)
                else:
                    words[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(words) - 1)

        all_doc_tokens, tok_to_orig_index, orig_to_tok_index = [], [], []
        # Go from white-space separated words to real tokens for the model
        for i, word in enumerate(words):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(word)

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position, tok_end_position = -1, -1

        # The -3 accounts for [CLS], [SEP] and [SEP]
        query_tokens = self.tokenizer.tokenize(question_text)
        query_tokens = query_tokens[:self.max_query_length]

        max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
        all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
        tokens = [self.cls_token] + query_tokens + [self.sep_token]
        doc_offset = len(query_tokens) + 2
        token_to_orig_map = {doc_offset+i: char_to_word_offset.index(tok_to_orig_index[i]) for i in range(len(all_doc_tokens))} # This can be useful to recover from start_idx, end_idx to character indices in the original paragraph
        tokens += all_doc_tokens

        # SEP token
        tokens.append(self.sep_token)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        start_position, end_position = 0, 0 # Train the model to say there is no answer
        if answer_start > 0:
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_start]

            end_position = char_to_word_offset[answer_start + answer_length - 1]

            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(words) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            # Add the additional offset because of the question tokens & the special separators
            start_position = tok_start_position + doc_offset
            end_position = tok_end_position + doc_offset

        if start_position >= self.max_seq_length or end_position >= self.max_seq_length:
            # It's going to be cut out of the text, so it is infeasible
            start_position, end_position = 0, 0
            correct_is_beyond = True

        if pad:
            input_ids += [0] * (self.max_seq_length - len(input_ids))
            input_mask += [0] * (self.max_seq_length - len(input_mask))

        return {"input_ids": input_ids, "input_mask": input_mask,
        "start_position": start_position, "end_position": end_position,
        "token2char_idx": token_to_orig_map, # To go from token idx to raw text
        "char_to_word_offset": char_to_word_offset, "orig_to_tok_index": orig_to_tok_index, "doc_offset": doc_offset, "all_doc_tokens": all_doc_tokens, "words": words, # To go from raw text to token idx
        "paragraph": paragraph_text, "tokens": tokens, "correct_is_beyond": correct_is_beyond}

    def preprocess(self, questions, paragraphs):
        all_features = []
        input_ids, attention_mask = [], []
        for paragraph, question in zip(paragraphs, questions):
            features = self.tokenize_full(paragraph, question)
            input_ids.append(torch.LongTensor(features["input_ids"]))
            attention_mask.append(torch.LongTensor(features["input_mask"]))
            all_features.append(features)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(self.device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(self.device)

        return input_ids, attention_mask, all_features

    def predict_batch(self, questions, paragraphs):
        self.model.eval()
        outs = []

        input_ids, attention_mask, batch_features = self.preprocess(questions, paragraphs)
        with torch.no_grad():
            model_outs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = model_outs["start_logits"]
            end_logits = model_outs["end_logits"]

            for starts, ends, features in zip(start_logits, end_logits, batch_features):
                noans_score = (starts[0] + ends[0]).item()
                top_starts = (torch.argsort(starts, dim=0, descending=True)).tolist()[:5]
                top_ends = (torch.argsort(ends, dim=0, descending=True)).tolist()[:5]

                candidates = []
                for s_idx in top_starts:
                    for e_idx in top_ends:
                        if s_idx == 0 or e_idx == 0 or e_idx - s_idx < 0 or e_idx - s_idx > 30: # Importantly, e_idx == s_idx is possible for 1-word answers
                            continue
                        candidates.append({"s_idx": s_idx, "e_idx": e_idx, "score": (starts[s_idx] + ends[e_idx]).item()})

                candidates = sorted(candidates, key=lambda can: can["score"], reverse=True)

                out = {"answer": "no answer", "score": 0.0, "noans_score": noans_score, "features": features, "start_logits": starts, "end_logits": ends}
                noop_answers = ["no answer", "the", "and", "a", ""]
                if len(candidates) > 0 and candidates[0]["score"] > noans_score:
                    best_candidate = candidates[0]
                    start_index, end_index = best_candidate["s_idx"], best_candidate["e_idx"]
                    answer_text = self.logits2text(features, start_index, end_index)
                    if answer_text not in noop_answers:
                        out["answer"] = answer_text
                        out["score"] = best_candidate["score"]
                        out["best_s_idx"] = best_candidate["s_idx"]
                        out["best_e_idx"] = best_candidate["e_idx"]

                outs.append(out)
        return outs

    def predict(self, questions, paragraphs, max_batch_size=8, progress=False):
        iterator = range(0, len(questions), max_batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator)

        outputs = []
        for I in iterator:
            batch_outs = self.predict_batch(questions[I:(I+max_batch_size)], paragraphs[I:(I+max_batch_size)])
            outputs += batch_outs
        return outputs
