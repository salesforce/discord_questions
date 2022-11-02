from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
import torch, tqdm, datasets


class NLIAnswerEquivalence:
    def __init__(self, model_card="tals/albert-xlarge-vitaminc-mnli", mode="eonly"):
        self.model_card = model_card
        self.device = "cuda"
        self.mode = mode
        self.e_idx, self.c_idx = 0, 1
        if self.model_card == "roberta-large-mnli":
            self.e_idx, self.c_idx = 2, 0
        assert self.mode in ["eonly", "eminusc"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval().to(self.device)

    def compare_batch(self, paragraphs1, paragraphs2):
        batch_tokens = self.tokenizer.batch_encode_plus(list(zip(paragraphs1, paragraphs2)), padding=True, truncation=True, max_length=512, return_tensors="pt", truncation_strategy="only_first")

        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
        with torch.no_grad():
            model_outputs = self.model(**batch_tokens)

        batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
        batch_evids = batch_probs[:, self.e_idx]
        batch_conts = batch_probs[:, self.c_idx]
        if self.mode == "eminusc":
            scores = (1.0 + batch_evids - batch_conts) / 2.0
        elif self.mode == "eonly":
            scores = batch_evids
        return {"scores": scores.tolist()}

    def compare(self, question, paragraphs1, paragraphs2, batch_size=32, progress=False):
        scores = []
        ite = range(0, len(paragraphs1), batch_size)
        if progress and len(ite) > 1:
            ite = tqdm.tqdm(ite)

        for i in ite:
            p1s = [p1["paragraph"] for p1 in paragraphs1[i:i + batch_size]]
            p2s = [p2["paragraph"] for p2 in paragraphs2[i:i + batch_size]]
            scores += self.compare_batch(p1s, p2s)["scores"]
        return {"scores": scores}

    def score(self, questions, answers1, answers2, contexts1, contexts2, batch_size=32, progress=False):
        N = len(answers1)
        scores = []
        ite = range(0, N, batch_size)
        if progress and len(ite) > 1:
            ite = tqdm.tqdm(ite)
        for i in ite:
            batch_p1 = contexts1[i:i+batch_size]
            batch_p2 = contexts2[i:i+batch_size]
            batch_a1 = answers1[i:i+batch_size]
            batch_a2 = answers2[i:i+batch_size]

            batch_scores =  self.compare_batch(batch_p1, batch_p2)["scores"]
            for idx, a1, a2 in zip(range(len(batch_a1)), batch_a1, batch_a2):
                if a1 == a2:
                    batch_scores[idx] = 1.0
            scores.extend(batch_scores)
        return {"scores": scores}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class STS_AE:
    def __init__(self, model_card="sentence-transformers/stsb-bert-base"):
        self.model_card = model_card
        self.device = "cuda"
        self.model = AutoModel.from_pretrained(self.model_card).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)

    def encode(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def compare_batch(self, answers1, answers2):
        embs1 = self.encode(answers1)
        embs2 = self.encode(answers2)

        scores = torch.nn.functional.cosine_similarity(embs1, embs2, dim=1)
        return {"scores": scores.tolist()}


    def score(self, questions, answers1, answers2, contexts1, contexts2, batch_size=32, progress=False):
        N = len(answers1)
        scores = []
        ite = range(0, N, batch_size)
        if progress and len(ite) > 1:
            ite = tqdm.tqdm(ite)
        for i in ite:
            batch_a1 = answers1[i:i+batch_size]
            batch_a2 = answers2[i:i+batch_size]

            batch_scores = self.compare_batch(batch_a1, batch_a2)["scores"]
            for idx, a1, a2 in zip(range(len(batch_a1)), batch_a1, batch_a2):
                if a1 == a2:
                    batch_scores[idx] = 1.0
            scores.extend(batch_scores)
        return {"scores": scores}
