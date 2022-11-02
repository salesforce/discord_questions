from model_consolidation import ClusterBuilder, AEPairScorer
from collections import Counter
from model_qa import QAModel
import numpy as np, json

def load_discord_qg_eval_dataset(fn="/export/home/data/inqqg_eval_dataset_v1.jsonl"):
    dataset = []
    with open(fn) as f:
        for line in f:
            dataset.append(json.loads(line))
    for d in dataset:
        other_paragraphs = [dataset[e]["articles"][a]["paragraphs"][p] for e,a,p in d["other_paragraph_idx"]]
        d["other_paragraphs"] = other_paragraphs
    return dataset

def assign_label_and_score(data):
    target_answers = data["target_answers"]
    distractor_answers = data["distractor_answers"]
    clusters = data["clusters"]
    good_paragraphs = data["good_paragraphs"]
    other_paragraphs = data["other_paragraphs"]
    
    answer_lengths = [a["answer"].count(" ")+1 for a in target_answers]
    median_answer_length = np.median(answer_lengths)

    if len(distractor_answers) > 0:
        specificity = min((len(target_answers) / len(good_paragraphs)) / (len(distractor_answers) / len(other_paragraphs)), 5.0)
    else:
        specificity = 5.0

    if len(target_answers) == 0:
        score, label = 0, "unanswerable"
    elif specificity < 2.0:
        score, label = 0, "vague"
    else:
        num_clusters = len(clusters)
        if num_clusters >= 2 and median_answer_length > 3:
            score, label = 1, "analysis"
        else:
            score, label = 0, "factoid"
    data["label"] = label
    data["score"] = score
    return data

def compute_results(results):
    error_types = ["analysis", "factoid", "unanswerable", "vague"]
    pd_results = []
    model_names = results.keys()
    for model_name in model_names:
        avg_score = np.mean([s["score"] for s in results[model_name]])
        label_distrib = Counter([s["label"] for s in results[model_name]])
        N = len(results[model_name])
        row = {"QG Model": model_name, "Score": avg_score}
        row.update({"%%%s" % (l.capitalize()): 100*label_distrib[l]/N for l in error_types})
        pd_results.append(row)
    return pd_results

def generate_unique_hash(all_paragraphs):
    return "|||".join([p[:75] for p in all_paragraphs])

class InqQG_Eval:
    def __init__(self):
        self.qa = QAModel(model_card="roberta-large", starter_model="qa/roberta_large_bs12_new_newsqa_eloss_0.676.bin")
        self.qa.model.half()
        self.ae = AEPairScorer("/export/home/models/quip-hf/", model_file="/export/home/models/quip_ae_mocha_mae_0.5352.bin")
        self.clusterer = ClusterBuilder(self.ae, thresh=2.75)
        self.cache = {}

    def score_one(self, question, good_paragraphs, other_paragraphs):
        full_hash = question + " []" + generate_unique_hash(good_paragraphs + other_paragraphs)
        if full_hash not in self.cache:
            answers = self.qa.predict([question] * len(good_paragraphs), good_paragraphs)
            filtered_answers = [{"answer": a["answer"]} for a in answers if a["answer"] != "no answer"]

            distractor_answers = self.qa.predict([question] * len(other_paragraphs), other_paragraphs)
            distractor_answers = [{"answer": a["answer"]} for a in distractor_answers if a["answer"] != "no answer"]

            clusters = [] if len(filtered_answers) == 0 else self.clusterer.run_clusters(question, filtered_answers)
            self.cache[full_hash] = {"answers": filtered_answers, "distractor_answers": distractor_answers, "clusters": clusters}

        result = self.cache[full_hash]
        return assign_label_and_score({"question": question, "target_answers": result["answers"], "distractor_answers": result["distractor_answers"], "clusters": result["clusters"], "good_paragraphs": good_paragraphs, "other_paragraphs": other_paragraphs})
