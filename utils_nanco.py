import itertools, json

def load_nanco_groups(q_includes=None):
    with open("nanco_dataset.json", "r") as f:
        dataset = json.loads(f.read())
    if q_includes is not None:
        dataset = [d for d in dataset if d["answers"][0]["qid"] in q_includes]
    return dataset

def load_nanco_pairs(q_includes=None):
    cluster_dataset = load_nanco_groups(q_includes=q_includes)
    dataset_pairs = []
    for qdata in cluster_dataset:
        adata = qdata["answers"]
        for a1, a2 in itertools.combinations(adata, 2):
            label = 1 if a1["cluster_global"] == a2["cluster_global"] else 0
            dataset_pairs.append({"qid": a1["qid"], "question": qdata["question"], "label": label,
                                "answer1": a1["Answer"], "paragraph1": a1["Paragraph"],
                                "answer2": a2["Answer"], "paragraph2": a2["Paragraph"]})
    return dataset_pairs

if __name__ == "__main__":
    from collections import Counter

    dataset_pairs = load_nanco_pairs()
    print(len(dataset_pairs), Counter([d["label"] for d in dataset_pairs]))
    print(dataset_pairs[0])
