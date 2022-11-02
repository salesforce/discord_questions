from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils_questions import is_valid_question, fast_fuzzy_string_set
from model_consolidation import ConsolidationModel
from model_qa import QAModel
import random

class DiscordQuestions:
    def __init__(self, qg_card="Salesforce/discord_qg", qa_card="Salesforce/discord_qa", conso_card="Salesforce/qa_consolidation", device="cuda"):
        self.device = device

        # Step 1: QG
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_card)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_card).to(self.device)

        # Step 2: QA
        self.qa_model = QAModel(model_card=qa_card, device=self.device)

        # Step 3: QA Consolidation
        self.conso_model = ConsolidationModel(conso_card, model_file=None, device=self.device)
    
    def document_processor(self, documents, doc_key="content", paragraph_key="paragraphs"):
        for doc in documents:
            doc[paragraph_key] = doc[doc_key].split("\n\n")

    def generate_questions(self, paragraphs):
        N = len(paragraphs)
        all_questions = []
        for start_word in ["How", "Why", "What", "Who", "When"]:
            encoder_ids = self.qg_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
            encoder_ids = {k: t.to(self.device) for k, t in encoder_ids.items()}
            decoder_input_ids = self.qg_tokenizer.batch_encode_plus([start_word] * N, add_special_tokens=True, return_tensors="pt")["input_ids"][:, :-1].to(self.device)
            model_output = self.qg_model.generate(**encoder_ids, decoder_input_ids=decoder_input_ids, max_length=20)
            generated_questions = self.qg_tokenizer.batch_decode(model_output, skip_special_tokens=True)
            all_questions += [q for q in generated_questions if is_valid_question(q)]
        return all_questions

    def populate_questions(self, documents, paragraph_key="paragraphs", question_key="questions", printing=True):
        for doc in documents:
            doc[question_key] = self.generate_questions(doc[paragraph_key])

    def deduplicate_questions(self, documents, doc_id="id", question_key="questions", max_questions=None, printing=True):
        all_questions = [{"question": q, "doc_id": doc[doc_id], "answers": []} for doc in documents for q in doc[question_key]]
        N_before = len(all_questions)
        questions = fast_fuzzy_string_set(all_questions, simi=80, with_key="question")
        if max_questions is not None:
            random.shuffle(questions)
            questions = questions[:max_questions]
        N_after = len(questions)
        if printing:
            print("[Deduplication %d -> %d questions]" % (N_before, N_after))
        return questions

    def run_qa(self, documents, questions, doc_id="id", paragraph_key="paragraphs", qa_batch_size=512, printing=True):
        id2doc = {}
        for doc in documents:
            doc["answers"] = []
            id2doc[doc[doc_id]] = doc

        paragraphs = [{"doc_id": doc[doc_id], "paragraph": p, "pidx": pidx, "pid": "%s_%d" % (doc[doc_id], pidx), "answers": []} for doc in documents for pidx, p in enumerate(doc[paragraph_key])]
        
        pq_pairs = []
        for i, question in enumerate(questions):
            for j, paragraph in enumerate(paragraphs):
                pq_pairs.append({"question": question['question'], "paragraph": paragraph["paragraph"], "pidx": j, "qidx": i})

        answers = self.qa_model.predict([pq["question"] for pq in pq_pairs], [pq["paragraph"] for pq in pq_pairs], max_batch_size=qa_batch_size, progress=printing)
        for pq, a in zip(pq_pairs, answers):
            p, q = paragraphs[pq["pidx"]], questions[pq["qidx"]]
            doc = id2doc[p["doc_id"]]
            if a["answer"] != "no answer":
                q["answers"].append({"answer": a["answer"], "doc_id": p["doc_id"], "pidx": p["pidx"], "pid": p["pid"]})
                doc["answers"].append({"question": q["question"], "answer": a["answer"], "pidx": p["pidx"]})
        return documents, questions

    def consolidate_questions(self, questions, min_answers=3, printing=True):
        questions = sorted(questions, key=lambda q: -len(q["answers"]))
        final_questions = []
        for question in questions:
            if len(question["answers"]) < min_answers:
                break
            question["answer_groups"] = self.conso_model.consolidate(question["question"], question["answers"])
            final_questions.append(question)
        return final_questions

    def select_discord_questions(self, documents, candidate_questions, printing=True):
        final_dqs = []
        min_answers = int(0.3*len(documents))
        for q in candidate_questions:
            if len(q["answers"]) < min_answers:
                continue # Specific question (not broad enough)
            largest_group = max([len(group) for group in q["answer_groups"]])
            if largest_group > 0.5 * len(q["answers"]):
                continue # Consensus question
            q["pids"] = set([ans["pid"] for ans in q["answers"]])
            if any(len(q2["pids"] & q["pids"]) >= 0.8*len(q["pids"]) for q2 in final_dqs):
                continue # It is too similar to an already selected question
            final_dqs.append(q)
        return final_dqs

    def run_pipeline(self, documents, max_questions=None, printing=True):
        required_keys = ["id", "content"]
        assert all(k in documents[0] for k in required_keys)
        if printing:
            print("[Step 0] Processing %d documents" % (len(documents)))
        self.document_processor(documents)
        
        if printing:
            print("[Step 1] Generating Question Candidates")
        self.populate_questions(documents, printing=True)
        questions = self.deduplicate_questions(documents, max_questions=max_questions, printing=True)
        
        if printing:
            print("[Step 2] Finding Answers to Candidates (longest step)")
        documents, questions = self.run_qa(documents, questions, doc_id="id", paragraph_key="paragraphs", printing=True)
        
        if printing:
            print("[Step 3] Consolidating Answers")
        questions = self.consolidate_questions(questions, printing=True)

        # Selection
        discord_questions = self.select_discord_questions(documents, questions, printing=True)
        return discord_questions
