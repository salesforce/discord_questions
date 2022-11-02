# Near-Negative Distinction

Code repository for the paper: [Discord Questions: A Computational Approach To Diversity Analysis in News Coverage](https://tingofurro.github.io/pdfs/emnlp2022_discord_questions.pdf) accepted as a Findings paper at EMNLP 2022.

<p align="center" style="width: 7500px;">
  <img width="350" height="372" style='vertical-align: middle;' src="https://tingofurro.github.io/images/emnlp2022_discord_questions_examples.png">
  <img width="350" height="208" style='vertical-align: middle;' src="https://tingofurro.github.io/images/emnlp2022_discord_questions_pipeline.png">
  <div>Left: Examples of discord questions and their answer groups for two news stories in July 2022.</div>
  <div>Right: A diagram of the Discord Questions pipeline, composed of a question generation, question answering and answer consolidation steps.
</p>

## Discord Questions Pipeline

The Discord Questions pipeline can take as input collection of text documents, and extract questions that best represent the "discord" (disagreement) in the colleciton.
To see how to apply the pipeline in an end-to-end manner on your own collection of text, see the example in the Jupyter notebook [Example_Pipeline.ipynb](https://github.com/salesforce/discord_questions/blob/master/Example_Pipeline.ipynb).

## Pipeline Components

The pipeline is composed of three components, all of which we release publicly, as HuggingFace Hub models. Each model card comes with example usage, so that each component can be used independently.

### 1. Question Generation

The Question Generation model (https://huggingface.co/Salesforce/discord_qg) is a Bart-large model trained on a combination of QA datasets (see paper for detail).
It achieves the highest performance on our evaluation framework, by being able to generate the largest fraction of Discord Questions. See [DiscordQG_Eval.ipynb](https://github.com/salesforce/discord_questions/blob/master/DiscordQG_Eval.ipynb) for experimental comparison.

### 2. Question Answering

The Question Answering model (https://huggingface.co/Salesforce/discord_qa) is a RoBERTa-large model trained on a combination of SQuAD 2.0 and NewsQA. This model is a standard extractive QA model, and can be replaced with more domain-specific models based on the textual domain.

### 3. Answer Consolidation

The Answer Consolidation model (https://huggingface.co/Salesforce/qa_consolidation) is a RoBERTa-large model which achieves the highest performance on the NAnCo evaluation dataset we annotated to evaluate answer consolidation methods.
We release the NAnCo annotations as well ([nanco_dataset.json](https://github.com/salesforce/discord_questions/blob/master/nanco_dataset.json)), as well as a Jupyter Notebook that reproduces the experimental comparison of answer consolidation models presented in the paper ([NAnCo.ipynb](https://github.com/salesforce/discord_questions/blob/master/NAnCo.ipynb)).

## Cite the work

If you make use of the code, models, or pipeline, please cite our paper:
```
@inproceedings{laban2022discord_questions,
  title={Discord Questions: A Computational Approach To Diversity Analysis in News Coverage},
  author={Philippe Laban and Chien-Sheng Wu and Lidiya Murakhovs'ka and Xiang 'Anthony' Chen and Caiming Xiong
},
  booktitle={Proceedings of the 2022 Findings of Empirical Methods in Natural Language Processing},
  volume={1},
  year={2021}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, reach out in the Issues, or by email: plaban@salesforce.com
All contributions welcome, for example if you want to apply the Discord Questions pipeline to a new corpus of text, or want to improve pipeline components.



