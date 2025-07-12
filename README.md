# BioLaySumm 2025
## Lay Summarization of Biomedical Research Articles and Radiology Reports @ BioNLP Workshop, ACL 2025

This is the repo for Team MIRAGE's submission for BiolaySumm Shared Task at the BioNLP Workshop, ACL 2025. The goal of this shared task is to push the boundaries of research on biomedical lay summarization.

## Repo Structure
Code in repository are meant to be executed using Google Colab, and requires minimally of 1 T4 GPU for inference. Fine-tuning requires minimally 1 L4 GPU. Code in `preprocessing_script` comes from CoLab notebooks and are meant to go through the datasets and extract the top 40 sentences based on different methods of evaluation.

## Get the Dataset

Download the PLOS and eLife datasets from Huggingface.

## End-to-End Pipeline
### Step 1: Preprocessing and Extract Salient Sentences

Our preprocessing mostly uses embeddings from BioBERT to make judgements about what is salient. Our preprocessing techniques are as follows:
1. Control, just take the first 4096 tokens from the article. This doesn't have a preprocessing script.
2. Comparing every sentence to the embedding for the title of the article.
3. Comparing every sentence to the embedding for the title and keywords of the article.
4. SVD Topic Modeling
5. Turn the entire article into an embedding and compare every sentence to that mean embedding.
6. Prepends title and keywords to the article and segment the article into four core sections(abstract, introduction, results, and discussion). From this condensed content, 
   we rank sentences according to their similarity to the mean embedding of the uncondensed article, and selectthe top 40 sentences.
7. The reverse of 6, where we segment the article to the same four core sections, extract the top 40 sentences and prepend the title and keywords.

Each of the scripts are found in the `preprocessing_script` and can be reimported into CoLab for use directly. 

### Available Scripts;
-[`preprocess23.py`](./preprocessing_script/preprocess23.py) --For Strategies 2 (title) & 3 (title + keywords)

-[`preprocess4.py`](./preprocessing_script/preprocess4.py) – For Strategy 4 (SVD topic modeling)

-[`preprocess567.py`](./preprocessing_script/preprocess567.py) – For Strategies 5, 6, and 7

### Example usage:
```bash
python preprocessing_script/preprocess23.py --input data/plos_train.json --output data/preprocessed_output.json
```
### Step 2: Fine-tune Llama3-8B-Instruct(LoRA)
The data for finetuning was prepared by randomly selecting 650 training instances from both eLife and PLOS, totaling 1300 shuffled samples.
 ```bash 
python src/train/finetune.py \
 ```

### Step 3: Generate Lay Summaries
Run inference with the trained model:
```bash 
python src/inference/inference.py \
  --model_path llama_3_1_1000 \
  --input_file data/preprocessed_articles.json \
  --output_file output/summaries.json                              
```

### Step 4: Evaluation
For evaluation, we used 150 randomly selected validation samples from both datasets, totaling 300 shuffled samples.
We evaluate our system on the validation splits of the **PLOS** and **eLife** datasets using metrics provided by the BioLaySumm 2025 organizers. The evaluation focuses on three key aspects:

### Metrics

| Aspect           | Metrics Used                                                                   |
|------------------|--------------------------------------------------------------------------------|
| **Relevance**    | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore                                           |
| **Readability**  | Flesch-Kincaid Grade Level (FKGL), Dale-Chall (DCRS), Coleman-Liau Index (CLI) |
| **Factuality**   | SummaC, AlignScore                                                             |


### Run Evaluation
To evaluate a generated summary file (JSON format):
```bash 
python src/eval/evaluate.py \
  --preds outputs/summaries.json \
  --refs data/gold_summaris.json 
```

You may need to install evaluation packages:
```bash
pip install rouge-score bert-score summa
```

### Summarization
Run 'inference.py' to generate summaries using Llama-3-8b-instruct. 


### Counterfactual Data Augmentation Experiment ###
Run counterfactual_dataprep.py to prepare data for counterfactual finetuning. It swaps out biomedical entities in the gold summaries for random entities within the categories that they belong to. 
Run counterfactual_finetune.py to finetune model on counterfactually augmented data.

### Postprocessing: Definition-insertion ###
As a postprocessing step, after generating summaries, run postprocessing.py to generate term-definition dictionary. The output will be a list of dictionaries containing two fields; term-definition dictionary and the summaries.
Run postprocessing_inference.py to paraphrase summaries using term-dictionary
