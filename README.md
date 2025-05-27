# TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment
This is the official repository for our paper **TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment**
and contains code for (1) data preparation (2) prompting LLMs with quality criteria (3) training meta-TSRater  (4) scoring time series data samples  (5) evaluating TSRating on real time series datasets and models.

![overview](https://github.com/clsr1008/TSRating/blob/main/assets/overview.png)

## General Guidance for Running the Code

This repository does not provide a separate run script. Instead, each `.py` file contains a section with an `if __name__ == "__main__":` block that can be executed directly.

To run the code, simply modify the parameters within the `main` block to fit your specific dataset or use case. For each parameter, a comment has been added indicating whether it can be customized. You can easily change parameters like file paths, column names, indices, or block lengths directly in the `main` block.

##### example:

```python
if __name__ == "__main__": 
    # Example usage with replaceable parameters
    file_path = "../datasets/traffic/traffic.csv"  # to be changed
    column_name = "OT"  # to be changed
    start_idx = 4000  # to be changed
    end_idx = 8000  # to be changed
    block_length = 128  # to be changed
    prec = 4  # to be changed
    jsonl_path = "../middleware/traffic/blocks.jsonl"  # to be changed

    data = get_dataset_by_name(file_path, column_name, start_idx, end_idx, prec, block_length)

```

Each of the parameters marked with `# to be changed` is intended for customization based on your experimental setup.

#### Middleware usage

In our project, we have established a `middleware` directory to store the intermediate results for various datasets. This includes data files such as divided blocks, pairwise judgments, and annotation results. However, due to storage limitations, these files have not been uploaded to GitHub. We encourage users to organize their own data in a similar manner, ensuring that intermediate results are stored in a structured way for ease of access and future use.

## Datasets

This table outlines the datasets used in the project for various forecasting and classification tasks. All datasets are available for download on Hugging Face or can be access via https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2

| **Task**                   | **Dataset**s                                 |
| -------------------------- | -------------------------------------------- |
| **Long-term Forecasting**  | Electricity, Exchange Rate, Traffic, Weather |
| **Short-term Forecasting** | M4 yearly, monthly, dailly                   |
| **Classification**         | MedicalImages, CBF, BME, Handwriting         |

## Experiments
### Installing the Repo
Clone this repo and setup a new environment based on `python 3.11`. Install the requirements in the following order:
```bash
pip install packaging==23.2
pip install torch==2.1.2 torchaudio==2.5.1+cu118 torchvision==0.16.2
pip install -r requirements.txt
```

### Data preparation

The files `data_preparation/load_forecast_data.py`  and `data_preparation/load_classification_data.py`  can be used to process original datasets from forecasting and classification tasks, respectively.  The processing includes division into sliding blocks and serialization as LLM's input. 

The file `data_preparation/load_Time_300B.py` is designed specifically for sampling and serializing data from the large-scale **Time-300B** dataset. The Time-300B dataset can be downloaded from https://huggingface.co/datasets/Maple728/Time-300B.

In addition, we prepare `data_preparation/synthesis_data.py` for **Synthetic Validation** corresponding to Appendix B.2 in our paper. 

### Prompting LLMs with quality criteria

The script `prompting/run_score_pairwise.py` is used to collect pairwise judgments of LLMs.  The folder `prompting/templates/` contains the templates used in the paper. You can modify running configuration such as template_file, model and generations from the constructed command. The output dataset will be stored as `<output path>`, which is further converted to a excel file. 

### Training TSRater
We support two modes for training the TSRater model: **single-rater** and **meta-rater**.

Single-rater mode: Run `scoring/train_rater.py` to train a TSRater model on a single dataset for a specific quality criterion (e.g., trend, frequency, amplitude, or pattern). You can override the default hyperparameters or apply grid search for hyperparameter tuning. The trained models will be stored in the  `middleware/` folder.

Meta-rater mode: Use `meta_rater/meta_main.py` to train the meta-TSRater across multiple datasets. The model learns to generalize from diverse tasks. You can specify hyperparameter like adaptation steps, meta learning rate, inner learning rate and so on. After training, use `meta_rater/finetune_and_test.py` to perform few-shot finetuning and evaluation on a target dataset. 

### Scoring TS samples
`scoring/annotate.py` takes a dataset and a TSRater model and adds new columns to the dataset for the quality ratings. The quality ratings for all criteria are saved in `annotation.jsonl` file. Apart from our TSRating method, we investigate other baseline methods, scoring forecasting datasets datasets via `scoring/baseline_annotate.py` and classification dataset via `scoring/baseline_anotate_classification.py`. 

Finally, we provide `scoring/analysis.py` for visualization of the data samples with highest and lowest scores. The details can be found in Appendix B.1 in our paper.


### Evaluating
By running `evaluation/evaluate.py`, we can select data samples based on the obtained quality ratings and utilize them to train various time series models. The performance on a separate test set is acquired and printed on the console log. If you want to modify the running configuration, feel free to change the parameters in the main function.

## Citation
```bibtex
@
```
