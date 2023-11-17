from transformers import T5Tokenizer, T5ForConditionalGeneration
import csv
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from sklearn.model_selection import train_test_split

class T5Item:
    def __init__(self):       
        BATCH_SIZE = 16
        BLEU = "bleu"
        ENGLISH = "en"
        ENGLISH_TEXT = "english_text"
        EPOCH = "epoch"
        INPUT_IDS = "input_ids"
        FILENAME = "TranslationDataset.csv"
        GEN_LEN = "gen_len"
        MAX_INPUT_LENGTH = 128
        MAX_TARGET_LENGTH = 128
        MODEL_CHECKPOINT = "./Data/t5-GEC"
        MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
        LABELS = "labels"
        PREFIX = ""
        SCORE = "score"
        SOURCE_LANG = "Input"
        TARGET_LANG = "Target"
        TRANSLATION = "translation"
        UNNAMED_COL = "Unnamed: 0"

    def postprocess_text(preds: list, labels: list) -> tuple:
        """Performs post processing on the prediction text and labels"""

        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
        """Takes the input data lists and converts into translation list of dicts"""

        data_dict = dict()
        data_dict[TRANSLATION] = []

        for sr_text, tr_text in zip(source_lang, target_lang):
            temp_dict = dict()
            temp_dict["Input"] = sr_text
            temp_dict["Target"] = tr_text

            data_dict[TRANSLATION].append(temp_dict)

        return data_dict

    def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                    model_checkpoint: str,
                                    tokenizer: AutoTokenizer):
        """Makes the data training ready for the model"""

        preped_data = []

        for row in dataset:
            inputs = PREFIX + row[source]
            targets = row[target]

            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                    truncation=True, padding=True)

            model_inputs[TRANSLATION] = row

            # setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                    truncation=True, padding=True)
                model_inputs[LABELS] = labels[INPUT_IDS]

            preped_data.append(model_inputs)

        return preped_data

    def compute_metrics(eval_preds: tuple) -> dict:
        """computes bleu score and other performance metrics """

        metric = load_metric("sacrebleu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {BLEU: result[SCORE]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

        result[GEN_LEN] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    #Loading Model and Data
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

    translation_data = pd.read_csv("./Data/trainData.csv")
    # evaluation_data = pd.read_csv("./Data/evalData.csv")

    #Preprocess
    X = translation_data["input"]
    y = translation_data["target"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                        shuffle=True,
                                                        random_state=100)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=0.20,
                                                    shuffle=True,
                                                    random_state=100)
    print("FINAL X-TRAIN SHAPE: ", x_train.shape)
    print("FINAL Y-TRAIN SHAPE: ", y_train.shape)
    print("X-VAL SHAPE: ", x_val.shape)
    print("Y-VAL SHAPE: ", y_val.shape)
        
    training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)

    validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)

    test_data = prep_data_for_model_fine_tuning(x_test.values, y_test.values)
        

    train_data = generate_model_ready_dataset(dataset=training_data[TRANSLATION],
                                            tokenizer=tokenizer,
                                            source="Input",
                                            target="Target",
                                            model_checkpoint=MODEL_CHECKPOINT)

    validation_data = generate_model_ready_dataset(dataset=validation_data[TRANSLATION],
                                                tokenizer=tokenizer,
                                                source="Input",
                                                target="Target",
                                                model_checkpoint=MODEL_CHECKPOINT)

    test_data = generate_model_ready_dataset(dataset=test_data[TRANSLATION],
                                                tokenizer=tokenizer,
                                                source="Input",
                                                target="Target",
                                                model_checkpoint=MODEL_CHECKPOINT)

    train_df = pd.DataFrame.from_records(train_data)
    validation_df = pd.DataFrame.from_records(validation_data)
    test_df = pd.DataFrame.from_records(test_data)
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    test_dataset = Dataset.from_pandas(test_df)



    # training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)
    # validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)


    #Train 

    args = Seq2SeqTrainingArguments(
        "./Model/t5-GEC",
        evaluation_strategy=EPOCH,
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.02,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True
    )

    ##Training

    trainer = Seq2SeqTrainer(
        model = model,
        args=args,
        train_dataset = train_dataset,
        eval_dataset = validation_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer
    )

trainer.train()



