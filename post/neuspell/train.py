
from seq_modeling.helpers import load_data
from seq_modeling.helpers import get_tokens
from model import BertChecker
import argparse
from util import wer_from_files, cer_from_files, format_string_for_wer
import os

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-e", "--epoch", default = 100,type=int, help="provide epochs to train"
        )



    args = parser.parse_args()
    return args

def main():

    args = create_arg_parser()

    data_dir = 'data'

    if not os.path.exists(data_dir):
        print("Dataset folder does not exist")
        return

    #following files should be present in the data folder if create_dataset.py is executed properly

    train_clean_file = "train_clean.txt"
    train_corrupt_file = "train_corrupt.txt"

    test_clean_file = "test_clean.txt"
    test_corrupt_file = "test_corrupt.txt"

    valid_clean_file = "valid_clean.txt"
    valid_corrupt_file = "valid_corrupt.txt"

    # Step-0: Load your train and test files, create a validation split
    train_data = load_data(data_dir, train_clean_file, train_corrupt_file)
    

    # Create vocab file. This serves as the target vocab file and we use the defined model's default huggingface
    # tokenizer to tokenize inputs appropriately.
    vocab = get_tokens([i[0] for i in train_data], keep_simple=True, min_max_freq=(1, float("inf")), topk=100000)

    #Initialize a model
    checker = BertChecker(device="cuda")
    checker.from_huggingface(bert_pretrained_name_or_path="bert-base-cased", vocab=vocab)

    #Finetune the model on your dataset
    checker.finetune(train_clean_file = train_clean_file, 
            train_corrupt_file = train_corrupt_file,
            valid_clean_file = valid_clean_file,
            valid_corrupt_file = valid_corrupt_file,
            data_dir=data_dir, n_epochs=args.epoch)

    #evaluate the trained model
    pred = checker.evaluate(clean_file=test_clean_file, corrupt_file=test_corrupt_file, data_dir=data_dir)

    
    predPath = f'data/test_predictions.txt'
    truePath = f'data/{test_clean_file}'

    #save predictions in txt file
    with open(predPath,'w') as file:
        for sen in pred:
            file.write(f'{sen}\n')

    
    #print wer and cer for the test set
    print("###################################################")
    print(f'test')
    print(f"sample size: {len(pred)}")
    print(f"cer: {cer_from_files(truePath,predPath)}")
    print(f"wer: {wer_from_files(truePath,predPath)}")
    print("###################################################")

if __name__ == "__main__":

    main()
