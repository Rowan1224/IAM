import sys

# sys.path.append(f'{sys.path[0]}/post/neuspell')
# sys.path.append(f'{sys.path[0]}/line')

sys.path.append('post/neuspell')
sys.path.append('line')

from line import predict
from post import utils
from post.neuspell import predict as neuspell
import argparse
import os

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inDir", default='datasets/IAM-data/formatted/test', type=str, help="provide the path of test image directory/folder"
    )
    parser.add_argument(
        "-m", "--modelDir", default='line/models', type=str, help="provide the pre-trained model directory"
    )

    parser.add_argument(
        "-p", "--post", choices=['brute', 'candidate', 'neuspell','neuspell-edit'], default='brute', type=str, help="select the post edit method"
    )




    args = parser.parse_args()
    return args


if __name__ == "__main__":

    
    args = create_arg_parser()

    modelDir = args.modelDir
    inDir = args.inDir

    lines = predict.predict(modelDir, inDir)

    bert_model, bert_vocab, english_words_set = utils.init('post/vocab')

    if args.post == 'brute':

        edited_lines = [(name, utils.edit_brute(sen, bert_vocab, english_words_set)) for name,sen in lines]

    elif args.post == 'candidate':

        edited_lines = [(name, utils.edit_candidate(sen, english_words_set, bert_model)) for name,sen in lines]

    elif args.post == 'neuspell':

               
        neu_preds = neuspell.predict(lines,'post/neuspell/data/new_models/bert-base-cased')
        
        edited_lines= [(name,sen) for name,sen in neu_preds.items()]
  
    
    elif args.post == 'neuspell-edit':

               
        neu_preds = neuspell.predict(lines,'post/neuspell/data/new_models/bert-base-cased')
        
        edited_lines = []
        for name,sen in lines:
            neu_pred = neu_preds[name]
            edited_lines.append((name, utils.edit_neuspell(sen, neu_pred, english_words_set)))


    outDir = 'results'
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    for name,prediction in edited_lines:
        if '.' in name:
            name = name.split('.')[0]
            

        fileName = os.path.join(outDir,f'{name}.txt')
        with open(fileName,'w') as file:
            file.write(prediction)


        