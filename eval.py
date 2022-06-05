import sys
sys.path.append('post/neuspell')
sys.path.append('line')
from basic.utils import format_string_for_wer, wer_from_list_str, cer_from_list_str
import argparse
import os
from post import utils
import pandas as pd
from post.neuspell import predict as neuspell

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inFile", required=True, type=str, help="provide the path of prediction file that also contains true value"
    )

    parser.add_argument(
        "-p", "--post", choices=['brute', 'candidate', 'neuspell','neuspell-edit', 'no'], default='brute', type=str, help="select the post edit method"
    )

    parser.add_argument(
        "-nm", "--neuModel", default='post/neuspell/data/models/bert-base-cased', type=str, help="In case of any neuspell method, provide the model path"
    )




    args = parser.parse_args()
    return args


def eval(true, pred, set_name):

    """print the wer and cer for a given list of prediction and ground truths"""

    true = [format_string_for_wer(line) for line in true]

    print("###################################################")
    print(f'{set_name}')
    print(f"sample size: {len(true)}")
    print(f"cer: {cer_from_list_str(true,pred)}")
    print(f"wer: {wer_from_list_str(true,pred)}")
    print("###################################################")



def main():
    args = create_arg_parser()


    inFile = args.inFile

    if not os.path.exists(inFile):
        print("File does not exist")
        return


    if args.post=='neuspell' or args.post=='neuspell-edit':
        if not os.path.exists(args.neuModel):
            print("INVALID neuspell model directory")
            return
    
    
    if not inFile.endswith('.csv'):
        print("CSV is expected but not found")
        return

    df = pd.read_csv(inFile)

    preds = df['predictions'].values.tolist()
    true = df['true'].values.tolist()

       
    #load model and vocabs for post edit methods
    bert_model, bert_vocab, english_words_set = utils.init('post/vocab')

    if args.post == 'brute':

        edited_lines = [utils.edit_brute(sen, bert_vocab, english_words_set) for sen in preds]

    elif args.post == 'candidate':

        edited_lines = [ utils.edit_candidate(sen, english_words_set, bert_vocab, bert_model) for sen in preds]

    elif args.post == 'neuspell':

               
        neu_preds = neuspell.predict(preds,args.neuModel)
        
        edited_lines= [sen for _,sen in neu_preds.items()]
  
    
    elif args.post == 'neuspell-edit':

               
        neu_preds = neuspell.predict(preds,args.neuModel)
        
        edited_lines = []
        for name,sen in preds:
            neu_pred = neu_preds[name]
            edited_lines.append(name, utils.edit_neuspell(sen, neu_pred, english_words_set))

    elif args.post == 'no':

        edited_lines = preds

    
    #perform evaluation
    eval(true,edited_lines,inFile.replace('.csv',''))


if __name__ == "__main__":

    main()

