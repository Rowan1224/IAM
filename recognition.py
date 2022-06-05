import sys
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
        "-i", "--inDir", required=True, type=str, help="provide the path of test image directory/folder"
    )
    parser.add_argument(
        "-m", "--modelDir", default='line/best/models', type=str, help="provide the pre-trained model directory"
    )

    parser.add_argument(
        "-p", "--post", choices=['brute', 'candidate', 'neuspell','neuspell-edit','no'], default='brute', type=str, help="select the post edit method"
    )

    parser.add_argument(
        "-nm", "--neuModel", default='post/neuspell/data/models/bert-base-cased', type=str, help="In case of any neuspell method, provide the model path"
    )




    args = parser.parse_args()
    return args

def main():

    args = create_arg_parser()

    modelDir = args.modelDir
    inDir = args.inDir

    if not os.path.exists(modelDir):
        print("INVALID model directory")
        return
    if not os.path.exists(inDir):
        print("INVALID image directory")
        return

    if args.post=='neuspell' or args.post=='neuspell-edit':
        if not os.path.exists(args.neuModel):
            print("INVALID neuspell model directory")
            return
    
        

    lines = predict.predict(modelDir, inDir)

    bert_model, bert_vocab, english_words_set = utils.init('post/vocab')

    if args.post == 'brute':

        edited_lines = [(name, utils.edit_brute(sen, bert_vocab, english_words_set)) for name,sen in lines]

    elif args.post == 'candidate':

        edited_lines = [(name, utils.edit_candidate(sen, english_words_set, bert_vocab, bert_model)) for name,sen in lines]

    elif args.post == 'neuspell':

               
        neu_preds = neuspell.predict(lines,args.neuModel)
        
        edited_lines= [(name,sen) for name,sen in neu_preds.items()]
  
    
    elif args.post == 'neuspell-edit':

               
        neu_preds = neuspell.predict(lines,args.neuModel)
        
        edited_lines = []
        for name,sen in lines:
            neu_pred = neu_preds[name]
            edited_lines.append((name, utils.edit_neuspell(sen, neu_pred, english_words_set)))

    elif args.post == 'no':

        edited_lines = lines


    outDir = 'results'
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    for name,prediction in edited_lines:
        if '.' in name:
            name = name.split('.')[0]
            

        fileName = os.path.join(outDir,f'{name}.txt')
        with open(fileName,'w') as file:
            file.write(prediction)


    print("##################################\n")
    print(f"Total predicted {len(edited_lines)} images")
    print(f"outputs saved in {outDir} directory")
    print("\n##################################")

if __name__ == "__main__":

    main()