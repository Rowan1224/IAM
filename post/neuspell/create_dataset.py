import os
import pandas as pd
import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inDir", default = '../../line/best/results', type=str, help="provide the path of prediction files that also contains true value"
    )
    parser.add_argument(
        "-o", "--outDir", default = 'data', type=str, help="provide output directory"
    )
    parser.add_argument(
        "-e", "--epoch", default = '2990', type=str, help="provide the epoch number on which predictions files should be extracted"
    )

    args = parser.parse_args()
    return args

def main():
    
    args = create_arg_parser()
    inDir = args.inDir
    outDir = args.outDir

    if not os.path.exists(inDir):
        print("INVALID directiry")
        return
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    sets = {'train','test','valid'}
    
    for s in sets:

        location = f'{inDir}/IAM-{s}-{args.epoch}-predictions.csv'

        df = pd.read_csv(location)
        incorrect = df['predictions'].values.tolist()
        correct = df['true'].values.tolist()



        with open(f'{outDir}/{s}_clean.txt','w') as file:
            for sen in correct:
                file.write(f'{sen}\n')


        with open(f'{outDir}/{s}_corrupt.txt','w') as file:
            for sen in incorrect:
                file.write(f'{sen}\n')


if __name__ == "__main__":

    main()