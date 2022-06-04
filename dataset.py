import pickle
import os
import random
import shutil
import string
import argparse
random.seed(10)


def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inDirectory", default='IAM-data/img', type=str, help="Provide the raw input image directory"
    )
    parser.add_argument(
        "-o", "--outDirectory", default='IAM-data/formatted', type=str, help="Provide the formatted output image directory"
    )

    parser.add_argument(
        "-l", "--labelFile", default='IAM-data/iam_lines_gt.txt', type=str, help="Provide the label data file location"
    )
    
    

    args = parser.parse_args()
    return args



def split_dataset(labelFile):


    with open(labelFile,'r') as file:
        lines = [line.strip() for line in file if len(line)>1]
        texts = {lines[i-1]:lines[i] for i in range(1,len(lines),2)}
    
    keys = list(texts.keys())
    random.shuffle(keys)
    split = int(len(keys)*0.80)
    train = keys[:split]
    test = keys[split:]
    val_split = int(len(train)*0.10)
    random.shuffle(train)
    valid = train[:val_split]
    train = train[val_split:]

    dataset = {
    'train':train,
    'test':test,
    'valid':valid
    }

    return dataset, texts
    

def get_charset():

    puncs = ['!','"','#','&',"'",'(',')','*','+',',','-','.','/',':',';','?']
    chars = string.ascii_lowercase+string.ascii_uppercase+string.digits+"".join(puncs)    
    charset = list(chars)
    charset.append(' ')
    return sorted(charset)

def format_dataset(dataset, labels, inDirectory, outDirectory):


    data = {'ground_truth':dict()}
    for sets, names in dataset.items():

        distFolder = f'{outDirectory}/{sets}'

        if not os.path.exists(outDirectory):
            os.mkdir(outDirectory)

        if not os.path.exists(distFolder):
            os.mkdir(distFolder)

        for i,file in enumerate(names):
            source = f'{inDirectory}/{file}'
            name =f'{sets}_{i}.png'
            dest = f'{distFolder}/{name}'
            shutil.copy(source,dest)
            if sets not in data['ground_truth']:
                data['ground_truth'][sets] = dict()
            data['ground_truth'][sets][name]={'text':labels[file]}


    data['charset'] = get_charset()

    outfile = open(f'{outDirectory}/labels.pkl','wb')
    pickle.dump(data,outfile)
    outfile.close()




def main():

    
    args = create_arg_parser()

    dataset, labels = split_dataset(args.labelFile)
    format_dataset(dataset,labels,args.inDirectory, args.outDirectory)




if __name__ == "__main__":
    main()



