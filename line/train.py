import sys
sys.path.append('../')
from config import get_config
import torch.multiprocessing as mp
from config import TrainerLineCTC
import pandas as pd
import argparse
import os
import torch



def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataDir", required=True, type=str, help="provide the path of dataset directory/folder"
    )
    parser.add_argument(
        "-m", "--modelDir", default='models', type=str, help="provide a directory path to save models"
    )

    parser.add_argument(
        "-o", "--outDir", default='outputs', type=str, help="provide a directory path to save outputs and results"
    )
    parser.add_argument(
        "-e", "--epochs", default=10, type=int, help="provide number of epochs"
    )

    parser.add_argument(
        "-c", "--continue_training", action='store_true', help="define if you continue training from last saved model or start again"
    )

    




    args = parser.parse_args()
    return args



def train_and_test(rank, params, continue_training=True):
    params["training_params"]["ddp_rank"] = rank
    params["training_params"]["load_prev_weights"] = continue_training
    
    if torch.cuda.device_count()==0:
        params['training_params']['force_cpu'] = True

    model = TrainerLineCTC(params)
    # Model trains until max_time_training or max_nb_epochs is reached

    
    if continue_training:
        if params['training_params']['max_nb_epochs'] < model.latest_epoch:
            max_epoch = model.latest_epoch+params['training_params']['max_nb_epochs']
            print(f"continuing training from {model.latest_epoch} to {max_epoch}")
            params['training_params']['max_nb_epochs'] = max_epoch
    else:
        print(f"Starting from the begining")
        model.latest_epoch = -1


    model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer",'pred']
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            output = model.predict(f'{dataset_name}-{set_name}', [(dataset_name, set_name), ], metrics, output=True)

            if output is not None:
                df = pd.DataFrame(output)
                filename = f'{dataset_name}-{set_name}-{model.latest_epoch}-predictions.csv'
                path = os.path.join(model.paths['results'],filename)
                df.to_csv(path,index=False)
            scorePath = os.path.join(model.paths["results"], f"predict_{dataset_name}-{set_name}_{model.latest_epoch}.txt")

            with open(scorePath,'r') as file:
                results = {}
                for line in file:
                    line = line.strip().split(":")
                    if line[0] in metrics:
                        results[line[0]] = float(line[1].strip())
            print("###################################################")
            print(f'{set_name}-set')
            print(f"sample size: {len(output)}")
            print(f"cer: {results['cer']}")
            print(f"wer: {results['wer']}")
            print("###################################################")






def main():

    dataset_name = "IAM" 

    
    args = create_arg_parser()

    modelDir = args.modelDir
    dataDir = args.dataDir
    outDir = args.outDir


    if not os.path.exists(dataDir):
        print("INVALID dataset directory")
        return


     
    #get model parameters
    params = get_config(datasetPath=dataDir,dataset_name=dataset_name,outputDir=outDir,modelDir=modelDir, epochs=args.epochs)

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params, args.continue_training)





if __name__ == "__main__":
    main()
