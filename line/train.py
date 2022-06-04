import config
import torch.multiprocessing as mp
from config import TrainerLineCTC



def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    # Model trains until max_time_training or max_nb_epochs is reached
    model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "time", "worst_cer",'pred']
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)





if __name__ == "__main__":
    dataset_name = "IAM"  

    params = config.params

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)
