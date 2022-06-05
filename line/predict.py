from config import TrainerLineCTC
from config import get_config
import torch

def predict(modelDir,imageDir):

    '''
    returns the predictions from the best line recognition model 

    '''

    params = get_config()
    params["training_params"]["ddp_rank"] = 0
    params['training_params']['checkpoint_folder'] = modelDir

    if not torch.cuda.is_available():
        params['training_params']['force_cpu'] = True
    else:
        params['training_params']['nb_gpu'] = torch.cuda.device_count()

    model = TrainerLineCTC(params, is_inference=True)

    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

            
    preds = model.predict_custom("{}-{}".format('IAM', 'test'), [('IAM', imageDir), ], custom=True)

    return preds

