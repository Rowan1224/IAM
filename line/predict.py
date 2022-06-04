from config import TrainerLineCTC
from config import get_config


def predict(modelDir,imageDir):

    params = get_config()
    params["training_params"]["ddp_rank"] = 0
    params['training_params']['checkpoint_folder'] = modelDir

    model = TrainerLineCTC(params, is_inference=True)

    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

            
    preds = model.predict_custom("{}-{}".format('IAM', 'test'), [('IAM', imageDir), ], custom=True)

    return preds

