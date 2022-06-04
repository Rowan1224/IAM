from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
from torch.nn.functional import log_softmax
from torch.nn import Conv2d, AdaptiveMaxPool2d
from torch.nn import Module
from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
import editdistance
import re
import torch
from torch.nn import CTCLoss


class Decoder(Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.vocab_size = params["vocab_size"]

        self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.end_conv = Conv2d(in_channels=256, out_channels=self.vocab_size+1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.ada_pool(x)
        x = self.end_conv(x)
        x = torch.squeeze(x, dim=2)
        return log_softmax(x, dim=1)


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params, is_inference=False):
        super(TrainerLineCTC, self).__init__(params, is_inference)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        self.optimizer.zero_grad()
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        self.backward_loss(loss)
        self.optimizer.step()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")

        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, x, y, x_len, y_len, loss=None, metric_names=list()):
        batch_size = y.shape[0]
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]

        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss / metrics["nb_chars"]
        metrics["nb_samples"] = len(x)
        return metrics


    def evaluate_batch_custom(self, batch_data):

        x = batch_data["imgs"].to(self.device)
        names = batch_data["names"]
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
     

        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        batch_size = len(names)
        ind_x = [pred[i][:x_reduced_len[i]] for i in range(batch_size)]
        
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]

        output = [(name,prediction) for name, prediction in zip(names,str_x)]

        return output


def get_config(datasetPath = "../datasets", dataset_name = "IAM", outputDir = "outputs", modelDir="models", epochs=10):

    
    params = {
            "dataset_params": {
                "datasets": {
                    dataset_name: datasetPath,
                },
                "train": {
                    "name": "{}-train".format(dataset_name),
                    "datasets": [dataset_name, ],
                },
                "valid": {
                    "{}-valid".format(dataset_name): [dataset_name, ],
                },
                "dataset_class": OCRDataset,
                "config": {
                    "width_divisor": 8,  # Image width will be divided by 8
                    "height_divisor": 32,  # Image height will be divided by 32
                    "padding_value": 0,  # Image padding value
                    "padding_token": 1000,  # Label padding value (None: default value is chosen)
                    "charset_mode": "CTC",  # add blank token
                    "constraints": ["CTC_line"],  # Padding for CTC requirements if necessary
                    "preprocessings": [
                        {
                            "type": "dpi",  # modify image resolution
                            "source": 300,  # from 300 dpi
                            "target": 150,  # to 150 dpi
                        },
                        {
                            "type": "to_RGB",
                            # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                        },
                    ],
                    # Augmentation techniques to use at training time
                    "augmentation": {
                        "dpi": {
                            "proba": 0.2,
                            "min_factor": 0.75,
                            "max_factor": 1.25,
                        },
                        "perspective": {
                            "proba": 0.2,
                            "min_factor": 0,
                            "max_factor": 0.3,
                        },
                        "elastic_distortion": {
                            "proba": 0.2,
                            "max_magnitude": 20,
                            "max_kernel": 3,
                        },
                        "random_transform": {
                            "proba": 0.2,
                            "max_val": 16,
                        },
                        "dilation_erosion": {
                            "proba": 0.2,
                            "min_kernel": 1,
                            "max_kernel": 3,
                            "iterations": 1,
                        },
                        "brightness": {
                            "proba": 0.2,
                            "min_factor": 0.01,
                            "max_factor": 1,
                        },
                        "contrast": {
                            "proba": 0.2,
                            "min_factor": 0.01,
                            "max_factor": 1,
                        },
                        "sign_flipping": {
                            "proba": 0.2,
                        },
                    },
                }
            },

            "model_params": {
                # Model classes to use for each module
                "models": {
                    "encoder": FCN_Encoder,
                    "decoder": Decoder,
                },
                "transfer_learning": None,  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
                "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
                "dropout": 0.5,  # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
            },

            "training_params": {
                "output_folder": outputDir,  # folder names for logs
                "checkpoint_folder": modelDir,
                "max_nb_epochs": epochs,  # max number of epochs for the training
                "max_training_time":  3600*(24+23),  # max training time limit (in seconds)
                "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
                "interval_save_weights": None,  # None: keep best and last only
                "use_ddp": False,  # Use DistributedDataParallel
                "use_apex": False,  # Enable mix-precision with apex package
                "nb_gpu": torch.cuda.device_count(),
                "batch_size": 4,  # mini-batch size per GPU
                "optimizer": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": True,
                    }
                },
                "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
                "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
                "focus_metric": "cer",   # Metrics to focus on to determine best epoch
                "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
                "set_name_focus_metric": "{}-valid".format(dataset_name),
                "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
                "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
                "force_cpu": False,  # True for debug purposes to run on cpu only
            },
        }

    return params