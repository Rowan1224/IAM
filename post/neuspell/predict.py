from model import BertChecker
from seq_modeling.subwordbert import  model_inference_custom




def predict(test_data,modelDir='data/new_models/bert-base_cased'):

    
    checker = BertChecker(device="cuda")

    checker.from_pretrained(
        
        ckpt_path=modelDir 
    )
    
    pred = model_inference_custom(checker.model,
                                    test_data,
                                    topk=1,
                                    device=checker.device,
                                    batch_size=1,
                                    vocab_=checker.vocab)


    

    return pred