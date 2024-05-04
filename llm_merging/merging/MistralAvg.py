import torch 
from typing import List, Tuple, Dict, Callable, Any

from llm_merging.merging.Merges import Merges

from peft import get_peft_model, set_peft_model_state_dict

class MistralAvg(Merges):
    def __init__(self):
        super().__init__()

        # Give an interesting name to your method
        self.name = "mistral_avg"
        # Give a list of models to load for the merge 
        self.list_models = ["predibase/gsm8k", 
                            "predibase/glue_mnli"]
        
        # Loaded models and configs 
        self.loaded_models = {}
        self.loaded_configs = {}

        # Hyperparameters 
        self.base_model_name = "mistralai/Mistral-7B-v0.1"
        self.max_seq_len = 1024
        self.max_gen_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "decoder"

    # Implement merge function 
    def merge(
        self,
    ):

        '''
        1) Load HuggingFace checkpoints and configs 
        '''
        super()._load_huggingface_models_and_configs()
        '''
        2) Merge checkpoints  
        '''
        parameter_lambdas = [1, 0]

        # Get individual models 
        all_models = list(self.loaded_models.values())

        # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        all_parameter_names = all_models[0].keys()
        merged_model = {}

        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                if merged_parameter is None:
                    merged_parameter = torch.clone(parameter) * parameter_lambda
                else:
                    # # first model has rank 16 and second model has rank 8, so we expand the second model to rank 16 by adding zeros
                    # if "A" in parameter_name:
                    #     parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=0)
                    # else:
                    #     assert "B" in parameter_name
                    #     parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=1)
                    merged_parameter += parameter * parameter_lambda
            merged_model[parameter_name] = merged_parameter

        '''
        3) Load base model and tokenizer 
        '''
        self._load_base_model()
        self._load_tokenizer()

        # '''
        # 4) Load merged model into base model 
        # '''
        # # Modify the base model. This is needed for Peft, which wraps the base_model in a Peft wrapper. 
        # huggingface_config = list(self.loaded_configs.values())[0]
        # if huggingface_config is not None:
        #     self.base_model = get_peft_model(self.base_model, huggingface_config)
        #     set_peft_model_state_dict(self.base_model, merged_model)
        
        # else:
        #     self.base_model.load(merged_model)

        return self.base_model