import torch 

from llm_merging.merging.Merges import Merges
from peft import get_peft_model, set_peft_model_state_dict

class TinyLlamaAvg(Merges):
    def __init__(self, name):
        super().__init__(name)

        '''
        These values are meant to be modified by the user.
        '''
            # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31
        self.list_models = [("TinyLlama/TinyLlama_v1.1", "f67f7cf6a907e567552b946699a9b9b45394fc46"),
                            ("TinyLlama/TinyLlama_v1.1_math_code", "36978c95f61ba8078250f04d71b5404fa9733614")]

        # Hyperparameters 
        self.base_model_name = "TinyLlama/TinyLlama_v1.1"
        # We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.base_model_revision_id = "f67f7cf6a907e567552b946699a9b9b45394fc46"
        self.is_peft = False

        self.max_seq_len = None
        self.max_gen_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "decoder"
        '''
        These are variables used later in the code and not intended to be set, but feel free to adapt to your use case.  
        '''
        # Loaded models and configs 
        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}



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
        parameter_lambdas = [0.8, 0.2]

        # Get individual models 
        all_models = list(self.loaded_models.values())

        # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        all_parameter_names = all_models[0].keys()

        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                if merged_parameter is None:
                    merged_parameter = torch.clone(parameter) * parameter_lambda
                else:
                    merged_parameter += parameter * parameter_lambda
            self.merged_model[parameter_name] = merged_parameter

        '''
        3) Load base model and tokenizer
        '''
        self._load_base_model()
        self._load_tokenizer()

        '''
        4) Load merged model into base model 
        '''
        self.base_model.load_state_dict(self.merged_model)

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model