import torch 

from llm_merging.merging.Merges import Merges
from peft import get_peft_model, set_peft_model_state_dict

class LlamaAvg(Merges):
    def __init__(self, name):
        super().__init__(name)

        '''
        These values are meant to be modified by the user.
        '''
        # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.list_models = [("abcdabcd987/gsm8k-llama2-7b-lora-16", "636b5eb8da724edae406ba69ef90fd06478e6df7"), 
                            ("FinGPT/fingpt-forecaster_dow30_llama2-7b_lora", "69f77190315afdb03a889d89bf2a0f932b311617")]

        # Hyperparameters 
        self.base_model_name = "meta-llama/Llama-2-7b-hf"
        # We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.base_model_revision_id = "01c7f73d771dfac7d292323805ebc428287df4f9"


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
                    # first model has rank 16 and second model has rank 8, so we expand the second model to rank 16 by adding zeros
                    if "A" in parameter_name:
                        parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=0)
                    else:
                        assert "B" in parameter_name
                        parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=1)
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
        # Modify the base model. This is needed for Peft, which wraps the base_model in a Peft wrapper. 
        huggingface_config = list(self.loaded_configs.values())[0]
        if huggingface_config is not None:
            self.base_model = get_peft_model(self.base_model, huggingface_config)
            set_peft_model_state_dict(self.base_model, self.merged_model)
        
        else:
            self.base_model.load(self.merged_model)

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model