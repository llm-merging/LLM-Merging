import torch 

from llm_merging.merging.Merges import Merges

from peft import get_peft_model, set_peft_model_state_dict

class FlanT5Avg(Merges):
    def __init__(self, name):
        super().__init__(name)


        '''
        These values are meant to be modified by the user.
        '''
        # Give a list of models to load for the merge 
        self.list_models = [("lorahub/flan_t5_xl-wiki_qa_Is_This_True_", "30a1ee2f857196c1eb996d854548cc19f45ac642"), 
                            ("lorahub/flan_t5_xl-kilt_tasks_hotpotqa_complex_question", "27d014366bec1c5333ba2e2fae966b7de3c02df1")]
        
        # Hyperparameters 
        self.base_model_name = "google/flan-t5-xl"
        self.base_model_revision_id = "7d6315df2c2fb742f0f5b556879d730926ca9001"
        self.is_peft = True
        self.max_seq_len = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "encoder_decoder"

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
        parameter_lambdas = [0.5, 0.5]

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