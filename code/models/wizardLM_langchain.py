import torch
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from pydantic import Extra

class WizardLM(LLM, extra = Extra.allow):
    
    # HuggingFace models
    base_model: str = "E:\\derivingStructure\\llm_models"
    #base_model: str = "ehartford/WizardLM-13B-Uncensored"
    
    # Model parameters
    temperature:float = 0
    num_beams:int = 6
    max_new_tokens:int = 512
    
    # Program parameters
    verbose:bool = False
    
    def __init__(self):
        super().__init__()
        
        # load the pipeline
        #tokenizer = LlamaTokenizer.from_pretrained(self.base_model, use_fast = True)
        #llm_model = LlamaForCausalLM.from_pretrained(self.base_model, torch_dtype = torch.float16, device_map = "auto")
        #llm_model.tie_weights()
        
        self._pipe = pipeline(task = "text-generation", model = self.base_model, tokenizer = self.base_model,
                              device_map = "auto", torch_dtype = torch.float16)
        
        # Generation configuration
        self._pipe.tokenizer.truncation_side = 'left'
        self.generation_configs = GenerationConfig.from_pretrained(
            pretrained_model_name = self.base_model, temperature = self.temperature, num_beams = self.num_beams, max_new_tokens = self.max_new_tokens, top_p = .85)
        
        #       num_beam_groups = 2

        self._device = self._pipe.device
        
        print(f"\n\n[INFO] {self.base_model} ({self._pipe.torch_dtype}) has been loaded! "\
            f"({self._device}, {torch.cuda.get_device_name(self._device)})\n")
    
    def __evaluate(self, prompts: Union[List, str]):

        # Generate the output
        outputs = self._pipe(prompts, return_full_text = False, clean_up_tokenization_spaces = True, batch_size = 50, 
                             generation_config = self.generation_configs)

        # Extract the generated text
        outputs = [output['generated_text'].strip() for output in outputs] 
        
        # Extract the textual data
        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
    
    def _call(self, prompts: Union[List, str], stop: Optional[List[str]] = None) -> str: 
        return self.__evaluate(prompts)  # type: ignore
    
    @property
    def _llm_type(self) -> str:
        return self.base_model
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        
        model_params = {"base_model": self.base_model}
        model_params.update(self.generation_configs.to_dict())
        return model_params

class GLM4CausalLM:
    def __init__(self, model_path: str, trust_remote_code: bool = True, torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()

        print(f"\n\n[INFO] {self.base_model} ({self._pipe.torch_dtype}) has been loaded! ")

    def generate_text(self, prompts: Union[str, List[str]], **generate_kwargs) -> Union[str, List[str]]:
        # Generate text using the model
        generation_output = self._pipe(prompts, **generate_kwargs)
        # Extract the generated text
        generated_texts = [output['generated_text'].strip() for output in generation_output]
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts
