import torch,huggingface_hub,os
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch
)

class ModelSetup:
    def __init__(self, model_id, access_token):
        self.model_id = model_id
        self.access_token = access_token
        self.tokenizer = None
        self.model = None

    def setup_model_n_tokenizer(self):
        # Login to Hugging Face
        huggingface_hub.login(token=self.access_token, add_to_git_credential=True)
        
        # Set up device and data type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Determine available resources
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        cpu_count = os.cpu_count()
        
        # Set up BitsAndBytesConfig for 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # Choose loading strategy based on available resources
        if total_memory >= 6 * 1024 * 1024 * 1024:  # If GPU has more than 6GB
            print("Loading model on GPU with 4bit-quantization having memory greater than or equal to 6GB")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quantization_config
            )
        elif cpu_count > 4:  # If CPU has more than 4 cores
            print(f"Loading model on CPU with quantization as CPU count is {cpu_count}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="cpu",
                quantization_config=quantization_config
            )
        else:
            print("Loading model with disk offloading")
            with init_empty_weights():
                config = AutoModelForCausalLM.from_pretrained(self.model_id).config
                self.model = AutoModelForCausalLM.from_config(config)
            
            self.model = load_checkpoint_and_dispatch(
                self.model,
                self.model_id,
                device_map="auto",
                no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "GPTNeoXLayer"],
                dtype=dtype,
                offload_folder="model_offload"
            )

        return self.model,self.tokenizer

# Main only for testing purposes.
def moderate_chat(model,tokenizer,device,chat_template):
        input_ids = tokenizer.apply_chat_template(chat_template,return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]

        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    hf_access_token = os.getenv(key="HF_LLAMA_GUARD")
    setup = ModelSetup("meta-llama/LlamaGuard-7b", hf_access_token)
    model,tokenizer = setup.setup_model_n_tokenizer()
    chat_template = [
        {'role':'user','content':f"Hello how are you"}
    ]
    print(moderate_chat(model=model,tokenizer=tokenizer,device="cuda",chat_template=chat_template))