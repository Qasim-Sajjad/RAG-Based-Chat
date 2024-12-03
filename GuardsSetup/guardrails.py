import torch
from groq import Groq
from GuardsSetup.model_setup import ModelSetup
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=".env")
# hf_llama_guard_tk = os.getenv(key="HF_LLAMA_GUARD")
# groq_api_key = os.getenv(key="GROQ_API_KEY")

class GuardRail:

    def __init__(self,
                 model_name : str,
                 groq_api_key: str = None,
                 hf_llama_guard_tk: str = None,
                 run_model_locally: bool = False
        ):
        #check validation of api keys.
        if run_model_locally and hf_llama_guard_tk is None:
            raise ValueError("Hugging Face Api Key for Meta-LlamaGuard 7b not found.Please get api key from hugging face")
        elif groq_api_key is None:
            raise ValueError("GROQ API KEY not found.Please set the GROQ_API_KEY environment variable in your .env file.")

        #setting up access token for either hf or groq.
        self.access_token = hf_llama_guard_tk if run_model_locally==True else groq_api_key
        self.run_locally = run_model_locally

        #check validation of model name for both cases.
        assert model_name in ["meta-llama/LlamaGuard-7b"] if run_model_locally else ["llama-guard-3-8b"] , f"Invalid Model Name:{model_name}"

        self.model_id = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def setup_model_n_tokenizer(self) -> None:
        
        #loading model and tokenizer from ModelSetup class.
        model_setup = ModelSetup(self.model_id,self.access_token)
        self.model,self.tokenizer = model_setup.setup_model_n_tokenizer()
        
        return None
    
    def read_unsafe_categories(self) -> str:

        #read unsafe categories defined in utils folder.
        with open(file="utils/unsafe_categories.txt",mode="r") as f:
            unsafe_categories = f.read()

        return unsafe_categories
    
    def moderate_prompt_for_chat(self,
                                 chat : list[dict]) -> str:

        conversation = [turn["content"] for turn in chat]
        role = "Agent" if len(conversation) % 2 == 0 else "User"

        # PROMPT TEMPLATE FROM LLAMA_GUARD PAPER
        # SEE LLAMA_GUARD_PAPER_REF IN IMAGES FOR REFERENCE
        prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {self.read_unsafe_categories()}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    """

        # Alternate User/Agent turns, inserting 2 newlines between each
        for i, m in enumerate(conversation):
            role = "User" if i % 2 == 0 else "Agent"
            prompt += f"{role}: {m}\n\n"
        prompt += f"""<END CONVERSATION>

    Provide your safety assessment for {role} in the above conversation:
    - A Single line which either read 'safe' or 'unsafe'.[/INST]"""
        return prompt

    def moderate_custom_chat(self,
                             chat : list[dict]) -> str: 
        
        #setup prompt for usage,check prompt problem.
        #prompt = self.moderate_prompt_for_chat(chat)

        #Get input ids from tokenizer of our model and get resultant output.
        input_ids = self.tokenizer.apply_chat_template(chat,return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]

        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def sanitize_input(self,
                       user_description : str) -> str:
        #setting up chat template for usage.
        chat_template = [
            {'role':'user','content':f"{user_description}"}
        ]

        #set up model and tokenizer locally for inference.
        if self.run_locally:
            self.setup_model_n_tokenizer()
            self.user_content = user_description
            result = self.moderate_custom_chat(chat=chat_template)
            
            return result
        # Using groq with llama-guard for groq cloud inference.
        else:
            print(self.access_token)
            client = Groq(api_key=self.access_token)

            chat_completion = client.chat.completions.create(
                messages=chat_template,
                model=self.model_id
            )

            #response eiter contains 'safe' or 'unsafe' with category it violated.
            # returning only 'safe' or 'unsafe' for now.
            response = chat_completion.choices[0].message.content.split(sep="\n")
            return response[0]

    def sanitize_output(self,
                        initial_description : str,
                        llm_response : str):
        
        #right now taking initial_description in parameter could be removed
        #and self.user_content can be used for same initial description output.

        chat_template = [
            {'role':'user','content':f"{initial_description}"},
            {'role':'assistant','content':f"{llm_response}"}
        ]

        #if running model locally, check if model and tokenizer are set.
        if self.run_locally:
            if self.model is None or self.tokenizer is None:
                self.setup_model_n_tokenizer()
                    
            result = self.moderate_custom_chat(chat=chat_template)
            return result
        #use groq for groq cloud inference.
        else:
            client = Groq(api_key=self.access_token)

            chat_completion = client.chat.completions.create(
                messages=chat_template,
                model=self.model_id
            )

            #response eiter contains 'safe' or 'unsafe' with category it violated.
            # returning only 'safe' or 'unsafe' for now.
            response = chat_completion.choices[0].message.content.split(sep="\n")
            return response[0]


# Main For Testing Purpose.
# if __name__ == "__main__":
#     #while running model locally, use GPU having Dedicated vram 6gb or above.
#     #causing errors on cpu running for now, use run_locally as False for alternative.
#     guard = GuardRail(model_name="llama-guard-3-8b",run_model_locally=False)

#     #check input.
#     input_response = guard.sanitize_input(user_description="I am a good boy")
#     print(f"Input response:{input_response}")
#     #check output.
#     output_response = guard.sanitize_output(initial_description="I am a good boy",llm_response="yeah i know man")
#     print(f"Output response:{output_response}")