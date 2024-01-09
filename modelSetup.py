from torch import bfloat16
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from vllm import LLM, SamplingParams


class ModLoader():

    def __init__(self, model_path: str="/mnt/MixtralVolume/Mixtral-8x7B-Instruct-v0.1/", quant: str='16b'):
        
        initial_time = time.time()
        
        if quant == 'vllm':
            self.model = VLLMMod(model_path)
        if quant == 'vllm4':
            self.model = VLLMMod(model_path, 4)
        elif quant == '4':
            self.model = Mod4Quant(model_path)
        elif quant == '8':
            self.model = Mod8Quant(model_path)
        elif quant == '16b':
            self.model = Mod16B(model_path)
        else:
            raise ValueError(f'Q\Input quantization method: {quant} isnt supported. Please use one of the following as a -q parameter: vllm, 4, 8, 16b')

        after_load = time.time()

        load_time = round(after_load - initial_time, 3)

        print(f"Model loaded. Took {load_time} seconds")

    def query(self, i: int, prompt: str, do_print: bool=False):

        initial_time = time.time()

        formatted_prompt = self.format_prompt(prompt)
            
        res = self.model.query(formatted_prompt)

        generated_time = time.time()

        query_time = generated_time - initial_time
        query_time = round(query_time, 5)


        if do_print:
            self.pretty_print(i, query_time, prompt, res)
        
        return (res, query_time)

    def format_prompt(self, prompt: str):
        return f'<s> [INST] {prompt} [/INST]'

    def pretty_print(self, i: int, query_time: float, prompt: str, res: str):
        print('-'*15)
        print(f'Took {query_time} seconds to test prompt #{i}')
        print(f'Prompt: {prompt}')
        print(f'Response: {res}')
        print('-'*15)
        print()

class Mod16B():

    def __init__(self, model_path):

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=bfloat16,
                device_map='auto'
                )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        self.generate_text = transformers.pipeline(
                model=self.model, tokenizer=self.tokenizer,
                return_full_text=False,  # if using langchain set True
                task="text-generation",
                # we pass model parameters here too
                #temperature=0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                #top_p=0.15,  # select from top tokens whose probability add up to 15%
                #top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                #max_new_tokens=20,  # max number of tokens to generate in the output
                #repetition_penalty=1.1  # if output begins repeating increase
                )

        self.eos_token = self.tokenizer.eos_token_id

    def query(self, prompt:str, max_tokens: int=20):

        res = self.generate_text(prompt, do_sample=True, max_new_tokens=max_tokens, pad_token_id=self.eos_token)[0]['generated_text']
        
        return res

class Mod8Quant():
    
    def __init__(self, model_path: str):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)

        self.generate_text = transformers.pipeline(
                model=self.model, tokenizer=self.tokenizer,
                return_full_text=False,  # if using langchain set True
                task="text-generation",
                # we pass model parameters here too
                #temperature=0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                #top_p=0.15,  # select from top tokens whose probability add up to 15%
                #top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                #max_new_tokens=20,  # max number of tokens to generate in the output
                #repetition_penalty=1.1  # if output begins repeating increase
                )

        self.eos_token = self.tokenizer.eos_token_id

    def query(self, prompt: str, max_tokens: int=20):

        res = self.generate_text(prompt, do_sample=True, max_new_tokens=max_tokens, pad_token_id=self.eos_token)[0]['generated_text']

        return res

class Mod4Quant():
    
    def __init__(self, model_path: str):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)

        self.generate_text = transformers.pipeline(
                model=self.model, tokenizer=self.tokenizer,
                return_full_text=False,  # if using langchain set True
                task="text-generation",
                # we pass model parameters here too
                #temperature=0.8,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                #top_p=0.15,  # select from top tokens whose probability add up to 15%
                #top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                #max_new_tokens=20,  # max number of tokens to generate in the output
                #repetition_penalty=1.1  # if output begins repeating increase
                )

        self.eos_token = self.tokenizer.eos_token_id

    def query(self, prompt: str, max_tokens: int=20):

        res = self.generate_text(prompt, do_sample=True, max_new_tokens=max_tokens, pad_token_id=self.eos_token)[0]['generated_text']

        return res

class VLLMMod():

    def __init__(self, model_path: str, quant: int=0):
        # choosing the large language model
        if quant == 4:
            self.model = LLM(model=model_path, quantization='AWQ')
        else:
            self.model = LLM(model=model_path)


    def query(self, i: int, prompt: str, max_tokens: int=20):

        # setting the parameters
        self.sampling_params = SamplingParams(max_tokens = max_tokens)

        # generating the answer
        res = self.model.generate(prompt, self.sampling_params)

        # getting the generated text out from the answer variable
        return res[0].outputs[0].text
