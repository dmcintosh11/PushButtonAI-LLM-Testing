from subprocess import call    
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import sys, getopt
import os



def quantize_mod(model_path, quant_path):
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    #   Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

quantize_mod('/workspace/models/Mixtral-8x7B-Instruct-v0.1', '/workspace/models/Mixtral-8x7B-Instruct-v0.1-QUANT-4/')
