from subprocess import call    
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import sys, getopt

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

def install_requirements():
    call(['python3', '-m', 'venv', 'llm-env'])
    call(['source', 'llm-env/bin/activate'])
    call(['pip3', 'install', '-r', 'requirements.txt'])

def download_mod(hf_path, models_dir):

    dest_path = models_dir + hf_path
    call(['mkdir', dest_path])
    call(['huggingface-cli', 'download', hf_path, '--local-dir', dest_path, '--local-dir-use-symlinks', 'False'])


def download(argv):
    
    single_mod = False
    hf_path = ''

    #Parses command line arguments
    opts, args = getopt.getopt(argv, 'hv:m:',['vol_dir', 'hf_path='])
    for opt, arg, in opts:
        if opt == '-h':
            print('downloadModels.py -v <vol_dir> -m <hf_path>')
            print('Only use -m if you want to download a single specific model')
            sys.exit()
        elif opt in ('-v', '--vol_dir'):
            vol_dir = arg
        elif opt in ('-m', '--hf_path'):
            single_mod = True
            hf_path = arg

    if single_mod:
        download_mod(hf_path, models_dir)

    install_requirements()
    
    mixtral_hf = 'mistralai/Mixtral-8x7B-v0.1'
    models_dir = vol_dir + '/models/'
    
    download_mod(mixtral_hf, models_dir)
    
    call(['mkdir', models_dir + 'Quantized/'])
    quantize_mod(models_dir + mixtral_hf, models_dir + 'Quantized/Mixtral-8x7B-4Q')




if __name__ == '__main__':
    download(sys.argv[1:])
