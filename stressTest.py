from modelSetup import *
import threading
import sys, getopt
import pandas as pd


def get_query(mod, i, prompt, do_print=True):
    res, query_time = mod.query(i, prompt, do_print)
    generation_data.append({'index':i, 'response':res, 'query_time':query_time})

def stress_test(argv):
    global generation_data
    generation_data = []
	
    mod_dir = '/mnt/MixtralVolume/Mixtral-8x7B-Instruct-v0.1/'
    quantization = '16b'
    prompt_dir = ''

    #Parses command line arguments
    opts, args = getopt.getopt(argv, 'hm:q:p:',['model_dir=','quantization=','prompt_dir='])
    for opt, arg, in opts:
        if opt == '-h':
            print('stress_test.py -m <model_dir> -q <quantization> -p <prompt_dir>')
            sys.exit()
        elif opt in ('-m', '--model'):
            mod_dir = arg
        elif opt in ('-q', '--quantization'):
            quantization = arg
        elif opt in ('-p', '--prompt_dir'):
            prompt_dir = arg

    #Creates model
    mod = ModLoader(mod_dir, quantization)
    
    #Loads prompts

    if prompt_dir == '':
        prompts = [
            'Tell me about cheese.',
            'Teach me about AI.',
            'Build a business.'
            ]
    else:
        prompt_df = pd.read_csv(prompt_dir)
        prompts = prompt_df['prompts']
    

    #Loops through each prompt
    for i, prompt in enumerate(prompts):
        #threading.Timer(0.5*i, mod.threaded_query_print, args=(i, prompt,)).start()
        get_query(mod, i, prompt, do_print=True)

    if prompt_dir != '':
        generation_df = pd.DataFrame(generation_data)
        prompt_df['index'] = prompt_df.index
        merged_df = pd.merge(prompt_df, generation_df, left_on='index', right_on='index')
        
        merged_df.to_csv(mod_dir + 'metric_data.csv')

if __name__ == '__main__':
    stress_test(sys.argv[1:])
