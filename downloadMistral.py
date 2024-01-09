from huggingface_hub import snapshot_download

print(snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2", local_dir='/workspace/models/Mistral-7B-Instruct-v0.2', local_dir_use_symlinks=False, cache_dir='/workspace/models/Mistral-7B-Instruct-v0.2'))
