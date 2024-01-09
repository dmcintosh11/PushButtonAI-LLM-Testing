from huggingface_hub import snapshot_download

print(snapshot_download(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", local_dir='/workspace/models', local_dir_use_symlinks=False, cache_dir='/workspace/models'))
