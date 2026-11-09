import os

from . import env


def push_checkpoint(path, repo_id=None, path_in_repo=None, token=None):
    from huggingface_hub import HfApi

    repo_id = repo_id or env.hf_repo()
    if not repo_id:
        raise ValueError("no HuggingFace repository configured: set HF_REPO_ID or pass repo_id")
    path_in_repo = path_in_repo or os.path.basename(path)
    api = HfApi(token=token or env.hf_token())
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
    )
    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "url": f"https://huggingface.co/{repo_id}",
    }


def pull_checkpoint(filename, repo_id=None, local_dir="models/part1", token=None):
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or env.hf_repo()
    if not repo_id:
        raise ValueError("no HuggingFace repository configured: set HF_REPO_ID or pass repo_id")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token or env.hf_token(),
    )
