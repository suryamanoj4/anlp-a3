import os

from dotenv import load_dotenv

load_dotenv()


def hf_token():
    return os.environ.get("HUGGINGFACE_KEY") or os.environ.get("HF_TOKEN") or None


def wandb_key():
    return os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY") or None


def wandb_project(default="anlp-a3"):
    return os.environ.get("WANDB_PROJECT") or default


def wandb_entity():
    return os.environ.get("WANDB_ENTITY") or None


def hf_repo(default=None):
    return os.environ.get("HF_REPO_ID") or default


def roll_number(default="rollnumber"):
    return os.environ.get("ROLL_NUMBER") or default


def apply_wandb_key():
    key = wandb_key()
    if key:
        os.environ["WANDB_API_KEY"] = key
    return key is not None
