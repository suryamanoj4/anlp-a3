import os

from dotenv import load_dotenv

load_dotenv()


def wandb_key():
    return os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY") or None


def wandb_project(default="anlp-a3-part2"):
    return os.environ.get("WANDB_PROJECT") or default


def wandb_entity():
    return os.environ.get("WANDB_ENTITY") or None


def apply_wandb_key():
    key = wandb_key()
    if key:
        os.environ["WANDB_API_KEY"] = key
    return key is not None
