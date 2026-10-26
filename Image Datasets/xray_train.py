"""Compatibility launcher; the shared workflow lives in train.py."""
from train import cli

if __name__ == "__main__":
    cli("xray")
