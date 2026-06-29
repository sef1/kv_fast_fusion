import os
from vllm.entrypoints.cli.main import main as vllm_main
import sys
  
import argparse  
import json

  
def main():  
    vllm_main()  
  
if __name__ == "__main__":  
    main()