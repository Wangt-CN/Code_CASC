from vocab import Vocabulary
import evaluation
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
evaluation.evalrank("/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/runs/f30k_scan/log/model_best.pth.tar",
                    data_path="/data6/wangzheng/data", split="test")
