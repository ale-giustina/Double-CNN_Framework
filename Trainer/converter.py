import cv2
import numpy as np
import time
import os
import module1 as md
import torch

model = torch.load('mod/best_model_ep57.ckp')

torch.save(model.state_dict(), 'mod/best_model_ep57_state_dict.ckp')