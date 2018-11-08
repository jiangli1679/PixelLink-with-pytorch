from configs.resnet50_2s import *

batch_size = 12
all_trains = 48

learning_rate1 = 1e-3  # 0.25e-3  # 1e-3
learning_rate2 = 1e-4  # 1e-2
step2_start = 300

use_crop = True
image_size_test = (720, 1280)

test_model_index = 2000