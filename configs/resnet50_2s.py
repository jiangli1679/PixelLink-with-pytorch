version = "2s"
net_params = {'backbone': 'resnet50', 'add_extra_block': False}

epoch = 60000
learning_rate1 = 1e-3  # 1e-3
learning_rate2 = learning_rate1  # 1e-2
step2_start = 100

all_trains = 1000
batch_size = 12 #24
momentum = 0.9
weight_decay = 5e-4
dilation = True
use_crop = False
use_rotate = True
# iterations = 10
gpu = True
multi_gpu = False  # only useful when gpu=True
pixel_weight = 2
link_weight = 1

r_mean = 123.
g_mean = 117.
b_mean = 104.
mean = (r_mean, g_mean, b_mean)

image_size_train = (512, 512)
image_size_test = (512, 512)
image_channel = 3

link_weight = 1
pixel_weight = 2
neg_pos_ratio = 3  # parameter r in paper

train_images_dir = "train_images/images/"
train_labels_dir = "train_images/ground_truth/"
retrain_model_index = 26200 # retrain from which model, e.g. ${saving_model_dir}/156600.mdl
test_model_index = 43800 # test for which model, e.g. ${saving_model_dir}/156600.mdl
test_batch = 1

test_images_dir = "test_images/images/"
test_labels_dir = "test_images/ground_truth/"
all_tests = 500
