INSTALL : 

conda create --name pixelink --file spec-file.txt
(the spec-file was create with "conda list --explicit > spec-file.txt")

after the environment has been create you should activate it with :
source activate pixelink
  
COMMANDS : 

1) choose config file from "configs" folder (e.g. vgg_2s)
2) you can run one of the following commands :
  a. train from epoch 0 		    : python main.py vgg_2s --train 1 
  b. continue training from latest snapshot : python main.py vgg_2s --retrain 1
  c. run benchmark on training set          : python main.py vgg_2s
  d. run benchmark on test set              : python main.py vgg_2s --test 1

python main.py vgg_2s --command [train]/[retrain]/[benchmark-train]/[benchmark-test]

for c & d commands the epoch chosen is the one defined in the config file (test_model_index)
