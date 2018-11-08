INSTALL : 

conda create --name pixelink --file spec-file.txt
(the spec-file was create with "conda list --explicit > spec-file.txt")

after the environment has been create you should activate it with :
source activate pixelink
  
COMMANDS : 

1) choose config file from "configs" folder (e.g. vgg_2s)
2) you can run one of the following commands :
  a. train from epoch 0 		    : python main.py vgg_2s --mode train
  b. continue training from latest snapshot : python main.py vgg_2s --mode retrain
  c. run benchmark on training set          : python main.py vgg_2s --mode benchmark-test --epoch <epoch>
  d. run benchmark on test set              : python main.py vgg_2s --mode benchmark-train --epoch <epoch>
