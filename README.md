## Environment
- Python >= 3.7
- PyTorch >= 1.7 
- NVIDIA GPU + CUDA

### Installation
```
pip install -r requirements.txt
python setup.py develop
```
## How To Inference or Test
- Refer to ./options/test for the configuration file of the model to be tested.  
- Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1cslBiWi1UY33pvhvFLjsb6baTP9z1Pud/view?usp=sharing) and put it into ./experiments/UDCUNet_4gpu_pretrained/models .
- Prepare the testing data (validation data) into ./datasets/test/input (./datasets/validation/input and ./datasets/validation/GT)
- All datasets can be downloaded at the [MIPI Challenage official website](https://codalab.lisn.upsaclay.fr/competitions/4874#participate).
- Then run the follwing codes (Inference for example):  

```
python basicsr/test.py -opt options/inference/UDCUNet_inference.yml
```
The testing results will be saved in the ./results folder.

## How To Train
- Refer to ./options/train for the configuration file of the model to train.  
- Prepare the training data in ./datasets/training/input and ./datasets/training/GT 
- All datasets can be downloaded at the [MIPI Challenage official website](https://codalab.lisn.upsaclay.fr/competitions/4874#participate).
- The training command is like  
```
Single GPU: CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/UDCUNet_train.yml
Multi GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/UDCUNet_train.yml --launcher pytorch
```
For more training commands and details, please check the docs in [BasciSR](https://github.com/XPixelGroup/BasicSR)  

## Contact
If you have any question, please email jf.hu1@siat.ac.cn.
