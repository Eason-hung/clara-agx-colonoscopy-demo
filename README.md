# Clara AGX colonoscopy demo
The demo demonstrates the development sample of realtime colonoscopy polyp segmentation on Clara AGX devkit. The model is a Resnet101-based Unet for polyp segmentation, which trained with Nvidia TLT SDK, the performance on Deepstream (fp16) is ~23 FPS on Xavier iGPU, ~105 FPS on Clara AGX Xavier dGPU (RTX6000).
## Prerequired
1. Ubuntu Host with nvidia GPU for training models with Nvidia Transfer Learning Toolkit, recommended HW please see https://docs.nvidia.com/tlt/tlt-user-guide/text/tlt_quick_start_guide.html#hardware.
2. Nvidia TLT, following the link for installation https://docs.nvidia.com/tlt/tlt-user-guide/text/tlt_quick_start_guide.html#software-requirements.
3. Clara AGX devkit or Jetson AGX Xavier devkit
4. Clara AGX SDK for Clara AGX devkit / Jetpack 4.5.1 for Jetson AGX Xavier devkit
## Dataset
1. Training dataset: Kvasir-SEG Dataset. https://datasets.simula.no/kvasir-seg/
2. Testing video: http://www.depeca.uah.es/colonoscopy_dataset/
## Training models with Nvidia TLT 3.0 (the steps here need to be run on Ubuntu host)
### Downloading dataset
1. Download and unzip the dataset in https://datasets.simula.no/kvasir-seg/.
2. Run the preprocessing code
```
  $ python3 data_preprocess.py
```
3. If you are going to train with your own dataset, please follow the annotation guide https://docs.nvidia.com/tlt/tlt-user-guide/text/data_annotation_format.html#structured-images-and-masks-folders
### Training model in TLT
1. You will need to install Nvidia Transfer Learning Toolkit first, please see the product page https://developer.nvidia.com/transfer-learning-toolkit
2. Please follow the steps in the jupyter notebook unet_Kvasir_SEG, you will need to set some env variables (such as wokspace path, data path).
3. You will export your model file as .etlt file at the end of the notebook, then we are going to deploy the file on Xavier with deepstream. 

## Inferencing the models with Nvidida Deepstream 5.1 (the steps here need to be run on Clara AGX Xavier/Jetson Xavier)
![image](https://github.com/Eason-hung/clara-agx-colonoscopy-demo/blob/main/pipeline.JPG)
### Building models
Currently, TLT Unet model needs to be converted to TRT engine in order to run deepstream apps.
1. Download tlt-converter https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/overview.html
2. run tlt-converter to convert .etlt file to tensorrt engine
```
  $ ./tlt-converter -k your_model_key -e output_model_name -t fp16 -p input_1,1x3x320x320,4x3x320x320,16x3x320x320  input_etle_file
```
### Set path and properties:
set the following properties in the code before compiling
1. set engine path and label file in pgie_unet_tlt_config.txt
2. set SAVE_VIDEO in C++ code to set output option
### Compile sample code
```
  $ cd apps/deepstream-colonoscopy/
  $ make
```
### Run sample code
```
  $ ./deepstream-colonoscopy-app <uri>
```
e.g.
```
  $ ./deepstream-colonoscopy-app file:///home/nvidia/deepstream-colonoscopy/WL3.mp4
```
## Inferencing models in DGPU mode (Clara AGX Xavier)

NVIDIA Clara AGX Developer Kit includes an NVIDIA RTX 6000 GPU, user can swtich between igpu (in Jetson Xavier) and dgpu (RTX 6000) depends on the usecase.
More information about NVIDIA Clara AGX Developer Kit, please refer to https://developer.nvidia.com/clara-agx-devkit.
### swtitching igpu/dgpu mode
Before running deepstream application in dgpu mode, need to run gpu swithcing code.
1. To view the currently installed drivers and their version, use the query command:
```
$ nvgpuswitch.py query
```
2. To install the dGPU drivers, use the install command with the dGPU parameter (note that sudo must be used to install drivers):
```
$ sudo nvgpuswitch.py install dGPU
```
3. reboot
4. The dGPU driver install may be verified once again using the query command:
```
$ nvgpuswitch.py query
```
### Building models
run tlt-converter-dGPU to convert .etlt file to tensorrt engine.
```
  $ ./tlt-converter-dGPU -k your_model_key -e output_model_name -t fp16 -p input_1,1x3x320x320,4x3x320x320,16x3x320x320  input_etle_file
```
### Compile and run sample code
Follow the same steps in igpu mode.
## Reference
* Nvidia Clara AGX Xavier: https://developer.nvidia.com/clara-agx-devkit
* Nvidia deepstream: https://developer.nvidia.com/deepstream-sdk
* Nvidia Transfer Learning Toolkit: https://developer.nvidia.com/transfer-learning-toolkit
* The Kvasir-SEG Dataset: https://datasets.simula.no/kvasir-seg/
* Computer-Aided Classification of Gastrointestinal Lesions in Regular Colonoscopy: http://www.depeca.uah.es/colonoscopy_dataset/
