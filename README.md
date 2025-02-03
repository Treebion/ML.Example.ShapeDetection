# How to run for yourself

**N.B. the web service will only run on Windows due to using System.Drawing.Common**

## If you intend to you a GPU to train
If you want to use a GPU to train your model (it will definitely go faster), then you need to follow the instructions [here]([https://learn.microsoft.com/en-gb/dotnet/machine-learning/how-to-guides/install-gpu-model-builder](https://learn.microsoft.com/en-gb/dotnet/machine-learning/how-to-guides/install-gpu-model-builder) 

## Python Script
The python script can be used to generate images to both test and train the object detection model.

You'll need to install the following packages:

``` bash
pip install numpy pillow
```
Then run the script with:

``` bash
python3 GenerateData.py
```

This will create images in `output-1/train_images` and `output-1/test_images` relative to the where you ran the script and will create a COCO json file in `output-1coco_annotations.json`. You will need the fullpaths for these in the next steps.

## Training
You will need to update the `appsettings.json` file in the `ML.NET Shape Detection` project:

``` json
{
  "exclude": [
    "**/bin",
    "**/bower_components",
    "**/jspm_packages",
    "**/node_modules",
    "**/obj",
    "**/platforms"
  ],
  "modelTraining": {
    "useGPU": true,
    "imgFolder": "Path to where your images are stored",
    "cocoFile": "Path to where your COCO json file stored",
    "modelFile": "Path to where you want to save the trained model"
  }
}
```
You need to set:

* `useGPU` - set to `true` if you have set up the GPU or `false` if not.
* `imgFolder` - set to the full path to the training images. 
* `cocoFile` - set to the path to the COCO json file.
* `modelFile` - set to the path where you want to store the model e.g. `"C:\ML-models\object-detection.mlnet"` 

Set the ML.Net.ShapeDetection as the start up project and run. Depending on your GPU/CPU it can take several minutes to train the model.

## Evaluatin
Once your model has been saved you will need to update the `appsettings.json` file in the `ShapeDetection.API` project:

``` json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "AllowedHosts": "*",
  "modelTraining": {
    "useGPU": true,
    "modelFile": "Path to where you saved the trained model"
  }
}
```
You will need to set

* `useGPU` - set to `true` if you have set up the GPU or `false` if not.
* `modelFile` - set to the path where you want to store the model e.g. `"C:\ML-models\object-detection.mlnet"` 

Then set `ShapeDetection.API` as the start up project and run. It will open a swagger page where you can use the test images from the python stage to evaluate the model.
