## Overview

This is an script for the annotation tool implementation developed for polygonal binary annotation using Segment Anything Model (SAM)'
## Dependencies

Install all the packages in requirements.txt file. The code was executed in a Python 3.10.11 envirionment. The code was run with a Tesla T4 GPU. To use a similar GPU, please ensure you have the appropriate CUDA and cuDNN packages installed.

## Setup
```bash
pip install -r requirements.txt
```

# or
```bash
conda install --file requirements.txt
```


## Usage
Install the Segment anything model by following the instructions from [SAM GitHub repository](https://github.com/facebookresearch/segment-anything) Also Download the pretrained weights from the model checkpoint and save it under `weights/sam_vit_h_4b8939.pth`.

To run the script:

```bash
python ann.py
```
## Instructions for User Interface

1. **Modify the Folders:**
   - Update the respective input and output folders in the `ann.py` file.
   - Run the file.

2. **Open First Frame:**
   - The first frame will open in a separate window.

3. **Annotate Object:**
   - Draw a rough bounding box around the object you wish to annotate.
   - Press the "s" key to run the model.

4. **Review Annotation:**
   - A binary mask for the object will be generated.
   - If you are satisfied with the annotation, press the "s" key again to save it in the output folder.
   - If you are not satisfied, press the "c" key to cancel the bounding box and return to step 3.

5. **Next Frame:**
   - The next frame in the input folder will automatically open.
   - Repeat step 3 until all frames are annotated.
