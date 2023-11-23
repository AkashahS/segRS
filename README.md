# SegRS : Segmentation for Remote Sensing

[Akashah Shabbir](https://akashahs.github.io/), Muhammad Ibraheem Siddiqi, Sara Ghaboura

The **segment for Remote Sensing (SegRS)** is automated instance segmentation pipeline utilizing Segment Anything Model SAM and other foundational models (FM), conjointly adapting and maximizing their aptness for remote sensing domain. A mask refinement method combined with prompt-based RSI sampling techniques is designed to significantly enhance cross-domain generalization.

![segRS design](assets/segRS_design.png?raw=true)

The fundamental premise of this project is the fusion of various model capabilities to create an advanced pipeline tailored for complex remote sensing detection and segmentation. This workflow is designed to incorporate robust expert models that can be utilized independently or in conjunction, and are interchangeable with analogous but diverse models.

![segRS design](assets/segRS_example.png?raw=true)

# Features

- Zero-shot text-to-bbox and masks approach for object detection and instance segmentation for remote sensing.
- GroundingDINO detection model, remoteCLIP and Segment anything (SAM) integration.
- Customizable text prompts for precise object segmentation.

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

Clone the repository and install the required packages:

```
git clone https://github.com/AkashahS/SegRS && cd segRS
pip install torch torchvision
pip install -e .
```
Or use Conda
Create a Conda environment from the `environment.yml` file:
```
conda env create -f environment.yml
# Activate the new environment:
conda activate rsenv
```

### Getting Started

Use as a library:

```python
from PIL import Image
from segRS import segRS

model = segRS()
image_pil = Image.open("sample/plane.png").convert("RGB")
text_prompt = "plane"
text_queries = [
    "A plane", 
    "A helicopter", 
    "A ship", 
    "A car",
]
masks, boxes, phrases, logits = model.predict_segRS(image_pil, text_prompt,text_queries)
```

Use with custom checkpoint:

First download a model checkpoint. 

```python
from PIL import Image
from segRS import segRS

model = LangSAM("<SAM_model_type>","<remoteCLIP_model_type>", "<path/to/checkpoint>")
image_pil = Image.open("sample/plane.png").convert("RGB")
text_prompt = "plane"
text_queries = [
    "A plane", 
    "A helicopter", 
    "A ship", 
    "A car",
]
masks, boxes, phrases, logits = model.predict_segRS(image_pil, text_prompt,text_queries)
```

## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)

## License

This project is licensed under the Apache 2.0 License
