## Local model weights directory

This folder is the **default location** for local diffusers-format weights used by OFT training/inference scripts.

### Default expected layout

- `models/stable-diffusion-v1-5/`
- `models/stable-diffusion-xl-base-1.0/`
- `models/stable-diffusion-3.5-medium/`

Each model folder should contain `model_index.json` (standard diffusers layout).

### Override with environment variables

- `SD15_MODEL_DIR`: full path to SD1.5 model dir
- `SDXL_MODEL_DIR`: full path to SDXL Base model dir
- `SD35_MODEL_DIR`: full path to SD3.5 Medium model dir
- `OFT_MODELS_DIR`: a common root; scripts will resolve `${OFT_MODELS_DIR}/<model-subdir>`

