# Checkpoints

This repository does not ship model weights.

Required files:

- `checkpoints/sv3d_p.safetensors`

Optional local overrides:

- `ZERO123PLUS_MODEL_DIR` or `ZERO123PLUS_LOCAL_DIR`: local snapshot for `sudo-ai/zero123plus-v1.2`
- `INSTANTMESH_UNET_PATH` or `ZERO123PLUS_UNET_PATH`: local `diffusion_pytorch_model.bin`

If these variables are not set, the code will try to load cached Hugging Face files and then fall back to downloading them.
