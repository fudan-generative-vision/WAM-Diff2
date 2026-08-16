# WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA

<p align="center">
  Zhihao Zhu<sup>1,*</sup> · Hanlin Shang<sup>1,*</sup> · Mingwang Xu<sup>1,*</sup> ·
  Feipeng Cai<sup>2,*</sup> · Zhuolin He<sup>1</sup> · Yaoyi Li<sup>2</sup> ·
  Jianhua Han<sup>2</sup> · Hang Xu<sup>2</sup> · Siyu Zhu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Fudan University &nbsp;&nbsp; <sup>2</sup>Yinwang Intelligent Technology Co., Ltd.
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href="https://github.com/fudan-generative-vision/WAM-Diff2"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="GitHub"></a>
  <a href="https://arxiv.org/abs/2608.01035"><img src="https://img.shields.io/badge/arXiv-2608.01035-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/fudan-generative-ai"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface" alt="Hugging Face"></a>
</p>

## News

- **2026-08-16** — Released the initial training and inference code for WAM-Diff2.
- **2026-08-02** — Released WAM-Diff2 on arXiv.

## Roadmap

- [x] Release progressive block-wise adaptation code.
- [x] Release hierarchical distillation code and configurations.
- [x] Release GPU and NPU inference entry points.
- [ ] Release the WAM-Diff2 Block-32 checkpoint on Hugging Face.
- [ ] Release benchmark evaluation code and reproducible paper results.
- [ ] Release optimized FlashInfer and CUDA Graph inference.

## Quick Start

### Installation

Python 3.11–3.12 is supported. The reference environment uses PyTorch 2.8 and
CUDA 12.x.

```bash
git clone https://github.com/fudan-generative-vision/WAM-Diff2.git
cd WAM-Diff2

conda create -n wam-diff2 python=3.12 -y
conda activate wam-diff2
pip install -r environment/requirements_cuda.txt
pip install -e .
```

The default configuration uses PyTorch SDPA. FlashAttention is optional and must
match the installed PyTorch and CUDA versions.

### Model checkpoint

Download the WAM-Diff2 checkpoint from Hugging Face and place it at:

```text
checkpoints/WAM-Diff2-B32-2B/
```

## Inference

Inference defaults to Block-32 decoding with 32 denoising steps and dynamic
low-confidence remasking at a confidence threshold of 0.9.

Run inference on NVIDIA GPUs:

```bash
scripts/infer_gpu.sh \
  --model_id checkpoints/WAM-Diff2-B32-2B \
  --input_file /path/to/eval.json \
  --output_file outputs/predictions.json
```

Run inference on Ascend NPUs with a compatible `torch-npu` installation:

```bash
scripts/infer_npu.sh \
  --model_id checkpoints/WAM-Diff2-B32-2B \
  --input_file /path/to/eval.json \
  --output_file outputs/predictions.json
```

Set `NUM_GPUS` or `NUM_DEVICES` to launch multiple workers. The decoding
defaults can be overridden with `--block_size`, `--denoising_steps`,
`--remasking_strategy`, and `--confidence_threshold`.

## Training

### Data

<details>
<summary>Training data sample</summary>

A training JSON file is a list of samples in the following format:

```json
[
  {
    "datasource": "Navsim",
    "id": "43ebecce459052f2",
    "image": [
      "data/opensource/navsim_dataset/sensor_blobs/test/2021.05.25.14.16.10_veh-35_01690_02183/CAM_F0/d28200c994c15229.jpg"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "Here is a front-view images from a driving vehicle: <image>\nThe navigation information is: right\nThe current position is (0.00,0.00)\nThe current velocity is: (4.66,-0.09) and current accelerate is: (0.17,-1.53)\nInstruction: Based on the visual motion cues from the image (such as the relative speed of other vehicles and the changing distance to the intersection) and the provided telemetry, predict the optimal driving action for the next 4 seconds with 8 new waypoints."
      },
      {
        "from": "gpt",
        "value": "2.26,-0.17,4.43,-0.70,6.52,-1.55,8.48,-2.64,10.28,-3.88,11.99,-5.25,13.76,-6.83,15.59,-8.59"
      }
    ]
  }
]
```

</details>

### Progressive block-wise adaptation

Run Block-4 adaptation from the prepared Qwen3-VL checkpoint on one GPU:

```bash
CONFIG_PATH=configs/training/block4.yaml scripts/train_block.sh
```

For multi-GPU training with explicit data paths:

```bash
NUM_GPUS=8 \
CONFIG_PATH=configs/training/block4.yaml \
scripts/train_block.sh \
  --dataset.path_or_dataset /path/to/train.json
```

The tutorial recipe is defined in
[`configs/training/block4.yaml`](configs/training/block4.yaml). Any
`--section.key value` argument overrides the corresponding YAML field.

Continue the progressive adaptation with
[`block8.yaml`](configs/training/block8.yaml),
[`block16.yaml`](configs/training/block16.yaml), and
[`block32.yaml`](configs/training/block32.yaml).

### Hierarchical distillation

Launch block-wise/model-wise distillation with Accelerate and DeepSpeed:

```bash
NUM_GPUS=8 scripts/train_distil.sh
```

The experiment and distributed-training settings are defined in
[`configs/distillation/distil.yaml`](configs/distillation/distil.yaml) and
[`configs/accelerate/`](configs/accelerate/), respectively. They can also be
selected explicitly:

```bash
NUM_GPUS=8 \
EXPERIMENT_CONFIG=/path/to/distil.yaml \
ACCELERATE_CONFIG=/path/to/deepspeed_zero2.yaml \
scripts/train_distil.sh
```

## Citation

If WAM-Diff2 is useful for your research, please cite:

```bibtex
@article{zhu2026wam,
  title={WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA},
  author={Zhu, Zhihao and Shang, Hanlin and Xu, Mingwang and Cai, Feipeng and He, Zhuolin and Li, Yaoyi and Han, Jianhua and Xu, Hang and Zhu, Siyu},
  journal={arXiv preprint arXiv:2608.01035},
  year={2026}
}
```

## Acknowledgements

We sincerely thank the teams behind
[NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) and
[Bard-VL](https://github.com/fudan-generative-vision/Bard-VL). This project builds
upon their excellent open-source work.
