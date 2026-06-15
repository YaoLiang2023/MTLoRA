# Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning

This repository provides the source code for the paper:

**Matrix-Transformation based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning**  
Yao Liang, Yuwei Wang, Yang Li, Yi Zeng  
Published in **Neural Networks**, 199:108642, 2026  
DOI: https://doi.org/10.1016/j.neunet.2026.108642  
Preprint: https://arxiv.org/abs/2403.07440

MTLoRA is a plug-compatible extension of LoRA for parameter-efficient fine-tuning. Instead of using the standard LoRA update

$$
\Delta W = BA,
$$

MTLoRA introduces a learnable transformation matrix inside the low-rank subspace:

$$
\Delta W = BTA,
$$

where \(T\) reshapes the low-rank coordinates through rotations, scalings, shears, metric learning, staged mixing, or parallel superposition. LoRA is recovered when \(T = I\).

This codebase is organized to support the experiments reported in the published Neural Networks article, including:

- **Natural Language Understanding (NLU)** experiments on GLUE.
- **Natural Language Generation (NLG)** experiments on E2E, WebNLG, and DART.
- **Multimodal visual instruction tuning** experiments based on LLaVA-1.5-7B.

The NLU and NLG implementations were developed on top of the original LoRA codebase and were extended to support MTLoRA. The multimodal part follows the LLaVA/DoRA visual instruction tuning pipeline and integrates MTLoRA variants for vision-language fine-tuning.

---

## 1. Method Overview

MTLoRA inserts a lightweight transformation matrix \(T\) into the low-rank update path. This adds only a small \(O(r^2)\) parameter and computation overhead while enabling the low-rank subspace to learn task-adaptive geometry.

This repository includes four MTLoRA structures:

| Variant | Transformation | Intuition |
| --- | --- | --- |
| **SHIM** | \(T = C\) | Free linear change of basis in the low-rank subspace. |
| **ICFM** | \(T = CC^\top\) | Positive-semidefinite metric / intrinsic correlation filtering. |
| **CTCM** | \(T = CD\) | Staged composition of two transformations. |
| **DTSM** | \(T = C + D\) | Dual-path superposition for parallel subspace mixing. |

### Practical initialization tip

In practice, users may also try the following initialization strategy for MTLoRA: initialize the transformation matrix \(T\) as an identity matrix for all four structures, initialize the \(A\) matrix with Gaussian initialization, and initialize the \(B\) matrix as a zero matrix. This initialization preserves the initial behavior of the frozen backbone at the beginning of fine-tuning while allowing the transformation matrix to gradually learn task-adaptive subspace geometry. We have found that this strategy can often lead to better empirical performance, although the final effect may depend on the backbone, task, rank, learning rate, and random seed.

---

## 2. Repository Structure

The repository is organized as follows:

```text
MTLoRA-main/
├── NLU/                         # Natural language understanding experiments
├── NLG/                         # Natural language generation experiments
├── visual_instruction_tuning/   # Multimodal LLaVA-1.5 visual instruction tuning
├── transformation_matrix/       # MTLoRA transformation structures
│   ├── SHIM/
│   ├── ICFM/
│   ├── CTCM/
│   └── DTSM/
└── README.md
```

The three experimental directories, `NLU/`, `NLG/`, and `visual_instruction_tuning/`, are placed at the same level under the repository root.

---

## 3. Selecting an MTLoRA Structure

For the NLU and NLG experiments, choose one MTLoRA structure from `transformation_matrix/` and replace the corresponding `xtuning_layers.py` file in the target task directory.

For example, to use CTCM, copy:

```text
transformation_matrix/CTCM/xtuning_layers.py
```

to the following locations as needed:

```text
NLU/src/xtuninglib/xtuning_layers.py
NLU/src/transformers/models/roberta/xtuninglib/xtuning_layers.py
NLG/src/xtuninglib/xtuning_layers.py
```

Use the analogous file under `transformation_matrix/SHIM/`, `transformation_matrix/ICFM/`, or `transformation_matrix/DTSM/` to run the other variants.

---

# Natural Language Understanding Tasks

The NLU experiments are located in:

```bash
NLU/
```

## 4. NLU Environment Setup

Enter the NLU directory and create the environment:

```bash
cd NLU
conda env create -f environment.yml
conda activate mtlora_nlu  # or the environment name defined in environment.yml
pip install -e .
```

## 5. Running NLU Experiments

The GLUE scripts include:

```bash
bash roberta_base_cola.sh
bash roberta_base_mnli.sh
bash roberta_base_mrpc.sh
bash roberta_base_qnli.sh
bash roberta_base_qqp.sh
bash roberta_base_rte.sh
bash roberta_base_sst2.sh
bash roberta_base_stsb.sh
```

For **MRPC**, **RTE**, and **STS-B**, start from the MTLoRA-adapted MNLI checkpoint and modify the checkpoint path in the corresponding shell script.

---

# Natural Language Generation Tasks

The NLG experiments are located in:

```bash
NLG/
```

## 6. NLG Environment Setup

Enter the NLG directory and install dependencies:

```bash
cd NLG
conda create -n mtlora_nlg python=3.8.13 -y
conda activate mtlora_nlg
pip install -r requirement.txt
bash download_pretrained_checkpoints.sh
bash create_datasets.sh
cd ./eval
bash download_evalscript.sh
cd ..
```

The NLG directory contains:

```text
data/      # Raw and processed datasets
src/       # Training, decoding, and generation scripts
eval/      # Evaluation scripts
vocab/     # GPT-2 vocabulary files
```

## 7. E2E NLG

### 7.1 Train GPT-2 Medium with MTLoRA

```bash
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --xtuning_dim 4 \
    --xtuning_alpha 32 \
    --xtuning_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e_110 \
    --random_seed 110 \
    > e2e_110_train.log 2>&1 &
```

### 7.2 Generate Outputs with Beam Search

```bash
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29602 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e_110/model.26290.pt \
    --platform local \
    --xtuning_dim 4 \
    --xtuning_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e_110 \
    --output_file predict.26290.b10p08r4.jsonl \
    > e2e_110_beam_search.log 2>&1 &
```

### 7.3 Decode

```bash
nohup python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e_110/predict.26290.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_110_ref.txt \
    --output_pred_file e2e_110_pred.txt \
    > e2e_110_decode.log 2>&1 &
```

### 7.4 Evaluate

```bash
nohup python eval/e2e/measure_scores.py e2e_110_ref.txt e2e_110_pred.txt -p \
    > e2e_110_evaluation.log 2>&1 &
```

## 8. WebNLG

Follow the E2E training and beam-search pipeline, replacing the E2E data paths and output directories with WebNLG paths. Refer to the paper and scripts for the exact hyperparameters.

### 8.1 Decode WebNLG Outputs

```bash
nohup python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg_110/predict.11270.b10p08r4.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg_110 \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg_110 \
    --tokenize --lower \
    > webnlg_110_decode.log 2>&1 &
```

### 8.2 Evaluate WebNLG

```bash
cd ./eval/GenerationEval/
nohup python eval.py \
    -R data/references_webnlg_110/reference \
    -H data/hypothesis_webnlg_110 \
    -nr 6 \
    -m bleu,meteor,ter \
    > webnlg_110_evaluation.log 2>&1 &
cd ../..
```

## 9. DART

Follow the E2E training and beam-search pipeline, replacing the E2E data paths and output directories with DART paths. Refer to the paper and scripts for the exact hyperparameters.

### 9.1 Decode DART Outputs

```bash
nohup python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/dart_110/predict.39165.b10p08r4.jsonl \
    --input_file ./data/dart/test_formatted.jsonl \
    --ref_type dart \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_dart_110 \
    --output_pred_file eval/GenerationEval/data/hypothesis_dart_110 \
    --tokenize --lower \
    > dart_110_decode.log 2>&1 &
```

### 9.2 Evaluate DART

```bash
cd ./eval/GenerationEval/
nohup python eval.py \
    -R data/references_dart_110/reference \
    -H data/hypothesis_dart_110 \
    -nr 6 \
    -m bleu,meteor,ter \
    > dart_110_evaluation.log 2>&1 &
cd ../..
```

---

# Multimodal Visual Instruction Tuning

The multimodal experiments are located in:

```bash
visual_instruction_tuning/
```

This part reproduces the MTLoRA experiments with **LLaVA-1.5-7B**. The pipeline follows the visual instruction tuning setup used by DoRA and LLaVA, while the local `peft/` package and training scripts enable the four MTLoRA structures.

## 10. Multimodal Environment Setup

Create and activate the conda environment:

```bash
conda create -n mtlora_llava python=3.10 -y
conda activate mtlora_llava
```

Install the visual instruction tuning code and training dependencies:

```bash
cd visual_instruction_tuning
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.3.6 --no-build-isolation
pip install -e ./peft
pip install "numpy<2"
```

Notes:

- The multimodal environment follows the DoRA visual instruction tuning instructions: https://github.com/NVlabs/DoRA/tree/main/visual_instruction_tuning
- `flash-attn` is sensitive to CUDA, PyTorch, and compiler versions. If installation fails, verify that your local CUDA/PyTorch stack is compatible with `flash-attn==2.3.6`.
- The local `./peft` package should be installed after the main package so that the MTLoRA-specific PEFT implementation is used.

## 11. Multimodal Data Preparation

All paths below are relative to:

```bash
visual_instruction_tuning/
```

### 11.1 Pretrained Projector Weights

Download the pretrained projector weights from:

```text
https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/tree/main
```

Place the downloaded checkpoint under:

```text
./checkpoints/
```

### 11.2 Instruction-Tuning Data

Download the LLaVA v1.5 mixture annotation file:

```text
https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json
```

Then download the images from the constituent datasets:

| Dataset | Download source |
| --- | --- |
| COCO train2017 | http://images.cocodataset.org/zips/train2017.zip |
| GQA images | https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip |
| OCR-VQA | https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing |
| TextVQA train/val images | https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip |
| Visual Genome part 1 | https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip |
| Visual Genome part 2 | https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip |

For OCR-VQA, save all image files as `.jpg`.

After downloading and extracting all files, organize the training data as follows:

```text
playground/data/
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100
```

### 11.3 Evaluation Data

First download `eval.zip`:

```text
https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
```

Extract it to:

```text
./playground/data/eval
```

Then prepare each evaluation benchmark as follows.

#### VQAv2

Download `test2015`:

```text
http://images.cocodataset.org/zips/test2015.zip
```

Place it under:

```text
./playground/data/eval/vqav2
```

#### GQA

Download the GQA data and evaluation scripts following the official instructions:

```text
https://cs.stanford.edu/people/dorarad/gqa/download.html
https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
```

Place them under:

```text
./playground/data/eval/gqa/data
```

If needed, modify `eval.py` according to the following patch because of missing assets in the GQA v1.2 release:

```text
https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187
```

#### VisWiz

Download `test.json` from:

```text
https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
```

Download and extract `test.zip` to `test`:

```text
https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip
```

Place the files under:

```text
./playground/data/eval/vizwiz
```

#### ScienceQA

Under:

```text
./playground/data/eval/scienceqa
```

download the following files/folders from the `data/scienceqa` directory of the ScienceQA repository:

```text
https://github.com/lupantech/ScienceQA
```

Required files:

```text
images/
pid_splits.json
problems.json
```

#### TextVQA

Download the annotation file:

```text
https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
```

Download and extract the images:

```text
https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
```

Place them under:

```text
./playground/data/eval/textvqa
```

#### POPE

Download the `coco` folder from POPE:

```text
https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco
```

Place it under:

```text
./playground/data/eval/pope
```

#### MMBench

Download:

```text
https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
```

Place it under:

```text
./playground/data/eval/mmbench
```

## 12. Multimodal Training

Enter the multimodal directory:

```bash
cd visual_instruction_tuning
```

Run the four MTLoRA variants. The following commands use one GPU per variant and set Weights & Biases to offline mode:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 bash ./SHIMLora_7b_bs16_r128_alpha256_seed47.sh
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 bash ./ICFMLora_7b_bs16_r128_alpha256_seed47.sh
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 bash ./CTCMLora_7b_bs16_r128_alpha256_seed47.sh
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 bash ./DTSMLora_7b_bs16_r128_alpha256_seed47.sh
```

The example scripts correspond to:

- Backbone: LLaVA-1.5-7B.
- Batch size setting: `bs16`.
- Rank: `r128`.
- Scaling factor: `alpha256`.
- Random seed: `seed47`.

The fine-tuned MTLoRA checkpoints are expected to be saved under:

```text
./checkpoints/
```

For different hardware environments, modify the training scripts to adjust GPU IDs, batch size, gradient accumulation, output directories, and checkpoint paths.

## 13. Multimodal Evaluation

The evaluation script takes five arguments:

```text
bash 7B_eval_mtlora.sh \
  <checkpoint_folder_name> \
  <absolute_path_to_checkpoint_folder> \
  <absolute_path_to_eval_dataset_root> \
  <absolute_path_to_output_result_folder> \
  <absolute_path_to_visual_instruction_tuning_root>
```

Set the absolute path to your local `visual_instruction_tuning` directory:

```bash
export VIS_DIR=/mnt/home/code/MTLoRA-main/visual_instruction_tuning
```

Then evaluate each MTLoRA variant:

```bash
CUDA_VISIBLE_DEVICES=0 bash 7B_eval_mtlora.sh \
  shimlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/checkpoints/shimlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/playground/data/eval \
  ${VIS_DIR}/eval_result/shimlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR} \
  > shimlora-bs16-r128-alpha-256-seed47-eval.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash 7B_eval_mtlora.sh \
  icfmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/checkpoints/icfmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/playground/data/eval \
  ${VIS_DIR}/eval_result/icfmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR} \
  > icfmlora-bs16-r128-alpha-256-seed47-eval.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 bash 7B_eval_mtlora.sh \
  ctcmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/checkpoints/ctcmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/playground/data/eval \
  ${VIS_DIR}/eval_result/ctcmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR} \
  > ctcmlora-bs16-r128-alpha-256-seed47-eval.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 bash 7B_eval_mtlora.sh \
  dtsmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/checkpoints/dtsmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR}/playground/data/eval \
  ${VIS_DIR}/eval_result/dtsmlora-bs16-r128-alpha-256-seed47 \
  ${VIS_DIR} \
  > dtsmlora-bs16-r128-alpha-256-seed47-eval.log 2>&1 &
```

To evaluate a different checkpoint, update:

1. the checkpoint folder name;
2. the checkpoint path under `${VIS_DIR}/checkpoints/`;
3. the output path under `${VIS_DIR}/eval_result/`;
4. the log filename.

## 14. Submitting VQAv2 and MMBench Results

Some benchmarks require external submission.

### 14.1 VQAv2

Submit to EvalAI:

```text
https://eval.ai/web/challenges/challenge-page/830/my-submission
```

Choose:

```text
Test-Dev Phase
```

Upload the contents of:

```text
ABSOLUTE_PATH/visual_instruction_tuning/eval_result/<checkpoint_name>/vqav2/answers_upload
```

### 14.2 MMBench

Submit to the MMBench submission system:

```text
https://mmbench.opencompass.org.cn/mmbench-submission
```

Upload the contents of:

```text
ABSOLUTE_PATH/visual_instruction_tuning/eval_result/<checkpoint_name>/mmbench/answers_upload
```

---

## 15. Main Reported Results

In the published Neural Networks paper, MTLoRA is evaluated across NLU, NLG, and multimodal tasks. The main reported findings include:

- On GLUE with DeBERTaV3-base, MTLoRA improves the average score over LoRA by approximately 2.0 points and matches AdaLoRA without requiring a pruning schedule.
- On NLG with GPT-2 Medium, MTLoRA improves BLEU on DART by 0.95 and on WebNLG by 0.56 compared with LoRA.
- On multimodal visual instruction tuning with LLaVA-1.5-7B, DTSM achieves the best average score of 69.91 with about 4.7% trainable parameters, outperforming full fine-tuning and strong PEFT baselines on average.

These results support the central claim that learning geometry inside the low-rank subspace can improve parameter-efficient fine-tuning without increasing the adapter rank.

---

## 16. Reproducibility Notes

- The commands above reproduce the main experimental workflow. Exact scores may vary slightly because of hardware, CUDA/PyTorch versions, random seeds, dataset versions, and external evaluation servers.
- This repository does not redistribute third-party datasets, pretrained backbones, or benchmark annotations unless explicitly permitted by their original licenses. Please download them from the official sources listed above.
- For fair comparison, keep the adaptation scope, rank, batch size, seeds, and evaluation protocol consistent across LoRA, DoRA, AdaLoRA, and MTLoRA variants.
- When reporting results, specify the MTLoRA structure, rank, scaling factor, seed, backbone, and benchmark version.

---

## 17. Acknowledgements

This work was supported by the National Science and Technology Major Project under Grant No. 2022ZD0116202.

The NLU and NLG code in this repository was developed based on the original LoRA codebase. We sincerely thank the LoRA authors and maintainers for releasing their implementation, which provided an important foundation for our MTLoRA implementation and experiments.

The multimodal experiments build on the open-source LLaVA and DoRA visual instruction tuning ecosystems. We thank the authors and maintainers of these projects, as well as the creators of the datasets and benchmarks used in this work.

---

## 18. Contact

Please open a GitHub issue or contact the authors if you have questions.

- Yao Liang: liangyao2023@ia.ac.cn
- Yuwei Wang: yuwei.wang@ia.ac.cn
- Yang Li: liyang2019@ia.ac.cn
- Yi Zeng: yi.zeng@ia.ac.cn

---

## 19. Citation

If you find this repository useful, please cite the published Neural Networks paper:

```bibtex
@article{liang2026matrix,
  title     = {Matrix-Transformation based Low-Rank Adaptation (MTLoRA): A brain-inspired method for parameter-efficient fine-tuning},
  author    = {Liang, Yao and Wang, Yuwei and Li, Yang and Zeng, Yi},
  journal   = {Neural Networks},
  volume    = {199},
  pages     = {108642},
  year      = {2026},
  doi       = {10.1016/j.neunet.2026.108642},
  issn      = {0893-6080},
  publisher = {Elsevier}
}
```
