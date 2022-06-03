## Implementation of [EfficientNet B0-B7](https://arxiv.org/abs/1905.11946) in PyTorch

**Arxiv**: https://arxiv.org/abs/1905.11946

After 450 epochs EfficientNet-b0 reaches **Acc@1** 76.656 **Acc@5** 93.136 of validation accuracy

### Updates

* 2022.06.03:
    - Training EfficientNet-b0 with RandomAugment augmentation (ongoing...).
    - `RandomAugment` added.

* 2022.06.02:
    - After 450 epochs, EfficientNet-b0 reaches **Acc@1** 76.656 **Acc@5** 93.136 validation accuracy while using
      AutoAugment augmentation with IMAGENET policy.
    - Weights are uploaded to the `weights` folder. `last.ckpt` is checkpoint (81.4MB) (includes model, model_ema,
      optimizer, ...)

### Dataset

Specify the IMAGENET data folder in the `main.py` file.

``` python
parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")
```

IMAGENET folder structure:

```
├── IMAGENET 
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ....
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
```

#### Augmentation:

Random Augmentation [`RandomAugment`](efficientnet/utils/augment.py) in `efficientnet/utils/augment.py` is used as an
augmentation. To check the random augmentation run the `augment.py` file. Interpolation mode for Random Augmentation
randomly chosen from `BILINEAR` and `BICUBIC`. For resizing the input image `BICUBIC` interpolation is used.

### Train

Distributed Data Parallel - `bash main.sh`
`main.sh`:

```
torchrun --nproc_per_node=$num_gpu main.py --epochs 450 --batch-size 320 --model-ema --lr 0.048 --lr-warmup-init 1e-6 --weight-decay 1e-5 --model-ema-decay 0.9999 --interpolation bicubic --random-erase 0.2
```

Data Parallel (without DDP, not recommended) - `python main.py`

To resume the training add `--resume @path_to_checkpoint` to `main.sh`, e.g: `--resume weights/last.ckpt`

The training config taken from [timm's model training config](https://github.com/rwightman/pytorch-image-models)

### Evaluation

To validate the **Acc@1** 76.656 **Acc@5** 93.136(EfficientNet-b0) run the following command:

```
torchrun --nproc_per_node=$num_gpu main.py --interpolation bicubic --resume weights/last.ckpt --test-only
```