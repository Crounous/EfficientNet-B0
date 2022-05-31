## Implementation of [EfficientNet B0-B7](https://arxiv.org/abs/1905.11946) in PyTorch

**Arxiv**: https://arxiv.org/abs/1905.11946

<!-- After 300 epochs MobileNetV3L reaches **Acc@1**: 74.3025 **Acc@5**: 91.8342
 -->
### Updates

<!-- * 2022.05.13:
    - Weights are uploaded to the `weights` folder. `last.ckpt` is checkpoint (88.3MB) (includes model, model_ema, optimizer, ...) and last.pth is model with
      Exponential Moving Average (11.2MB) and converted to `half()` tensor. -->

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

`AutoAugment` for IMAGENET is used as a default augmentation. The interpolation mode is `BICUBIC`

### Train

Run `main.sh` (for DDP) file by running the following command:

```
bash main.sh
```

`main.sh`:

```
torchrun --nproc_per_node=$num_gpu main.py --epochs 450 --batch-size 320 --model-ema --lr 0.048 --lr-warmup-init 1e-6 --weight-decay 1e-5 --model-ema-decay 0.9999 --interpolation bicubic --auto-augment imagenet --random-erase 0.2
```

To resume the training add `--resume @path_to_checkpoint` to `main.sh`, e.g: --resume weights/last.ckpt 

The training config taken from [official torchvision models' training config](https://github.com/rwightman/pytorch-image-models)


### Evaluation
