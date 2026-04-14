# PLKDCN：Partial large kernel depth-wise convolutional Network for lightweight image super-resolution



### Installation
```
# Install dependent packages
cd PLKDCN
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```
You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation

Put ours_arch.py to the path "basicsr/archs".

### Training
- Run the following commands for training:
```python
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n5_x2.yml
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n5_x3.yml
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n5_x4.yml

# L
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n10_x2.yml
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n10_x3.yml
python basicsr/train.py -opt options/train/Ours/train_DF2K_k9d64n10_x4.yml
```

### Testing
- Download the pretrained models.
- Download the testing dataset.
- Run the following commands:
```python
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n5_x2.yml
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n5_x3.yml
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n5_x4.yml

# L
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n10_x2.yml
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n10_x3.yml
python basicsr/test.py -opt options/test/Ours/test_DIV2K_k9d64n10_x4.yml
```
- The test results will be in './results'.


### Results
## Citation
If you find this repository helpful, you may cite:

```tex

```

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox.
