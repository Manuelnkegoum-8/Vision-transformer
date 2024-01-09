
# Model description
Recently, the Vision Transformer (ViT), which applied the transformer structure to the image classification task, has outperformed convolutional neural networks.
However, the high performance of the ViT results from pre-training using a large-size dataset such as JFT-300M, and its dependence on a large dataset is interpreted as due to low locality inductive bias. 
This paper proposes Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA), which effectively solve the lack of locality inductive bias and enable it to learn from scratch even on small-size datasets. 
Moreover, SPT and LSA are generic and effective add-on modules that are easily applicable to various ViTs.

## Method
### Shifted Patch tokenization
<!-- <div align="center"> -->
  <img src="SPT.png" width="75%" title="" alt="teaser">
<!-- </div> -->

### Locality Self Attention
<!-- <div align="center"> -->
  <img src="LSA.png" width="75%" title="" alt="teaser">
<!-- </div> -->

## Citation
```
@article{lee2021vision,
  title={Vision Transformer for Small-Size Datasets},
  author={Lee, Seung Hoon and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2112.13492},
  year={2021}
}
```
