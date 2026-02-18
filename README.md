

## Reference Architecture
![VGG16 Architecture](https://viso.ai/wp-content/uploads/2024/04/vgg-16.bak-1280x708.png)

## TODO List

[-] check the effect of batch size 

[x] ~~use lr scheduler, for example, StepLR~~ 
    lr scheduler reduces fluctuations in loss values in later training stage 

[x] ~~use WandB to record train trajectory~~ 

[-] use a validation set during training, and show it along with a loss trajectory 

[-] Do a literature search on how VGG16 was improved 

[-] introduce the residual connection 

[-] integrate L1 or L2 regularization (or batch normalization) into the VGG16 


## Vision Transformers (next step)
![Vision Transformers (ViT)](https://discuss.pytorch.kr/t/vision-transformer-a-visual-guide-to-vision-transformers/4158)



- transfer learning is chosen 
- choosing a proper learning rate is critical in training
- 1e-4 is used, larger ones do not help 
