# Implementation of Caption Enriched Samples for Improving Hateful Memes Detection
A research repository for improved visiolingual tasks using caption enriched samples.

# A quick overview
The recently introduced hateful meme challenge demonstrates the difficulty of determining whether a meme is hateful or not. Specifically, both unimodal language models and multimodal vision-language models cannot reach the human level of performance. Motivated by the need to model the contrast between the image content and the overlayed text, we suggest applying an off-the-shelf image captioning tool in order to capture the first. We demonstrate that the incorporation of such automatic captions during fine-tuning improves the results for various unimodal and multimodal models. Moreover, in the unimodal case, continuing the pre-training of language models on augmented and original caption pairs, is highly beneficial to the classification accuracy.

# Datasets
We use the publicly available hateful memes dataset along with Vilio's features dataset for UNITER implementation.
Locations:
* Hateful Memes Dataset can be downloaded from here: https://www.drivendata.org/competitions/64/hateful-memes/page/206/. We use Phase 1 data.
* Features for UNITER can be found on Kaggle: https://www.kaggle.com/muennighoff/hmtsvfeats and https://www.kaggle.com/muennighoff/hmfeatureszipfin
We augment the dataset with generated captions. 

# Code
Some explanation about the code structure:
  * mmf-modications - changes done to mmf codebase to utilize captions for BERT, RoBERTa, VisualBert, ViLBERT. 
  * vilio\_with\_captions - changes done to Vilio codebase to support UNITER with captions.
  * MAX-Image-Caption-Generator - provided as is to for caption generations
  * my\_hateful\_memes - code that uses MAX-Image-Caption-Generator to create captions for the dataset and the generated captions in 2 formats - CSVs for the dataset and datasets for BERT and RoBERTa's pretraining (using huggingface transformers code, not supplied)
  * Kaggle-Ensemble-Guide - changes to the Kaggle-Ensemble-Guide codebase to generate ensembles for the results.

# Runtime and implementation details
Fine-tuning models using MMF codebase: we re-used best settings as for the original Hateful memes challenge:
22000 updates for number of updates. We use weighted Adam with cosine learning rate schedule and fixed 2000 warmup steps for optimization without gradient clipping. 
Batch size: 32 for all models.
Learning rate: 5e-5 for all models.

Finetuning after pretraining for the BERT and RoBERTA models - we use 5e-6 vs 5e-5  learning rate. 

All models are trained on a single GPU GeForce GTX TITAN X with 12GB of RAM (same GPU in all runs below). Different models have different training times but mostly it took 10 to 13 hours for each model.

To continue the pre-training of the language models (BERT and RoBERTA) we used the transformers library, each pretraining was with batch size: 32 and, spread across 4 GPUs (effective batch size: 128). Pretraining runs in about 2-3 hours (100 epochs).

For the UNITER baseline, we use batch size:8 on a single GPU (learning rate: 1e-5). For ensembling it with a pre-trained RoBERTA model, we use batch size:6 with gradient accumulation: 2 (same learning rate). Each run takes roughly 1 hour. We used 36 features for the images.

# Credits
* MMF - https://github.com/facebookresearch/mmf
* IBM Max caption generator - https://github.com/IBM/MAX-Image-Caption-Generator
* Transformers (huggingface) - https://github.com/huggingface/transformers/
* Vilio - https://github.com/Muennighoff/vilio
* Kaggle Ensemble Guide - https://github.com/MLWave/Kaggle-Ensemble-Guide


# Citation 
TBD - EMNLP 2021

