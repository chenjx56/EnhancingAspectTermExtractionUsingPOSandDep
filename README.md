# EnhancingAspectTermExtractionUsingPOSandDep
Source code for ICTAI2022: Enhancing Neural Aspect Term Extraction Using Part-Of-Speech and Syntactic Dependency Features

# Enhancing Neural Aspect Term Extraction Using Part-Of-Speech and Syntactic Dependency Features



## Steps to Run Code

- ### Step 1:

Download pre-trained model weight [[bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main)] [[pt-bert-base](https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/transformers/amazon_yelp.md)]



- ### Step 2:

**Run the single Model:**

Place model files as:

```
Single/PLM/bert-base-uncased/
Single/PLM/pt-bert-base/
```

Run:

```
cd Single
fitlog init .
python main.py
```



- ### Step3:

**Run the PST&Ours:** 

Place model files as: **（Please don't replace existing files in the folder）**

```
PST/train/model/bert-base-uncased/
PST/train/model/pt-bert-base/
```

Run:

```
cd PST/train/
bash pipeline.sh
```

