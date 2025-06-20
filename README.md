# text-simplification: A Dataset for Multilingual Text Simplification

Code and pretrained models to reproduce experiments in "MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases".

## Prerequisites

Linux with python 3.6 or above (not compatible with python 3.9 yet). If your operating system is Windows, you can use WSL with ubuntu 20.04LTS. 

## Installing

```
git clone git@github.com:facebookresearch/muss.git
cd muss/
pip install -e .  # Install package
python -m spacy download pt_core_news_md en_core_web_md fr_core_news_md es_core_news_md # Install required spacy models
ulimit -n 100000 # If you train a new model
```

## How to use
Some scripts might still contain a few bugs, if you notice anything wrong, feel free to open an issue or submit a Pull Request.

### Simplify sentences from a file using pretrained models

First, download the template of the desired language in the folder `resources/models`. Pretrained models should be downloaded automatically, but you can also find them here:

[muss_en_wikilarge_mined](https://dl.fbaipublicfiles.com/muss/muss_en_wikilarge_mined.tar.gz)  
[muss_en_mined](https://dl.fbaipublicfiles.com/muss/muss_en_mined.tar.gz)  
[muss_fr_mined](https://dl.fbaipublicfiles.com/muss/muss_fr_mined.tar.gz)  
[muss_es_mined](https://dl.fbaipublicfiles.com/muss/muss_es_mined.tar.gz)  
[muss_pt_mined](https://drive.google.com/uc?export=download&id=1QcSHDjTtsBYSX95NvL_gefrQ2IkzpH-4) 

Then run the command:
```python
python scripts/simplify.py FILE_PATH_TO_SIMPLIFY --model-name MODEL_NAME

# English
python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined
# French
python scripts/simplify.py scripts/examples.fr --model-name muss_fr_mined
# Spanish
python scripts/simplify.py scripts/examples.es --model-name muss_es_mined
# Portuguese
python scripts/simplify.py scripts/examples.pt --model-name muss_pt_mined
``` 

### Mine the data
If you are going to add a new language to this project, in folder `resources/models/language_models/wikipedia` donwload the files of the target language from https://huggingface.co/edugp/kenlm/tree/main/wikipedia. These language models are used to filter high quality sentences in the paraphrase mining phase.

To run paraphrase mining run the command below:

```python
python scripts/mine_sequences.py
```

### Train the model
```python
python scripts/train_model.py NAME_OF_DATASET --language LANGUAGE
```

### Evaluate simplifications
Please head over to [EASSE](https://github.com/feralvam/easse/) for Sentence Simplification evaluation.


## License

The MUSS license is CC-BY-NC. See the [LICENSE](LICENSE) file for more details.

## Authors

* **Louis Martin** ([louismartincs@gmail.com](mailto:louismartincs@gmail.com))
* **Raphael Assis** ([contato.raphael.assis@gmail.com](mailto:contato.raphael.assis@gmail.com))


## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}
```