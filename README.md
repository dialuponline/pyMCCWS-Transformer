
# Chinese Word Segmentation with Pytorch and Transformers

A Pytorch(`fastNLP`) implementation for an efficient and concise model for Multi-Criteria Chinese Word Segmentation using a Transformer.

## Instructions

- Start by placing your raw data in the `data/` directory.
- Use this command to prepare the corpora:
```
python prepoccess.py
```
- Next, prepare your inputs using:
```
python makedict.py
python make_dataset.py --training-data data/joint-sighan-simp/bmes/train-all.txt --test-data data/joint-sighan-simp/bmes/test.txt -o <output_path>
```
- This will produce a `.pkl` containing the necessary dictionaries and vocabularies in the followings format:
```
{
    'train_set': fastNLP.DataSet
    'test_set': fastNLP.DataSet
    'uni_vocab': fastNLP.Vocabulary, vocabulary of unigram
    'bi_vocab': fastNLP.Vocabulary, vocabulary of bigram
    'tag_vocab': fastNLP.Vocabulary, vocabulary of BIES
    'task_vocab': fastNLP.Vocabulary, vocabulary of criteria
}
```
- Use the following command to train your model (this will freeze the embeddings initially):
```
python main.py --dataset <output_path> --task-name <save_path_name> --word-embeddings <file_of_unigram_embeddings> --bigram-embeddings <file_of_bigram_embeddings> --freeze --crf --devi 0
```
- The embeddings can be found [here](https://drive.google.com/drive/folders/1Zarmj6WRf0jADXbklT4_c1PXM_1L8FT5). The files include both simplified and traditional Chinese.
- After the initial training, continue training the model without freezing the embeddings:
```
python main.py --dataset <output_path> --task-name <save_path_name> --num-epochs 20 --old-model result/<save_path_name>/model.bin --word-embeddings <file_of_unigram_embeddings> --bigram-embeddings <file_of_bigram_embeddings> --step <previous_training_step> --crf --devi 0
```
- For additional command details, use:
```
python main.py --help
```
Please note that this project is maintained by dialuponline. Feel free to contribute!