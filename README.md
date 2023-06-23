# Bangla Key2Text: Text Generation from Keywords in Bengali
**Author:** G M Shahariar Shibli, Tonmoy Talukder
<hr style="width:50px" />

### Bangla-Key2Text

#### Find in Hugging Face (Currently, the privacy is in private mode, after acceptance of the paper, it will be released in public mode.)
[:hugs: Bangla-Key2Text](https://huggingface.co/tonmoytalukder/Bangla-Key2Text)

#### Run using the below code
```python
!pip install sentencepiece
!pip install transformers
!pip install git+https://github.com/csebuetnlp/normalizer
!pip install torch

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from normalizer import normalize

model_dir = 'tonmoytalukder/Bangla-Key2Text'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(key): # Function to generate text from given keywords
    input_ids = tokenizer.encode(key, return_tensors='pt',add_special_tokens=True).to(device)

    with torch.no_grad():
      outputs = model.generate(
          input_ids=input_ids,
          max_length =512,
          num_beams =2,
          early_stopping =True,
          num_return_sequences = 1,
          top_k= 50,
          top_p= 0.95,
          repetition_penalty= 2.5,
          length_penalty= 1.0)

    preds = [tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=True) for g in outputs]

    generated_text = preds[0]
    return generated_text


keywords = "কেমন ডাটাসেট সময় ভাই বানাতে"
predict(normalize(keywords)) # This normalize function will preprocess (clean) the sentence. 
# Output: "ভাই, ডাটাসেট বানাতে কেমন সময় লাগে?"
```

### Bangla Keyword Extractor

#### Find in Hugging Face (Currently, the privacy is in private mode, after acceptance of the paper, it will be released in public mode.)
[:hugs: bn-keyword-extractor](https://huggingface.co/tonmoytalukder/bn-keyword-extractor)

We have uploaded this code as a PyPI project. [bn-keyword-extractor](https://pypi.org/project/bn-keyword-extractor/) 
#### Run using the below code
```python
!pip install bn-keyword-extractor

from keyword_extractor import KeywordExtractor
extractor = KeywordExtractor()
text = "আমি বাংলায় গান শোনা ভালবাসি।"
keywords = extractor.extract_keywords(text)
print(keywords) 
```
Output: ['শোনা', 'ভালবাসি', 'বাংলায়', 'গান']

### Dataset: Bangla Key2Text 2 Million 

We have uploaded our dataset as a Huggingface Dataset. (Currently, the privacy is in private mode, after acceptance of the paper, it will be released in public mode.)
[:hugs: Bangla-Key2Text-2Million](https://huggingface.co/datasets/tonmoytalukder/Bangla-Key2Text-2Million)
#### Run using the below code
```python
!pip install datasets

from datasets import load_dataset

dataset = load_dataset("tonmoytalukder/Bangla-Key2Text-2Million", split="train") #split="test"
dataset
```
