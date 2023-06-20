# Attention based Bangla Keyword Extractor
**Author: Tonmoy Talukder**
<hr style="width:50px" />

### Find in Hugging Face 
[:hugs: bn-keyword-extractor](https://huggingface.co/tonmoytalukder/bn-keyword-extractor)

### Run using the below code
```python
!pip install bn-keyword-extractor

from keyword_extractor import KeywordExtractor
extractor = KeywordExtractor()
text = "আমি বাংলায় গান শোনা ভালবাসি।"
keywords = extractor.extract_keywords(text)
print(keywords) 
```
Output: ['শোনা', 'ভালবাসি', 'বাংলায়', 'গান']

