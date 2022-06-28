# ICE Tokenizer

- Token id `[0, 20000)` are image tokens.
- Token id `[20000, 20100)` are common tokens, mainly punctuations. E.g., `icetk[20000] == '<unk>'`, `icetk[20003] == '<pad>'`, `icetk[20006] == ','`.
-  Token id `[20100, 83823)` are English tokens.
-  Token id `[83823, 145653)` are Chinese tokens.
-  Token id `[145653, 150000)` are rare tokens. E.g., `icetk[145803] == 'α'`.

You can install the package via 
```
pip install icetk
```

## Tokenization

```python
from icetk import icetk
tokens = icetk.tokenize('Hello World! I am icetk.')
# tokens == ['▁Hello', '▁World', '!', '▁I', '▁am', '▁ice', 'tk', '.']
ids = icetk.encode('Hello World! I am icetk.')
# ids == [39316, 20932, 20035, 20115, 20344, 22881, 35955, 20007]
en = icetk.decode(ids)
# en == 'Hello World! I am icetk.' # always perfectly recover (if without <unk>)

ids = icetk.encode('你好世界！这里是 icetk。')
# ids == [20005, 94874, 84097, 20035, 94947, 22881, 35955, 83823]

ids = icetk.encode(image_path='test.jpeg', image_size=256, compress_rate=8)
# ids == tensor([[12738, 12430, 10398,  ...,  7236, 12844, 12386]], device='cuda:0')
# ids.shape == torch.Size([1, 1024])
img = icetk.decode(image_ids=ids, compress_rate=8)
# img.shape == torch.Size([1, 3, 256, 256])
from torchvision.utils import save_image
save_image(img, 'recover.jpg')

# add special tokens
icetk.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

# transform \n
icetk.decode(icetk.encode('abc\nhi', ignore_linebreak=False))
# 'abc\nhi'
icetk.decode(icetk.encode('abc\nhi'))
# 'abc hi'
```
