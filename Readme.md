# Transformer from Scratch

### Running the code

```
python3 trans.py
```
Also, contact them and tell ki everytjing is disabled. Whenever they wish to register, contact us for the same (Only for golden tickets)

### Hyperparameters

```"learning_rate": 0.0001,
        "Batch_size" :8,
        "Epochs" :5,
        "Embedding_dim" :512,
        "num_layers" :2,

```

- ### optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
- ### criterion = nn.CrossEntropyLoss(ignore_index=0)

### Multiheaded Attention
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/23f0a012-65a1-41e5-8118-cca9142c1977/7d40cbc2-8f80-4439-aa20-89526a11fe2c/Untitled.png)

### Positional Encoding
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/23f0a012-65a1-41e5-8118-cca9142c1977/762f2a92-3976-4cb3-abfd-72c72de49d9f/Untitled.png)


Note :- 
1. Some online resources and LLM code was used for reference.