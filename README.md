# IDA Project

CSE5002 Intelligent Data Analysis Mini Project 2023, SUSTech

Input: MIT social network with attributes

Output: class year

The original dataset comes from [1].



## Environment

Python 3.7.16

packages required: `numpy`, `networkx`, `sklearn`, `node2vec`

`node2vec` is the official python implementation based on [2].



## How to run

The embedding is already generated in `./data`. If you want to generate it again, you can run

```bash
python emb_generate.py
```

To see the test results, you can run

```
python train.py
```



## Reference

[1] Traud, Amanda L., et al. "Comparing community structure to characteristics in online collegiate social networks." SIAM review 53.3 (2011): 526-543.

[2] Grover, A., & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. *KDD : proceedings. International Conference on Knowledge Discovery & Data Mining*, *2016*, 855–864.
