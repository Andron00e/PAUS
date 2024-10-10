# Supplementary code for "Bregman Proximal Method for Efficient Communications under Similarity"
[![arXiv](https://img.shields.io/badge/arXiv-2401.06766-b31b1b.svg)](https://arxiv.org/abs/2311.06953)

## Repository structure
* in <ins>methods</ins> you may see an implementation of our ```PAUS``` algorithm, and baselines: ```Decentralized Mirror Prox``` (based on Bregman proximal maps, but without data similarity),  ```Extra Gradient``` (under similarity, but in the Euclidean setup).
* <ins>experiments</ins> contains code examples to compare the three above methods.
* in the <ins>plots</ins> folder you may find all of the reported comparison on the ''Policeman vs Burglar'' problem.
* <ins>tests</ins> is just a demo for you, feel free to run it.

```bib
@misc{beznosikov2024bregmanproximalmethodefficient,
      title={Bregman Proximal Method for Efficient Communications under Similarity}, 
      author={Aleksandr Beznosikov and Darina Dvinskikh and Dmitry Bylinkin and Andrei Semenov and Alexander Gasnikov},
      year={2024},
      eprint={2311.06953},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2311.06953}, 
}
```
