## Abstract:
Continual relation extraction (CRE) is an important task of continual learning, which aims to learn incessantly emerging new relations between entities from texts. To avoid catastrophically forgetting old relations, some existing research efforts have focused on exploring memory replayed methods by storing typical historical learned instances or embedding all observed relations as prototypes in the episodic memory and replaying them in the subsequent training process. However, they generally fail to exploit the relation knowledge contained in the pre-trained language model (PLM), which could provide enlightening information to the representations of new relations from the known ones. To this end, we investigate the CRE from a novel perspective by generating knowledge-infused relation prototypes to leverage the relational knowledge from PLM with prompt tuning. Specifically, based on the typical samples collected from the historical learned instances with K-means algorithm, we devise novel relational knowledge-infused prompts to elicit relational knowledge from PLM for generating knowledge-infused relation prototypes. Then the prototypes are used to refine the typical examples embedding and calculate the stability-plasticity balance score for adjusting the memory replayed progress. The experimental results show that our method outperforms the state-of-the-art baseline models in CRE. The further extensive analysis presents that the proposed method is robust to memory size, task order, length of the task sequence, and the number of training instances.

# Run
## run KIP-Frame on FewRel: 
    bash run_KIP.sh FewRel 
## run KIP-Frame on TACRED
    bash run_KIP.sh TACRED 

## Citation 
@ARTICLE{9860068,
  author={Zhang, Han and Liang, Bin and Yang, Min and Wang, Hui and Xu, Ruifeng},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Prompt-Based Prototypical Framework for Continual Relation Extraction}, 
  year={2022},
  volume={30},
  number={},
  pages={2801-2813},
  doi={10.1109/TASLP.2022.3199655}}

## URL : https://ieeexplore.ieee.org/document/9860068