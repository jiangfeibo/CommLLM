# Large Language Model Enhanced Multi-Agent Systems for 6G Communications
## Authors
### Feibo Jiang, Li Dong, Yubo Peng, Kezhi Wang, Kun Yang, Cunhua Pan, Xiaohu You
## Paper
### https://arxiv.org/abs/2312.07850
## Code
### https://github.com/jiangfeibo/LLM-MAS.git
## Abstract
The rapid development of the Large Language Model (LLM) presents huge opportunities for 6G communications, e.g., network optimization and management by allowing users to input task requirements to LLMs by nature language. However, directly applying native LLMs in 6G encounters various challenges, such as a lack of communication data and knowledge, and limited logical reasoning, evaluation, and refinement abilities. Integrating LLMs with the capabilities of retrieval, planning, memory, evaluation and reflection in agents can greatly enhance the potential of LLMs for 6G communications. To this end, we propose a multi-agent system with customized communication knowledge and tools for solving communication-related tasks using natural language, comprising three components: (1) Multi-agent Data Retrieval (MDR), which employs the condensate and inference agents to refine and summarize communication knowledge from the knowledge base, expanding the knowledge boundaries of LLMs in 6G communications; (2) Multi-agent Collaborative Planning (MCP), which utilizes multiple planning agents to generate feasible solutions for the communication-related task from different perspectives based on the retrieved knowledge; (3) Multi-agent Evaluation and Reflexion (MER), which utilizes the evaluation agent to assess the solutions, and applies the reflexion agent and refinement agent to provide improvement suggestions for current solutions. Finally, we validate the effectiveness of the proposed multi-agent system by designing a semantic communication system, as a case study of 6G communications.
![img](SC.png)

## The function of each file
- [LLangchain.ipynb](LLangchain.ipynb): The implementation of LLM-enhanced multi-agent systems for the generation of semantic communication systems.

- [Test_generated_SC.py](Test_generated_SC.py): Test the generated SC model based on cosine similarity.

- [1.pdf](1.pdf): Reference paper.

- [code.txt](code.txt): Reference code.

- [movie_lines.txt](movie_lines.txt): Training and test data for the generated SC model.

## Citation   
```
@article{jiang2023large,
  title={Large language model enhanced multi-agent systems for 6G communications},
  author={Jiang, Feibo and Dong, Li and Peng, Yubo and Wang, Kezhi and Yang, Kun and Pan, Cunhua and Niyato, Dusit and Dobre, Octavia A},
  journal={arXiv preprint arXiv:2312.07850},
  year={2023}
}
```

