# The many types of RL for post-training LLMs

> 原贴: [April 29: The many types of RL for post-training LLMs](https://x.com/NandoDF/status/1916835195992277281)

RL有着多种形式和变体。正是这种多样性使得人们能在这个领域研究多年，并仍不时有新的惊喜。

![RL settings_img.jpg](../src/RL%20settings_img.jpg)

![RL settings_txt.jpg](../src/RL%20settings_txt.jpg)

首先，很多**"使用RL微调LLM"**的例子输入单步**（One-Step）**RL问题(如上图左上)。在此场景下，模型接收提示**（Prompt）**生成单一action并获得评估**（evaluation）**。单一action可以是文本回答、思维链**（Chain of Thought，CoT）**推理序列、草图、语音或任何其他行为信号——即任何词元（token）序列。Evaluation通常是单步结果奖励，例如回答是否正确。

在与chatbot的多步**（multi-step）**对话中，用户是environment，chatbot是agent。在用户未提供输入时决定下一步回复，属于one-step RL问题。从图中(左上)可以看出，三个action可轻松合并为一个而不破坏决策图结构。然而，规划整个对话以实现最终目标（期间用户和chat agent相互适应），则是multi-step RL问题(图中左下)。。此设定也可建模chatbot使用工具（如浏览器、编译器或科学实验室）收集信息的过程：当LLM policy选择工具（如实验室测试）时，它收集测试信息并输入语言模型，以决定下一步action。

RL可能涉及多步决策，每一步世界状态随之改变。因此，当获得reward时，难以确定是众多决策中的哪一个导致了该reward。这被称为信用分配问题**（Credit Assignment Problem）**。

由于多步特性，RL具有组合性**（Combinatorial）** 和极高维度**（High-Dimensional）**。有时决策视界**（Horizon）** 是无限的，有时视界未知（即决策步数不定）。这些场景下，需解决跨维度推理问题**（Trans-Dimensional Inference Problem）**。

简言之，RL及其困难，解决方案的方差可能很高。研究者发明了一系列概念（如价值函数**（Value Functions）**——代表期望reward的函数，含对未来reward的折扣**（Discount）**预测）以控制方差（代价是引入偏差**（Bias）**）。这些概念在multi-step决策问题中有用，但one-step RL中并非必需。虽然部分方法在电子游戏中表现良好，但在LLM中往往失效。

在控制领域，考虑具有T步决策、二次奖励函数和线性模型的决策问题很常见。这些被称为线性二次高斯**（Linear Quadratic Gaussian，LQG）**控制器或调节器，并构成了最普遍的控制方法之一模型预测控制**（Model Predictive Control，MPC）**的基础。

然而，将电子游戏或控制领域开发的RL理论与软件盲目套用于语言模型是危险的。这些方法需处理游戏引擎中异步生成的高成本数据和过期的回放缓冲区。相反，重要的是测量系统中所有组件的效率，并针对可用硬件和交互类型优化算法。

对于工具使用和multi-step辅助任务，LLM需要multi-step RL。然而，要达到[DeepSeek-R1](https://arxiv.org/abs/2501.12948)或[测试时强化学习（TTRL）](https://arxiv.org/pdf/2504.16084)等方法的效果，优先解决相对简单的one-step RL问题可能已足够。部分软件（如后文将介绍的policy gradient）在one-step和multi-step RL可能非常相似，因此我们不妨尝试通用灵活的软件框架。

所有RL agent都具有自学习**（Self-Learn）**和自我提升**自我改进（Self-Improve）**的能力。如果设计得当，它们可以构建质量不断提高的数据集，从而产生质量不断提高的policy。RL agent的这一特性对性能和安全性都至关重要。在one-step RL中，我们重复解决one-step问题，改进应随每次迭代而增长。

还有一些更复杂的RL情况未在此讨论：决策视界**（Decision Horizon）**可能未知或无限（增加推理复杂度）；时间步长可为连续或中断驱动；action和观察可为离散与连续混合（如带插图的文本文档）；奖励可组合多个期望项；还有协作或对抗环境下的multi-agent扩展。

出于教学目的，我们明天将首先介绍最简单的情况：one-step RL。涵盖核心算法，这足以阐述deepseek-R1、[Reinforced Self-Training（ReST）for Language Modelling](https://arxiv.org/abs/2308.08998)以及[Beyond Human Data: Scaling Self-Training](https://arxiv.org/abs/2312.06585)（ReST^EM）等方法。之后，我们将讨论支持工具使用的multi-step设定。我们有意避免引入过多复杂性，因为更简单的方法通常足以满足LLM微调需求。

