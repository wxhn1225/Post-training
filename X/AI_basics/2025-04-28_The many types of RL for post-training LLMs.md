# The many types of RL for post-training LLMs

> 原贴: [The many types of RL for post-training LLMs](https://x.com/NandoDF/status/1916835195992277281)

RL有着多种形式和变体。正是这种多样性使得人们能在这个领域研究多年，并仍不时有新的惊喜。

![RL settings_img.jpg](../src/RL%20settings_img.jpg)

![RL settings_txt.jpg](../src/RL%20settings_txt.jpg)

首先，很多"**使用RL微调LLM**"的例子实际上是单步(**one-step**)RL问题(如上图左上)。在这类问题中，模型根据给定提示生成单一动作并获得评估(**evaluation**)。单一动作可以是文本回答、思维链(Chain of Thought, **CoT**)推理序列、草图、语音或任何其他行为信号，即任何标记序列。*Evaluation*通常是单一结果奖励，例如回答是否正确。

在与chatbot的多步(**multi-step**)对话中，用户是*environment*，chatbot是*agent*。当用户不提供输入时，决定下一步说什么是*one-step* RL问题。从图表(左上)可以看出，三个动作可以合并为一个单一动作而不破坏决策图结构。然而，规划整个对话以实现最终目标，其中用户和chat *agent*都在适应，则是*multi-step* RL问题(图表左下)。这种设置也可模拟chatbot使用工具的情况，如网络浏览器、编译器或科学实验室来收集信息。当LLM *policy*选择工具(如实验室测试)时，它从测试中收集信息并输入到语言模型中，以决定下一步*action*。

RL可能涉及多个步骤，每一步世界都在变化。因此，获得*reward*时，我们不知道哪些决策导致了这个*reward*。这被称为信用分配问题(credit assignment problem)。

由于多步骤特性，RL具有组合性和高维度特点。有时视界是无限的，有时视界未知，即决策步骤数量不确定。在这些情况下，我们必须解决跨维度推理问题。

简言之，RL非常困难，解决方案的方差可能很高。研究人员发明了一系列概念来控制方差，代价是引入偏差，包括价值函数(表示预期*reward*的函数，包含对未来*reward*折扣的估计)。这些概念在*multi-step*决策问题中有用，但*one-step* RL中不总是必需。虽然某些方法在电脑游戏中表现良好，但在LLM中往往失效。

在控制领域，考虑具有T步骤、二次奖励函数和线性模型的决策问题很常见。这些被称为线性二次高斯(linear quadratic Gaussian )控制器或调节器，构成了**模型预测控制(Model Predictive Control, MPC)**这一广泛应用控制类型的基础。

然而，盲目将为电脑游戏或控制开发的RL方法理论和软件应用到语言模型领域是危险的。这些方法需处理游戏引擎的异步昂贵数据生成和陈旧的重放缓冲区。相反，重要的是测量系统中所有组件的效率，并针对可用硬件和交互类型优化算法。

为实现工具使用和*multi-step*辅助，需采用*multi-step* RL训练LLM。然而，要达到[DeepSeek-R1](https://arxiv.org/abs/2501.12948)或[测试时强化学习(TTRL)](https://arxiv.org/pdf/2504.16084)等方法的效果，可能首先解决相对简单的*one-step* RL问题就足够了。一些软件，如后文将介绍的*policy gradient*，对*one-step*和*multi-step* RL可能非常相似，因此我们不妨使用通用灵活的软件框架。

所有RL *agent*都能自学习和自我提升。如果设计得当，它们可以构建质量不断提高的数据集，从而形成质量不断提高的*policy*。RL *agent*的这一特性对性能和安全性都至关重要。在*one-step* RL中，我们通过反复求解*one-step*问题来实现改进，改进应随每次迭代增长。

还有一些更复杂的RL情况未在此讨论。有时决策视界未知或无限，增加了推理复杂性。此外，时间步可以是连续的或中断驱动的。动作和观察可以是离散和连续的混合，如带插图的文本文档，奖励可结合多个期望组件。还有合作和对抗环境下的多智能体(**multi-agent**)扩展。

出于教学目的，我们明天将首先介绍最简单的情况：*one-step* RL。我们将介绍主要算法，这足以解释deepseek-R1、[Reinforced Self-Training (ReST) for Language Modelling](https://arxiv.org/abs/2308.08998)以及[Beyond Human Data: Scaling Self-Training](https://arxiv.org/abs/2312.06585)(ReST^EM)等方法。之后，我们将讨论支持工具使用的*multi-step*设置。我们有意避免引入过多复杂性，因为更简单的方法通常足以用于LLM微调。

