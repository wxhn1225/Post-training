# Distributed RL Systems

> 原帖: [April 25:  Distributed RL Systems](https://x.com/NandoDF/status/1915548697548464359)

Agent是一种能够感知environment、自主采取行动**（Action）**以实现目标，并可通过RL或指导提升性能的实体。Agent可拥有内在目标（如推断出的子目标，以及对更多观察、更多学习或更多控制的追求——这正是我们需要考虑安全问题的地方！），也可拥有外在目标（体现为指定的奖励**（Reward）**函数或反馈reward信号）。下图展示了RL的主要组成部分。

![RL ingredients.jpg](../src/RL%20ingredients.jpg)

Agent可以是与用户（其environment）交互的多模态神经网络，为用户提供个性化教育（目标）。Agent观察越多，就越容易为用户定制个性化的学习方案（课程）。

工业级LLM的RL可能涉及数百万次并发交互、数十亿参数模型以及整个数据中心。成本高昂！构建能在如此庞大规模下高效运行的RL系统**（RL System）**绝非易事。在此，我仅对这类可扩展的分布式系统**（Scalable Distributed System)**做一浅显概述。向Anthropic、DeepMind、DeepSeek、Meta、Microsoft AI、OpenAI、X等公司的杰出工程师们致敬！在我看来，他们如同英超顶级球员一般独特且才华横溢。

如我部分同事（参见[《IMPALA: Scalable Distributed Deep-RL》](https://arxiv.org/abs/1802.01561)和[acme: A library of reinforcement learning](https://github.com/google-deepmind/acme)）所述，现代distributed RL system可分为两个部分：Actor和Learner。

每个actor通过一个称为policy的网络**（Network）**与environment交互生成action。Actor还从environment中收集reward和观测结果。收集的数据被添加到公共经验回放池**（Replay Memory）**中。

Learner从replay memory中采样数据并用于更新policy network。更新network后，权重**（Weight）**检查点需要发送给每个actor。设计此类系统时，衡量每项操作的持续时间、测量每个通信链路的带宽等至关重要。这需要精确的工程设计和全面的测量与消融实验**（Ablations）**。

![modern distributed RL systems.jpg](../src/modern%20distributed%20RL%20systems.jpg)

在语言领域，actor是chatbot agent，而environment则是人。每次对话的数据被送入replay memory中供学习使用。通常，learner比actor需要更多的存储和计算资源，因为learner需要处理梯度Gradient等大规模统计量。

了解actor推理成本、通信成本和学习成本非常重要。在某些场景下，这些成本允许agent进行在线策略**（On-Policy）**学习。由于不同actor可能以不同速度和时间收集数据，这个过程通常是异步的。

如果数据收集速度不够快，learner可能需要重放回放池中的旧样本以更新policy。这就是离线策略**（Off-Policy）**设置。此时必须校正使用陈旧数据学习模型带来的偏差——还记得[4月24日推文](2025-04-24_RL%20vs%20SFT.md)中关于驾驶的例子吗？过度off-policy可能很危险！幸运的是，研究者已有解决方案，例如重要性权重**（Importance Weights）**和其他加权机制，如[近端策略优化(Proximal Policy Optimization, **PPO**)](https://en.wikipedia.org/wiki/Proximal_policy_optimization)和[DeepSeek-R1](https://arxiv.org/abs/2501.12948)论文中采用的权重。

最后，有时仅依靠大型回放数据库即可学习policy。这被称为离线RL**（off-line RL)**或批量RL**（batch RL）**。Off-line RL优于supervised learning，因为它包含前文讨论的selection mechanism，但逊于on-line RL，因其缺乏在environment中直接生成action的能力。然而，在交互成本过高或存在危险的场景中，off-line RL极具实用价值。

顺便说一下，欢迎提出任何疑问。即使涉及枯燥的技术话题，也欢迎持完全相反的意见——虽然这话在X平台上实属多余😂