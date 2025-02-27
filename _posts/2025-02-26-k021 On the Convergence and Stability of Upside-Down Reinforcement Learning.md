---
layout: mypost
title: k021 On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers
categories: [强化学习]
---

# 论文链接

- [Arxiv Link](https://arxiv.org/abs/2502.05672v1)
- [智源社区链接](https://hub.baai.ac.cn/paper/c566545e-57f1-495f-af73-1bc9730da17d)

发表日期：[v1] Sat, 8 Feb 2025 19:26:22 UTC (1,956 KB)（2025年2月8日）

# 论文导读

这篇论文《On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers》由Miroslav Štrupl等人撰写，提供了对“通过监督学习或序列建模解决强化学习问题”这一范式下算法的收敛性和稳定性的深入理论分析。这些算法包括Episodic Upside-Down Reinforcement Learning (eUDRL)、Goal-Conditioned Supervised Learning (GCSL) 和 Online Decision Transformers (ODT)。论文的核心目标是填补这些算法在理论基础方面的空白，尤其是它们在不同环境条件下的收敛性和稳定性。

## **研究背景**
强化学习（RL）算法通常通过与环境的交互来学习最优策略，而监督学习（SL）则依赖于标记数据进行学习。近年来，一些算法尝试将RL问题转化为SL问题，例如eUDRL、GCSL和ODT。这些算法在多个基准测试中表现出色，但其理论理解主要局限于启发式方法和特定环境。论文指出，尽管这些算法在确定性环境中表现出良好的性能，但在随机环境中可能会出现不稳定性和收敛问题。

## **研究方法**
论文提出了一个统一的理论框架，用于分析这些算法的收敛性和稳定性。研究的核心是分析环境的转移核（transition kernel）对算法性能的影响，特别是当转移核接近确定性时，算法是否能够收敛到最优解。为此，作者引入了“命令扩展”（Command Extension）的概念，这是一种特殊的马尔可夫决策过程（MDP），其中命令（如目标和时间范围）被纳入状态空间。

论文还引入了“相对连续性”（relative continuity）的概念，用于描述在确定性转移核附近，策略和值函数的连续性。这一概念允许作者在有限迭代次数下证明策略的相对连续性，并在无限迭代次数下研究策略的渐近行为。

## **实验与关键结论**
论文通过理论分析和数值实验验证了以下关键结论：

1. **确定性环境下的收敛性**：在确定性环境中，eUDRL算法能够收敛到最优策略。论文通过数学证明展示了这一点，并指出在确定性环境中，策略的更新是单调的，且不会出现非最优行为。

2. **随机环境下的稳定性**：在随机环境中，eUDRL算法可能会出现不稳定性。论文通过构造具体的例子展示了在某些随机环境中，即使转移核与确定性核非常接近，算法的性能也可能出现显著的下降。

3. **相对连续性**：论文证明了在有限迭代次数下，eUDRL生成的策略和值函数在确定性转移核附近是相对连续的。这意味着，即使在非确定性环境中，只要转移核足够接近确定性核，算法的性能也不会与最优解有太大偏差。

4. **渐近行为**：论文还研究了eUDRL算法在无限迭代次数下的渐近行为。在某些特殊条件下（如初始分布的支持集包含所有关键状态，或者最优策略是唯一的），论文证明了算法的累积点集（accumulation points）在确定性转移核附近是连续的。

5. **正则化的影响**：论文进一步探讨了正则化对eUDRL算法稳定性的影响。通过引入ϵ-greedy正则化，作者证明了在更一般的情况下，算法的累积点集在确定性转移核附近是相对连续的。

## **研究意义**
论文的理论分析为理解eUDRL、GCSL和ODT等算法在不同环境下的行为提供了重要的理论支持。这些结论不仅有助于解释这些算法在实际应用中的表现，还为设计更稳定、更高效的RL算法提供了理论指导。此外，论文中提出的“相对连续性”概念和分析方法也为未来研究提供了新的视角。

## **局限性与未来工作**
尽管论文在理论分析方面取得了重要进展，但作者也指出，对于更一般的随机环境，eUDRL算法的全局收敛性和稳定性仍然是一个开放问题。未来的研究可以进一步探索如何在更广泛的环境中保证算法的稳定性和收敛性，以及如何将这些理论结果应用于实际的复杂任务中。

# 关键问题解释

## Episodic（阶段性）

在强化学习（Reinforcement Learning, RL）和相关领域中，**Episodic**（阶段性）是一个重要的概念，用来描述任务或环境的结构特性，具体含义如下：

### 1. **Episodic（阶段性）的定义**
- **Episodic** 指的是任务或环境中的每个**“剧集”（Episode）**是独立的、有限长度的交互序列。每个剧集从初始状态开始，经过一系列动作和状态转移后，最终达到一个**终止状态（Terminal State）**，标志着该剧集的结束。
- 在每个剧集结束时，环境会重置到初始状态，开始一个新的剧集。这种结构使得学习过程可以分阶段进行，每个阶段都有明确的开始和结束。

### 2. **Episodic 与 Continuous（连续性）的对比**
- **Episodic（阶段性）**：每个剧集独立，有明确的开始和结束。例如，一个游戏从开始到结束是一个剧集，游戏结束后重新开始新的游戏。
- **Continuous（连续性）**：任务是一个无限的、持续的交互过程，没有明确的剧集边界。例如，自动驾驶汽车的运行是一个连续的任务，没有明确的“剧集”概念。

### 3. **Episodic 的应用场景**
- **游戏**：大多数游戏是阶段性任务。例如，在Atari游戏或棋类游戏中，每局游戏从开始到结束是一个剧集。
- **机器人任务**：例如，机器人完成一个特定任务（如抓取物体）是一个剧集，任务完成后重新开始新的任务。
- **强化学习算法**：许多强化学习算法（如Q-learning、SARSA等）假设任务是阶段性的，以便在每个剧集结束时更新策略。

### 4. **Episodic 的优势**
- **学习效率**：阶段性任务允许算法在每个剧集结束时进行评估和更新，从而提高学习效率。
- **简化问题**：阶段性任务可以将复杂的连续任务分解为多个独立的小任务，便于算法处理。
- **适应性**：阶段性任务允许算法在每个剧集结束时重置，从而更好地适应环境的变化。

### 5. **Episodic 在论文中的具体应用**
在你提到的论文中，**Episodic** 用于描述任务的结构，特别是在讨论 **Episodic Upside-Down Reinforcement Learning (eUDRL)** 时。eUDRL 是一种强化学习算法，它假设任务是阶段性的，每个剧集从初始状态开始，经过一系列动作和状态转移后，最终达到目标状态或终止状态。这种阶段性结构使得算法可以在每个剧集结束时进行策略更新，从而逐步优化策略。


## Command Extension（命令扩展, CE）

在论文中，**CE** 是一个缩写，指的是 **Command Extension**（命令扩展）。它是一种特殊的马尔可夫决策过程（MDP），用于扩展传统的强化学习（RL）框架，使其能够处理更复杂的任务，例如基于目标的任务（goal-conditioned tasks）和序列建模问题。以下是 **CE** 的详细解释：

### **1. Command Extension（命令扩展）的定义**
**Command Extension** 是一种特殊的马尔可夫决策过程（MDP），它通过引入“命令”（如目标和范围）来扩展传统的MDP框架。具体来说，CE 包含以下关键组成部分：

- **扩展状态空间**：在CE中，状态空间不仅包含原始MDP的状态，还包含额外的命令信息。扩展状态表示为：
  $$
  \bar{s} = (s, h, g) \in \bar{S} = S \times \bar{N}_0 \times G,
  $$
  其中：
  - $ s $ 是原始MDP的状态。
  - $ h $ 是剩余范围（horizon），表示从当前状态到目标状态的步数。
  - $ g $ 是目标（goal），表示需要达成的状态或条件。

- **目标映射**：目标映射 $ \rho: S \to G $ 用于评估目标是否达成。如果 $ \rho(s) = g $，则表示目标 $ g $ 在状态 $ s $ 中达成。

- **转移核**：扩展的转移核 $ \bar{\lambda} $ 定义了在扩展状态空间中的转移概率。例如：
  $$
  \bar{\lambda}((s', h-1, g) | (s, h, g), a) = \lambda(s' | s, a),
  $$
  其中 $ \lambda $ 是原始MDP的转移核。

- **奖励函数**：奖励函数 $ \bar{r} $ 通常设计为二元奖励，用于指示目标是否达成。例如：
  $$
  \bar{r}((s', h', g'), (s, h, g), a) = 
  \begin{cases}
  1, & \text{如果 } h = 1, h' = 0, g' = g \text{ 且 } \rho(s') = g, \\
  0, & \text{其他情况}.
  \end{cases}
  $$

- **吸收状态**：当范围 $ h $ 为0时，扩展状态 $ (s, g, 0) $ 是吸收状态，表示任务结束。


### **2. CE 的作用**
CE 的主要作用是将强化学习问题转化为一个更易于处理的形式，特别是在处理目标导向的任务时。通过引入目标和范围作为状态的一部分，CE 允许算法直接学习如何根据目标信息进行决策，而不是仅仅依赖于状态和奖励信号。

### **3. CE 在论文中的应用**
在论文中，CE 被用于以下几种算法的分析和实现：

- **Episodic Upside-Down Reinforcement Learning (eUDRL)**：eUDRL 是一种基于CE的算法，它通过将目标和范围作为输入，学习如何预测最优动作。eUDRL 的策略更新公式（如方程 2.7）直接在扩展状态空间中进行，允许算法学习如何根据目标信息进行决策。

- **Goal-Conditioned Supervised Learning (GCSL)**：GCSL 是一种基于CE的监督学习算法，它通过目标信息指导模型的学习过程。GCSL 的目标是学习如何根据目标和状态信息预测最优动作。

- **Online Decision Transformers (ODT)**：ODT 是一种基于CE的在线决策算法，它利用Transformer架构对整个轨迹进行建模，预测最优动作。ODT 的目标是学习如何根据目标和历史信息进行决策。


### **4. CE 的优势**
- **目标导向**：CE 明确地将目标信息纳入状态空间，使得算法能够更好地处理目标导向的任务。
- **灵活性**：CE 的框架允许算法在不同的任务和环境中灵活调整策略。
- **理论分析**：CE 提供了一个统一的理论框架，用于分析和证明算法的收敛性和稳定性。


# 详细讲解

这篇论文《On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers》由Miroslav Štrupl等人撰写，旨在深入研究通过监督学习（SL）或序列建模解决强化学习（RL）问题的算法，特别是Episodic Upside-Down Reinforcement Learning (eUDRL)、Goal-Conditioned Supervised Learning (GCSL)和Online Decision Transformers (ODT)。论文的核心目标是提供这些算法在收敛性和稳定性方面的理论基础，填补现有研究中的空白。

## **1. 研究背景与动机**
强化学习（RL）通常通过与环境的交互学习最优策略，而监督学习（SL）则依赖于标记数据。近年来，一些算法尝试将RL问题转化为SL问题，例如：
- **Upside-Down Reinforcement Learning (UDRL)**：将奖励信号映射到动作，将RL问题转化为SL问题。
- **Goal-Conditioned Supervised Learning (GCSL)**：利用目标信息指导模型学习。
- **Online Decision Transformers (ODT)**：利用Transformer架构建模整个轨迹，将RL视为序列建模问题。

这些算法在多个基准测试中表现出色，但其理论理解主要局限于启发式方法和特定环境。论文指出，尽管这些算法在确定性环境中表现出良好的性能，但在随机环境中可能会出现不稳定性和收敛问题。

## **2. 研究方法**
论文提出了一个统一的理论框架，用于分析这些算法的收敛性和稳定性。研究的核心是分析环境的转移核（transition kernel）对算法性能的影响，特别是当转移核接近确定性时，算法是否能够收敛到最优解。为此，作者引入了以下概念和方法：

### **2.1 命令扩展（Command Extension）**
命令扩展是一种特殊的马尔可夫决策过程（MDP），其中命令（如目标和时间范围）被纳入状态空间。这种扩展允许将eUDRL、GCSL和ODT等算法统一在同一个框架下进行分析。

### **2.2 相对连续性（Relative Continuity）**
为了处理策略和值函数在确定性转移核附近的不连续性，作者引入了“相对连续性”的概念。这一概念允许在某些状态下策略和值函数表现出不连续性，但在整体上仍然保持某种形式的连续性。

### **2.3 段空间（Segment Space）**
论文定义了“段”（segment）的概念，表示轨迹中的一段连续状态-动作序列。通过分析段的分布，作者能够研究算法在不同迭代次数下的行为。

## **3. 实验与关键结论**
论文通过理论分析和数值实验验证了以下关键结论：

### **3.1 确定性环境下的收敛性**
在确定性环境中，eUDRL算法能够收敛到最优策略。论文通过数学证明展示了这一点，并指出在确定性环境中，策略的更新是单调的，且不会出现非最优行为。具体来说：
- **定理10**：在确定性环境中，eUDRL生成的策略在第一次迭代后即为最优。
- **定理16**：在有限迭代次数下，eUDRL生成的策略和值函数在确定性转移核附近是相对连续的。

### **3.2 随机环境下的稳定性**
在随机环境中，eUDRL算法可能会出现不稳定性。论文通过构造具体的例子展示了在某些随机环境中，即使转移核与确定性核非常接近，算法的性能也可能出现显著的下降。例如：
- **图1和图2**：展示了在随机环境中，eUDRL生成的目标到达目标（goal-reaching objective）可能表现出不连续性。

### **3.3 渐近行为**
论文还研究了eUDRL算法在无限迭代次数下的渐近行为。在某些特殊条件下（如初始分布的支持集包含所有关键状态，或者最优策略是唯一的），论文证明了算法的累积点集（accumulation points）在确定性转移核附近是连续的。具体来说：
- **定理25**：在初始分布支持集包含所有关键状态的条件下，eUDRL生成的策略在无限迭代次数下是相对连续的。
- **定理31**：在最优策略唯一的条件下，eUDRL生成的策略在无限迭代次数下是连续的。

### **3.4 正则化的影响**
论文进一步探讨了正则化对eUDRL算法稳定性的影响。通过引入ϵ-greedy正则化，作者证明了在更一般的情况下，算法的累积点集在确定性转移核附近是相对连续的。具体来说：
- **定理34**：在ϵ-greedy正则化下，eUDRL生成的策略在无限迭代次数下是相对连续的。

## **4. 研究意义**
论文的理论分析为理解eUDRL、GCSL和ODT等算法在不同环境下的行为提供了重要的理论支持。这些结论不仅有助于解释这些算法在实际应用中的表现，还为设计更稳定、更高效的RL算法提供了理论指导。此外，论文中提出的“相对连续性”概念和分析方法也为未来研究提供了新的视角。

## **5. 局限性与未来工作**
尽管论文在理论分析方面取得了重要进展，但作者也指出，对于更一般的随机环境，eUDRL算法的全局收敛性和稳定性仍然是一个开放问题。未来的研究可以进一步探索如何在更广泛的环境中保证算法的稳定性和收敛性，以及如何将这些理论结果应用于实际的复杂任务中。

## **6. 总结**
这篇论文通过引入新的理论工具和分析方法，为理解通过监督学习解决强化学习问题的算法提供了深刻的洞见。它不仅填补了现有研究中的空白，还为未来的研究提供了新的方向和方法。


# 逐段总结

## 1. 引言

本文提供了对Episodic Upside-Down Reinforcement Learning（eUDRL）、Goal-Conditioned Supervised Learning（GCSL）和Online Decision Transformers（ODT）的收敛性和稳定性的严格分析。这些算法在各种基准测试中表现出竞争力，从游戏到机器人任务，但其理论理解仅限于特定的环境条件。本文旨在为基于监督学习或序列建模的强化学习算法奠定理论基础。

### 背景知识和动机

- **强化学习（RL）**：RL算法旨在通过与环境交互来学习选择最优动作的策略。环境不直接揭示最优动作，而是通过奖励机制来引导。
- **监督学习（SL）**：SL中，每个输入都有一个正确的输出用于学习。UDRL、GCSL和ODT尝试使用SL技术来解决RL任务。
- **UDRL**：通过将期望的回报/目标映射到动作，将动作预测视为SL问题。
- **GCSL**：利用目标信息来指导模型的学习过程。
- **ODT**：利用Transformer架构来建模整个轨迹，将过去的状态、动作和奖励视为序列来预测最优动作。

### 研究问题

1. 假设给定底层马尔可夫环境的显式模型（转移核），UDRL、GCSL和ODT的收敛性如何？在无限资源的极限情况下，策略、状态和动作值等典型对象的行为如何？
2. 这些量在环境模型的扰动或误差下有多稳定？确保算法可靠地识别最优解并在不同条件下保持稳定的保证对于其在现实世界系统中的实际部署至关重要。

### 算法概述

- **eUDRL**：通过SL迭代更新策略，基于从轨迹样本中计算的命令（目标和时间范围）。
- **GCSL**：专注于状态到达任务，使用固定时间范围。
- **ODT**：将RL视为序列建模问题，使用Transformer架构来预测动作。

### 关键发现

- **收敛性**：在确定性环境中，eUDRL能够收敛到最优策略。
- **稳定性**：在非确定性环境中，eUDRL的稳定性是一个问题，尤其是在转移核的边界处。
- **连续性**：eUDRL策略在第二次迭代后变得不连续，尤其是在转移核的边界处。

## 2. 背景

### 2.1 马尔可夫决策过程（MDP）

MDP是不确定动态环境中序列决策问题的数学框架。形式上，MDP是一个五元组 $ M = (S, A, \lambda, \mu, R) $，其中：
- $ S $ 是允许的状态集。
- $ A $ 是可能的动作集。
- $ \lambda $ 是转移概率核，定义了在给定状态下采取动作后进入新状态的概率。
- $ \mu $ 是初始状态的分布。
- $ R $ 是通过转移获得的奖励。

### 2.2 命令扩展型马尔可夫决策过程（Command Extension）

在eUDRL中，目标是以“目标”和“时间范围”的形式提供给代理。目标图 $ \rho : S \to G $ 用于评估在命令的时间范围内是否达到了目标。命令扩展（CE）的定义如下：
- **命令**：命令是 $ (g, h) \in G \times \bar{N}_0 $ 的对，其中 $ G $ 是目标集，$ \bar{N}_0 = \{0, 1, ..., N\} $ 是最大时间范围。
- **扩展状态空间**：扩展状态空间 $ \bar{S} = S \times \bar{N}_0 \times G $，扩展状态 $ \bar{s} = (s, h, g) $ 由原始MDP状态 $ s $ 和命令 $ (g, h) $ 组成。
- **初始分布**：扩展MDP的初始分布由命令的分布和原始MDP的初始分布的乘积给出。
- **转移核**：定义了状态转移的概率。
- **奖励函数**：如果在时间范围结束时达到了指定目标，则授予奖励。

### 2.3 轨迹和段分布（Trajectory and Segment Distributions）

- **轨迹**：CE轨迹包含一系列连续的状态-动作转移。
- **段**：段是轨迹中的一段连续状态-动作转移，包含初始时间范围、目标和长度。

### 2.4 eUDRL算法

eUDRL通过以下步骤生成策略序列：
1. 根据当前策略生成一批轨迹。
2. 根据段分布 $ d_{\pi_n} \Sigma $ 从批中采样段。
3. 使用SL拟合新策略 $ \pi_{n+1} $ 到采样的轨迹段。

### 2.5 eUDRL与ODT的关系

ODT将RL视为序列建模问题，使用Transformer架构来预测动作。ODT与eUDRL的关系如下：
- ODT使用时间嵌入而不是位置编码。
- ODT仅在尾段（trailing segments）上操作。
- ODT引入了熵正则化，这与eUDRL中的策略正则化类似。

### 关键公式和意义

- **策略更新公式**：
  $$
  \pi_{n+1} = \arg\max_{\pi} \mathbb{E}_{\sigma \sim d_{\pi_n} \Sigma} \log \pi(a_{\sigma_0} | s_{\sigma_0}, l(\sigma), \rho(s_{\sigma_{l(\sigma)}}))
  $$
  这个公式表示新策略 $ \pi_{n+1} $ 是通过最大化采样段的对数似然来更新的。

- **段分布**：
  $$
  P(\Sigma = \sigma; \pi) = c^{-1} \sum_{t \leq N - l(\sigma)} P(S_t = s_{\sigma_0}, H_t = h_{\sigma_0}, G_t = g_{\sigma_0}, A_t = a_{\sigma_0}, \ldots, S_{t+l(\sigma)} = s_{\sigma_{l(\sigma)}}; \pi)
  $$
  这个公式表示段 $ \sigma $ 的概率分布，其中 $ c $ 是归一化常数。

### 结论

本文通过严格的数学分析，为eUDRL、GCSL和ODT的收敛性和稳定性提供了理论基础。这些算法在确定性环境中能够收敛到最优策略，但在非确定性环境中存在稳定性问题。通过引入新的数学工具，如相对连续性和固定点理论，本文为理解这些算法的行为提供了新的视角。

## 3. eUDRL递归、奖励加权回归和实际实现

### 3.1 在特定段空间中的eUDRL递归

eUDRL递归通常在整个段空间上进行。本节将展示在尾段（trailing segments）和对角线段（diagonal segments）的子空间上进行递归的公式。这将使我们能够将eUDRL与奖励加权回归（Reward-Weighted Regression, RWR）算法联系起来，RWR的已知收敛性将为后续证明提供直观指导。

#### 定义和公式

- **尾段（Trailing Segments）**：如果段的长度 $ l(\sigma) $ 等于其初始时间范围 $ h_\sigma^0 $，则该段是尾段。
- **对角线段（Diagonal Segments）**：如果段是尾段，并且其初始目标 $ g_\sigma^0 $ 在段的末尾被实现（即 $ \rho(s_\sigma^{l(\sigma)}) = g_\sigma^0 $），则该段是对角线段。

**引理3**：考虑由公式 (2.7) 描述的eUDRL中的递归策略更新。

1. **在整个段空间 $ Seg $ 上进行递归**：
   $$
   \pi_{n+1}(a | s, h, g) = P(A_\Sigma^0 = a | S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g; \pi_n)
   $$
   $$
   \propto \sum_{h' \geq h, g' \in G} P(\rho(S^h) = g | A^0 = a, H^0 = h', G^0 = g', S^0 = s; \pi_n)
   $$
   $$
   \cdot \pi_n(a | s, h', g') P(H_\Sigma^0 = h', G_\Sigma^0 = g' | S_\Sigma^0 = s, l(\Sigma) = h; \pi_n)
   $$

2. **在尾段空间 $ Seg_{trail} $ 上进行递归**：
   $$
   \pi_{trail, n+1}(a | s, h, g) = P(A_\Sigma^0 = a | S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g, l(\Sigma) = H_\Sigma^0; \pi_{trail, n})
   $$
   $$
   \propto \sum_{g' \in G} P(\rho(S^h) = g | A^0 = a, H^0 = h, G^0 = g', S^0 = s; \pi_{trail, n})
   $$
   $$
   \cdot \pi_{trail, n}(a | s, h, g') P(H_\Sigma^0 = h, G_\Sigma^0 = g' | S_\Sigma^0 = s, l(\Sigma) = h; \pi_{trail, n})
   $$

3. **在对角线段空间 $ Seg_{diag} $ 上进行递归**：
   $$
   \pi_{diag, n+1}(a | s, h, g) = P(A_\Sigma^0 = a | S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g, \rho(S_\Sigma^h) = G_\Sigma^0, l(\Sigma) = H_\Sigma^0; \pi_{diag, n})
   $$
   $$
   \propto P(\rho(S^h) = g | A^0 = a, H^0 = h, G^0 = g, S^0 = s; \pi_{diag, n})
   $$
   $$
   \cdot \pi_{diag, n}(a | s, h, g) P(H_\Sigma^0 = h, G_\Sigma^0 = g | S_\Sigma^0 = s, l(\Sigma) = h; \pi_{diag, n})
   $$
   此外，策略可以表示为Q函数的形式：
   $$
   \pi_{diag, n+1}(a | s, h, g) \propto Q^{\pi_{diag, n}}((s, h, g), a) \pi_{diag, n}(a | s, h, g)
   $$

#### 公式的意义

- **公式 (3.1)**：在整个段空间上进行递归时，新策略 $ \pi_{n+1} $ 依赖于所有可能的初始时间范围 $ h' $ 和目标 $ g' $ 的条件概率。
- **公式 (3.2)**：在尾段空间上进行递归时，新策略 $ \pi_{trail, n+1} $ 仅依赖于当前时间范围 $ h $ 和目标 $ g $ 的条件概率。
- **公式 (3.3)**：在对角线段空间上进行递归时，新策略 $ \pi_{diag, n+1} $ 仅依赖于当前时间范围 $ h $ 和目标 $ g $ 的条件概率，并且可以表示为Q函数的形式。

### 3.2 eUDRL与奖励加权回归的关系

奖励加权回归（RWR）是一种RL算法，它通过在当前策略下生成一批轨迹，并使用监督学习来拟合新策略，但动作的贡献由轨迹回报加权。RWR的更新公式为：
$$
\pi_{n+1}(a | s) \propto Q(s, a) \pi_n(a | s)
$$
在CE的上下文中，这可以表示为：
$$
\pi_{n+1}(a | s, h, g) \propto Q^{\pi_n}((s, h, g), a) \pi_n(a | s, h, g)
$$
根据引理3的第3点，这对应于eUDRL在对角线段空间 $ Seg_{diag} $ 上的更新规则。换句话说，在这种特殊情况下，eUDRL变成了RWR。然而，与一般的eUDRL不同，RWR的递归总是收敛到最优策略。

### 3.3 优先级重放的作用

eUDRL的实现中使用了优先级重放缓冲区，其中初始目标或回报从缓冲区中高回报的轨迹中采样。以下示例说明了按回报优先级重放对学习系统性能的影响：

- **性能恶化**：在某些环境中，高回报很难实现，而适度回报很容易实现。尝试实现高回报可能导致“死胡同”，即低回报。在这种情况下，优先级重放可能导致小的预期回报。
- **性能提升**：在某些环境中，只有两种可能的回报 $\{0, 1\}$，并且只有尾段。在这种情况下，优先级重放缓冲区最终会充满回报为1的轨迹。这使得算法实际上变成了RWR，从而保证收敛。

## 4. eUDRL在转移核中的连续性和稳定性

本节研究了eUDRL生成的各种量（如策略、相关值和目标达成目标）在转移核发生微小变化时的连续性。这些连续性结果将用于将eUDRL在确定性核（见下文第5节）下的已知最优性推广到在近确定性核（即核位于确定性核的邻域内）下的近最优性。

### 4.1 兼容的MDP族和状态支持的稳定性

设 $ M $ 是一个给定的MDP，$ \bar{M} $ 是其命令扩展（CE）。给定一个状态和动作，可以将转移核 $ \lambda(\cdot | s, a) $ 视为概率单纯形 $ \Delta^S $ 中的一个向量，即可以写成 $ \lambda \in (\Delta^S)^{S \times A} \subset \mathbb{R}^{S \times S \times A} $。我们研究 $ M $ 和 $ \bar{M} $ 的性质（如eUDRL生成的策略和值）如何依赖于 $ \lambda $。为了我们的目的，只需研究 $ \lambda $ 的变化对这些性质的影响，而MDP的其他部分保持不变。为此，我们定义了兼容的MDP族。

**定义4（兼容的MDP族）**：设 $ \lambda_0 $ 是一个给定的转移核，$ M = (S, A, \lambda_0, \mu, r) $ 是相应的MDP，其CE为 $ \bar{M} = (\bar{S}, A, \bar{\lambda}_0, \bar{\mu}, \bar{r}, \rho) $。记 $ \{M_\lambda = (S, A, \lambda, \mu, r) | \lambda \in (\Delta^S)^{S \times A}\} $ 为转移核属于与 $ \lambda_0 $ 相同的单纯形积 $ (\Delta^S)^{S \times A} $ 的MDP族。类似地，记 $ \{\bar{M}_\lambda = (\bar{S}, A, \bar{\lambda}, \bar{\mu}, \bar{r}, \rho) | \lambda \in (\Delta^S)^{S \times A}\} $ 为相应的CE族。我们称任何通过上述构造得到的MDP族对为兼容族。

从现在起，我们保留下标 $ \lambda $ 来指代来自给定兼容族的MDP $ M_\lambda $ 或 $ \bar{M}_\lambda $ 的量。我们偶尔会加上下标 $ \lambda $ 来强调某个量（如策略族）与MDP族相关联。

**定义5**：设 $ \{M_\lambda | \lambda \in (\Delta^S)^{S \times A}\} $ 和 $ \{\bar{M}_\lambda | \lambda \in (\Delta^S)^{S \times A}\} $ 是兼容族。

1. **eUDRL递归中的分子和分母**：对于所有 $ a \in A $ 和 $ (s, h, g) \in \bar{S}^T $，定义分子 $ \text{num}_{\lambda, \pi}(a, s, h, g) $ 和分母 $ \text{den}_{\lambda, \pi}(s, h, g) $ 为：
   $$
   \text{num}_{\lambda, \pi}(a, s, h, g) = P_\lambda(A_\Sigma^0 = a, S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g; \pi)
   $$
   $$
   \text{den}_{\lambda, \pi}(s, h, g) = P_\lambda(S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g; \pi)
   $$

2. **CE状态访问分布**：对于转移核 $ \lambda $ 和策略 $ \pi $，CE状态访问分布定义为：
   $$
   \nu_{\lambda, \pi}(\bar{s}) = \sum_{t < N} P_\lambda(\bar{S}_t = \bar{s}, \bar{S}_t \in \bar{S}^T) \quad \forall \bar{s} \in \bar{S}
   $$
   一个状态 $ \bar{s} $ 在 $ \lambda $ 和 $ \pi $ 下被访问当且仅当 $ \nu_{\lambda, \pi}(\bar{s}) > 0 $。

3. **关键状态集**：设 $ \lambda_0 \in (\Delta^S)^{S \times A} $ 是一个确定性核，$ \pi_0 > 0 $ 是一个策略。eUDRL学习的关键状态集为：
   $$
   \bar{S}_{\lambda_0} = \text{supp} \, \text{den}_{\lambda_0, \pi_0} \cap \text{supp} \, \nu_{\lambda_0, \pi_0}
   $$
   注意，特定选择的 $ \pi_0 > 0 $ 并不重要。

利用上述定义的分子和分母，eUDRL递归（2.8）可以写成：
$$
\pi_{n+1}(a | s, h, g) = \frac{\text{num}_{\lambda, \pi_n}(a, s, h, g)}{\text{den}_{\lambda, \pi_n}(s, h, g)} \quad \text{对于} \, (s, h, g) \in \text{supp} \, \text{den}_{\lambda, \pi_n}
$$
其中我们设 $ \pi_{n+1}(a | s, h, g) = \frac{1}{|A|} $ 在 $ \text{supp} \, \text{den}_{\lambda, \pi_n} $ 之外。

状态访问分布限制在瞬态状态，因为这是学习策略影响CE演变的状态集。所有被访问状态的集合等于 $ \nu_{\lambda, \pi} $ 的支持集。支持集可以简写为：
$$
\text{supp} \, \nu_{\lambda, \pi} = \{\bar{s} \in \bar{S}^T; (\exists t < N) : P_\lambda(\bar{S}_t = \bar{s}; \pi) > 0\}
$$

虽然从递归方程中很明显 $ \pi_n $ 在 $ n > 1 $ 时依赖于 $ \lambda $，但我们假设初始策略 $ \pi_0 $ 在MDP族中是常数。注意，“关键状态” $ \bar{S}_{\lambda_0} $ 的定义依赖于通过集合 $ (\Delta^S)^{S \times A} $ 指定的族。 $ \bar{S}_{\lambda_0} $ 的定义动机在于，我们讨论的非平凡行为发生在这些状态集上。事实上，选择的状态集交集确保了以下标准：i）命令实际上可以被满足（通过 $ P_{\lambda_0}(S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g; \pi_0) > 0 $ 在 $ \text{supp} \, \text{den}_{\lambda_0, \pi_0} $ 上），ii）关键状态具有非零访问概率（在 $ \text{supp} \, \nu_{\lambda_0, \pi_0} $ 上）。鉴于这些点，检查公式（2.2）可以得出，给定固定核 $ \lambda_0 $，在 $ \bar{S}_{\lambda_0} $ 之外的状态对目标达成目标没有贡献。我们将在第6节中严格讨论，其他状态的策略和值与目标达成目标的连续性无关。

**引理6（$ \bar{S}_{\lambda_0} $ 的稳定性）**：设 $ \{M_\lambda | \lambda \in (\Delta^S)^{S \times A}\} $ 和 $ \{\bar{M}_\lambda | \lambda \in (\Delta^S)^{S \times A}\} $ 是兼容族。设 $ (\pi_{n, \lambda})_{n \geq 0} $ 是由转移核 $ \lambda \in (\Delta^S)^{S \times A} $ 和初始策略 $ \pi_0 $（不依赖于 $ \lambda $）生成的策略序列。固定一个确定性转移核 $ \lambda_0 \in (\Delta^S)^{S \times A} $，则对于所有初始条件 $ \pi_0 > 0 $，有：
1. 对于所有 $ n \geq 0 $ 和所有 $ \lambda \in U_2(\lambda_0) $，我们有：
   $$
   \text{supp} \, \text{num}_{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) \subset \text{supp} \, \text{num}_{\lambda, \pi_n} \cap (A \times \text{supp} \, \nu_{\lambda, \pi_n})
   $$
   其中包含关系在 $ \lambda = \lambda_0 $ 时变为等式。

2. 对于所有 $ n \geq 0 $ 和所有 $ \lambda \in U_2(\lambda_0) $，我们有：
   $$
   \bar{S}_{\lambda_0} \subset \text{supp} \, \text{den}_{\lambda, \pi_n} \cap \text{supp} \, \nu_{\lambda, \pi_n}
   $$
   其中包含关系在 $ \lambda = \lambda_0 $ 时变为等式。

3. 对于所有 $ n \geq 0 $ 和所有 $ \lambda \in U_2(\lambda_0) $，我们有：
   $$
   P_\lambda(S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g, H_\Sigma^0 = h, G_\Sigma^0 = g; \pi_n) > 0
   $$
   对于所有 $ (s, h, g) \in \bar{S}_{\lambda_0} $。

4. 对于所有 $ (a, s, h, g) \in \text{supp} \, \text{num}_{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) $，$ h > 1 $，存在 $ s' \in S $ 使得 $ \lambda_0(s' | s, a) > 0 $ 且 $ (s', h - 1, g) \in \bar{S}_{\lambda_0} $。

5. 对于所有 $ (a, s, h, g) \notin \text{supp} \, \text{num}_{\lambda_0, \pi_0} $ 且 $ (s, h, g) \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $，我们有对于任何策略 $ \pi $，$ Q_\pi^{\lambda_0}((s, h, g), a) = 0 $。

**证明**：假设 $ \lambda_0 $ 是确定性的，以下包含关系成立：
$$
\forall \lambda \in U_2(\lambda_0) : \text{supp} \, \lambda_0 \subset \text{supp} \, \lambda
$$
这意味着通过核 $ \lambda_0 $ 可达的任何状态也可以通过 $ \lambda $ 达到。由于 $ \lambda_0 $ 是确定性的，存在唯一的 $ s' $ 使得 $ 1 = \lambda_0(s' | s, a) $。只需证明 $ \lambda(s' | s, a) $ 是非零的。我们有：
$$
2 > \|\lambda(\cdot | s, a) - \lambda_0(\cdot | s, a)\|_1 = |\lambda(s' | s, a) - \lambda_0(s' | s, a)| + \sum_{s'' \neq s'} |\lambda(s'' | s, a) - \lambda_0(s'' | s, a)|
$$
$$
= 1 - \lambda(s' | s, a) + \sum_{s'' \neq s'} \lambda(s'' | s, a) = 2(1 - \lambda(s' | s, a))
$$
这等价于 $ \lambda(s' | s, a) > 0 $。

1. 假设 $ (a, s, h, g) \in \text{supp} \, \text{num}_{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) $，则存在一个轨迹 $ \tau' = ((s'_0, H', g'), a'_0, (s'_1, H' - 1, g'), a'_1, \ldots) $ 使得 $ P_{\lambda_0}(T = \tau'; \pi_0) > 0 $，并且对于某个 $ t' \leq N - h $，我们有 $ s'_{t'} = s $，$ a'_{t'} = a $，$ H' - t' \geq h $ 且 $ \rho(s'_{t' + h}) = g $。我们声称存在一个轨迹 $ \tau = ((s_0, H, g), a_0, (s_1, H - 1, g), \ldots) $ 和 $ t \leq N $ 使得 $ P_{\lambda_0}(T = \tau; \pi_0) > 0 $，并且 $ s_t = s $，$ a_t = a $，$ h = H - t $ 且 $ \rho(s_{t + h}) = g $。存在一个轨迹的前缀 $ ((s_0, H, g), a_0, \ldots, (s_t, H - t, g)) $ 使得 $ P_{\lambda_0}(T = \tau; \pi_0) > 0 $，因为 $ (s_t, H - t, g) = (s, h, g) \in \bar{S}_{\lambda_0} \subset \text{supp} \, \nu_{\lambda_0, \pi_0} $。对于轨迹的后缀，我们取 $ \tau' $ 的后缀（从 $ t' $ 开始），并调整地平线和目标组件，使其与 $ \tau $ 在 $ t $ 处的组件一致。整个轨迹仍然具有正概率（在 $ \lambda_0, \pi_0 $ 下），即：
   $$
   P_{\lambda_0}(A_t = a_t, \bar{S}_{t+1} = (s_{t+1}, H - t + 1, g), A_{t+1} = a_{t+1}, \ldots | \bar{S}_t = (s_t, H - t, g); \pi_0) > 0
   $$
   因为我们保留了原始MDP状态组件，并且可以使用 $ \lambda_0(s'_{t' + i + 1} | s'_{t' + i}, a'_{t' + i}) $，$ 0 \leq i \leq h - 1 $ 和 $ \pi_0 > 0 $。这证明了 $ \tau $ 的存在。利用包含关系（4.2），我们有 $ P_\lambda(T = \tau; \pi_0) > 0 $ 对于所有 $ \lambda \in U_\delta(\lambda_0) $。假设 $ \lambda \in U_\delta(\lambda_0) $ 固定，我们通过归纳法证明：
   $$
   P_\lambda(T = \tau; \pi_n) > 0
   $$
   对于 $ n = 0 $ 的情况已经建立。假设它对 $ n \geq 0 $ 成立。由于轨迹 $ \tau $ 的概率是正的（在 $ \lambda, \pi_n $ 下），我们得到 $ \forall i : P_\lambda(A_\Sigma^0 = a_i, S_\Sigma^0 = s_i, l(\Sigma) = H - i, \rho(S_\Sigma^{H - i}) = g; \pi_n) > 0 $。因此，$ \pi_{n+1}(a_i | s_i, H - i, g) = P_\lambda(A_\Sigma^0 = a_i | S_\Sigma^0 = s_i, l(\Sigma) = H - i, \rho(S_\Sigma^{H - i}) = g; \pi_n) > 0 $ 对于所有 $ i < H $。由于转移核 $ \lambda $ 和初始分布 $ \mu $ 保持不变，$ P_\lambda(T = \tau; \pi_{n+1}) > 0 $ 跟随。从（4.3）可以得出 $ \text{num}_{\lambda, \pi_n}(a, s, h, g) > 0 $ 和 $ \nu_{\lambda, \pi_n}(s, h, g) > 0 $，并且我们看到 $ P_\lambda(A_\Sigma^0 = a, S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g, H_\Sigma^0 = h, G_\Sigma^0 = g; \pi_n) > 0 $。这完成了包含关系的证明。为了在 $ \lambda = \lambda_0 $ 时建立等式，只需证明 $ \text{supp} \, \text{num}_{\lambda_0, \pi_n} \subset \text{supp} \, \text{num}_{\lambda_0, \pi_0} $ 和 $ \text{supp} \, \nu_{\lambda_0, \pi_n} \subset \text{supp} \, \nu_{\lambda_0, \pi_0} $。但由于假设 $ \pi_0 $ 具有最大支持，这跟随而来。

2. 这立即从第1点和对于任何 $ \lambda $：
   $$
   \forall n \geq 0, \forall (s, h, g) \in \bar{S}^T : (s, h, g) \in \text{supp} \, \text{den}_{\lambda, \pi_n} \iff (\exists a \in A : (a, s, h, g) \in \text{supp} \, \text{num}_{\lambda, \pi_n})
   $$

3. 事实 $ P_\lambda(S_\Sigma^0 = s, l(\Sigma) = h, \rho(S_\Sigma^h) = g; H_\Sigma^0 = h, G_\Sigma^0 = g, \pi_n) > 0 $ 跟随于不等式（4.3）。

4. 这立即从第1点的证明中构造的轨迹 $ \tau $ 的存在性得出。

5. 为了矛盾，假设 $ Q_\pi^{\lambda_0}((s, h, g), a) > 0 $ 对于某个 $ \pi $。由 $ \pi_0 $ 的最大支持性，我们有 $ Q_{\pi_0}^{\lambda_0}((s, h, g), a) > 0 $。从 $ (s, h, g) \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $ 和第2点的等价性，我们推导出存在一个轨迹 $ \tau' $ 具有正概率（在 $ \lambda_0, \pi_0 $ 下），如第1点所述。我们只需要它的前缀 $ P_{\lambda_0}(S_0 = s'_0, H_0 = H', G_0 = g', \ldots, S_{t'} = s, H_{t'} = h', G_{t'} = g'; \pi_0) > 0 $，$ h' \geq h $。由于 $ 0 < Q_{\pi_0}^{\lambda_0}((s, h, g), a) = P_{\lambda_0}(\rho(S_h) = g | S_0 = s, H_0 = h, G_0 = g, A_0 = a; \pi_0) $，我们有 $ P_{\lambda_0}(\rho(S_{t' + h}) = g | S_{t'} = s, H_{t'} = h', G_{t'} = g', A_{t'} = a; \pi_0) > 0 $，其中我们使用了 $ \pi_0 $ 的支持忽略地平线和目标组件。同样地，$ \pi_0(a | s, h', g') > 0 $。将这些结合起来，我们得出存在一个轨迹 $ ((s'_0, H', g'), \ldots, (s, h', g'), a, \ldots, (s_{t' + h}, h' - h, g'), \ldots) $ 使得 $ \rho(s_{t' + h}) = g $ 且具有正概率（在 $ \lambda_0, \pi_0 $ 下），这证明了 $ \text{num}_{\lambda_0, \pi_0}(a, s, h, g) > 0 $ 并与我们的假设矛盾。

### 4.2 在段子空间中的支持稳定性

第4.1节中关于兼容族中关键状态集的稳定性讨论可以相应地转化为对段子空间 $ Seg_{trail} $ 和 $ Seg_{diag} $ 中eUDRL递归的讨论。受eUDRL类型迭代在ODT或RWR算法中的理论描述的启发，我们简要概述了这些稳定性属性。

设 $ \text{num}_{\text{trail}/\text{diag}}^{\lambda, \pi} $ 和 $ \text{den}_{\text{trail}/\text{diag}}^{\lambda, \pi} $ 分别表示eUDRL递归在 $ Seg_{trail} $ 和 $ Seg_{diag} $ 上的分子和分母。这些量通过引入定义相应段子空间的条件来修改。例如，对于所有 $ a \in A $ 和 $ (s, h, g) \in \bar{S}^T $，我们设：
$$
\text{num}_{\text{diag}}^{\lambda, \pi}(a, s, h, g) := P_\lambda(A_\Sigma^0 = a, S_\Sigma^0 = s, l(\Sigma) = h, S_\Sigma^h = g | l(\Sigma) = H_\Sigma^0, \rho(S_\Sigma^{l(\Sigma)}) = G_\Sigma^0; \pi)
$$
$$
\text{den}_{\text{diag}}^{\lambda, \pi}(s, h, g) := P_\lambda(S_\Sigma^0 = s, l(\Sigma) = h, S_\Sigma^h = g | l(\Sigma) = H_\Sigma^0, \rho(S_\Sigma^{l(\Sigma)}) = G_\Sigma^0; \pi)
$$

eUDRL递归在 $ Seg_{trail} $ 和 $ Seg_{diag} $ 上的形式与公式（4.1）相同，只是将 $ \pi_n $ 改为 $ \pi_{\text{diag}/\text{trail}, n} $，$ \text{num} $ 改为 $ \text{num}_{\text{diag}/\text{trail}} $，$ \text{den} $ 改为 $ \text{den}_{\text{diag}/\text{trail}} $。

**引理7（在尾部和对角线子空间中 $ \bar{S}_{\lambda_0} $ 的稳定性）**：在引理6的设置中，考虑eUDRL递归在 $ Seg_{diag} $ 和 $ Seg_{diag} $ 上的限制。则关键状态集 $ \bar{S}_{\lambda_0} $ 保持不变，引理6的结论在将 $ \pi_n $ 改为 $ \pi_{\text{diag}/\text{trail}, n} $，$ \text{num} $ 改为 $ \text{num}_{\text{diag}/\text{trail}} $，$ \text{den} $ 改为 $ \text{den}_{\text{diag}/\text{trail}} $ 后仍然有效。

**证明**：我们遵循引理6的证明，只突出差异。考虑任何 $ (a, s, h, g) \in \text{supp} \, \text{num}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) $，它包含在轨迹 $ \tau' $ 中。我们构造一个包含 $ (a, s, h, g) $ 的对角线段的轨迹 $ \tau $，且具有正概率。由于对角线/尾部段也包含在 $ Seg $ 中，我们可以采用之前的方法。作为构造 $ \tau $ 的副产品，我们得出：
$$
\text{supp} \, \text{num}_{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) = \text{supp} \, \text{num}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0})
$$
因此：
$$
\bar{S}_{\lambda_0} = \text{supp} \, \text{den}_{\lambda_0, \pi_0} \cap \text{supp} \, \nu_{\lambda_0, \pi_0} = \text{supp} \, \text{den}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0} \cap \text{supp} \, \nu_{\lambda_0, \pi_0}
$$
换句话说，集合 $ \bar{S}_{\lambda_0} $ 和其状态-动作变体 $ \text{num}_{\lambda_0, \pi_0} \cap (A \times \text{supp} \, \nu_{\lambda_0, \pi_0}) $ 保持不变。要证明 $ P_\lambda(T = \tau; \pi_{\text{diag}/\text{trail}, n}) > 0 $，只需意识到用于证明 $ \pi_{n+1}(a_i | s_i, H - i, g) > 0 $ 的段是对角线段，即 $ \pi_{\text{diag}/\text{trail}, n+1}(a_i | s_i, H - i, g) > 0 $。第1点和第2、3、4点的其余部分类似。在第5点的证明中，我们利用了我们从对角线/尾部段开始的事实来证明 $ \text{num}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0}(a, s, h, g) > 0 $ 并得出矛盾。


## 5. 确定性核的最优策略的收敛性

虽然eUDRL递归在确定性环境下可以收敛到最优策略，但在存在随机性的情况下可能会导致次优行为。这一事实早在Schmidhuber（2019年12月5日）的早期论文中就已明显，其中在随机环境下，eUDRL递归被替换为一个基于预期回报而非实际回报的算法。Ghosh等人（2021）提到GCSL可以收敛到最优策略。Brandfonbrener等人（2022）也讨论了这一相关事实，重点研究了离线RL中第一个UDRL迭代的返回到达任务，证明了在确定性环境下，预期回报与实际回报一致。GCSL收敛到最优策略的事实相对简单，通常省略完整证明。本文提供了完整证明，因为后续将基于这一结果，并且介绍我们的符号是方便的。我们首先在第5.1节中描述了在确定性环境下的CE的最优动作集。随后在第5.2节和第5.3节中，我们展示了这一集合对应于eUDRL递归过程中生成的策略的支持集。因此，对于 $ n \geq 1 $，$ \pi_n $ 的支持集是恒定的，这表明在确定性环境下，eUDRL生成的策略在 $ n \geq 1 $ 时是最优的。

### 5.1 确定性转移核的最优动作

本节的目标是确定由具有给定确定性转移核的MDP生成的CE的最优动作集。我们首先定义最优动作的概念。

**定义8（最优动作）**：设 $ M = (S, A, \lambda, \mu, r) $ 是一个MDP。如果存在一个策略 $ \pi $，使得 $ \pi(a|s) > 0 $ 并且 $ \pi $ 在状态 $ s $ 中达到最大（最优）值，则动作 $ a \in A $ 在状态 $ s \in S $ 中被称为最优的。我们用 $ O(s) $ 表示状态 $ s $ 中的最优动作集。

需要注意的是，由于CE是MDP的一个特例，这一定义包括了CE中的最优动作。因此，在CE的上下文中，任何最大化实现预期命令概率的动作都是最优的。

**引理9（确定性核的CE的最优动作集）**：设 $ \lambda_0 $ 是一个给定的确定性转移核，$ M = (S, A, \lambda_0, \mu, r) $ 是相应的MDP，其CE为 $ \bar{M} = (\bar{S}, A, \bar{\lambda}_0, \bar{\mu}, \bar{r}, \rho) $，考虑一个策略 $ \pi_0 > 0 $。对于任何状态 $ \bar{s} \in \text{supp} \, \text{den}_{\lambda_0, \pi_0}(\cdot, \bar{s}) $，最优动作集为 $ O(\bar{s}) = \text{supp} \, \text{num}_{\lambda_0, \pi_0}(\cdot, \bar{s}) $。

引理9识别了确定性核的一般CE算法的最优动作集，但没有特别提到eUDRL。需要注意的是，这里涉及的量与段空间 $ Seg $ 相关，而 $ Seg $ 可以在不使用eUDRL的情况下构造。在后续的连续性讨论中，我们将研究在某个固定的确定性核 $ \lambda_0 $ 附近改变 $ \lambda $ 的影响。我们将省略 $ O_{\lambda_0}(\bar{s}) $ 的索引，用 $ O(\bar{s}) $ 表示在固定的 $ \lambda_0 $ 下的最优动作（我们不会研究核变化时的 $ O_{\lambda}(\bar{s}) $）。

**证明**：我们从反向推论开始。固定一个状态 $ \bar{s} \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $ 和一个动作 $ a \in O(\bar{s}) $。由于 $ \text{num}_{\lambda_0, \pi_0}(a, \bar{s}) > 0 $，存在一个在 $ \pi_0 $ 下具有正概率的轨迹 $ \tau' = ((s_0, H', g'), a_0, (s_1, H' - 1, g'), a_1, \ldots) $，其中 $ s_t = s $，$ a_t = a $，$ \rho(s_{t+h}) = g $，且 $ t + h \leq H' $。此外，由于 $ \lambda_0 $ 是确定性的，对于 $ i < H' $，有 $ \lambda_0(s_{i+1} | s_i, a_i) = 1 $。由于 $ Q((s_{t+h-1}, 1, g), a_{t+h-1}) = \sum_{s' \in \rho^{-1}(\{g\})} \lambda_0(s' | s_{t+h-1}, a_{t+h-1}) = \lambda_0(s_{t+h} | s_{t+h-1}, a_{t+h-1}) = 1 $ 是最大值，因此 $ a_{t+h-1} $ 在 $ (s_{t+h-1}, 1, g) $ 中是最优的。因此，$ \pi^*(a_{t+h-1} | s_{t+h-1}, 1, g) = 1 $ 是该状态下的最优策略。此外，
$$
V_{\pi^*}(s_{t+h-1}, 1, g) = \sum_{a \in A} Q((s_{t+h-1}, 1, g), a) \pi^*(a | s_{t+h-1}, 1, g) = Q((s_{t+h-1}, 1, g), a_{t+h-1}) \pi^*(a_{t+h-1} | s_{t+h-1}, 1, g) = 1,
$$
并且
$$
Q_{\pi^*}(s_{t+h-2}, 2, g, a_{t+h-2}) = \sum_{s' \in S} V_{\pi^*}(s', 1, g) \lambda_0(s' | s_{t+h-2}, a_{t+h-2}) = V_{\pi^*}(s_{t+h-1}, 1, g) \lambda_0(s_{t+h-1} | s_{t+h-2}, a_{t+h-2}) = 1,
$$
这再次实现了最大可能值。因此，$ a_{t+h-2} $ 在 $ (s_{t+h-2}, 2, g) $ 中是最优的。重复这一过程 $ h $ 次，可以得出 $ Q_{\pi^*}(s_t, h, g, a_t) = 1 $，$ a_t $ 在 $ (s_t, h, g) $ 中是最优的，且值为 $ V_{\pi^*}(s_t, h, g) = 1 $。

对于正向推论，设 $ (s, h, g) \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $，假设 $ a $ 在 $ (s, h, g) $ 中是最优的。按照上述反向推论的证明过程，我们可以观察到 $ (s, h, g) $ 实现了最大值 1。由于 $ a $ 是最优的，存在一个策略 $ \pi^* $，在状态 $ s $ 下，通过动作 $ a $ 在 $ h $ 步内达到目标 $ g $。由于 $ \pi_0 > 0 $，可以得出 $ \pi_0 $ 也在状态 $ s $ 下，通过动作 $ a $ 在 $ h $ 步内达到目标 $ g $，即 $ P_{\lambda_0}(\rho(S_h) = g, A_0 = a | S_0 = s, H_0 = h, G_0 = g; \pi_0) > 0 $。由于 $ (s, h, g) \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $，我们可以类似地在引理6的第1点的证明中找到一个轨迹 $ \tau' = ((s'_0, H', g'), \ldots, (s, h', g'), \ldots) $，其中 $ h' \geq h $，在 $ \lambda_0 $ 和 $ \pi_0 $ 下具有正概率。因此，$ P_{\lambda_0}(\rho(S_h) = g, A_0 = a | S_0 = s, H_0 = h', G_0 = g'; \pi_0) > 0 $。结合 $ \tau' $ 的适当前缀，我们可以得出存在一个轨迹，证明 $ \text{num}_{\lambda_0, \pi_0}(a, s, h, g) > 0 $。

### 5.2 在所有段空间中的eUDRL的收敛性

在以下引理中，我们利用最优动作的概念来证明，如果转移核是确定性的，eUDRL生成的策略 $ (\pi_n)_{n \geq 1} $ 是最优的。

**引理10（确定性转移核的eUDRL策略的最优性）**：设 $ \lambda_0 $ 是一个给定的确定性转移核，$ M = (S, A, \lambda_0, \mu, r) $ 是相应的MDP，其CE为 $ \bar{M} = (\bar{S}, A, \bar{\lambda}_0, \bar{\mu}, \bar{r}, \rho) $。假设 $ \pi_0 > 0 $，令 $ (\pi_n)_{n \geq 0} $ 是由 $ \pi_0 $ 和 $ \lambda_0 $ 生成的eUDRL迭代的策略序列。则有：
1. 对于所有 $ n \geq 1 $ 和所有 $ \bar{s} \in \bar{S}_{\lambda_0} $，$ \pi_{n+1}(\cdot | \bar{s}) $ 的支持集与最优动作集 $ O(\bar{s}) $ 相同。特别是，策略 $ \pi_n $ 在 $ \bar{S}_{\lambda_0} $ 上是最优的。
2. 对于所有 $ n \geq 1 $ 和所有 $ \bar{s} \in \bar{S}_{\lambda_0} $，值是最大的，即
$$
V_{\pi_n}(\bar{s}) = 1.
$$
3. 对于所有 $ n \geq 1 $，所有 $ \bar{s} \in \bar{S}_{\lambda_0} $，以及所有 $ a \in A $，动作值函数为
$$
Q_{\pi_n}(\bar{s}, a) = \begin{cases} 1 & \text{如果 } a \in O(\bar{s}), \\ 0 & \text{否则}. \end{cases}
$$

**证明**：
1. 设 $ n \geq 0 $，$ (s, h, g) \in \bar{S}_{\lambda_0} (= \text{supp} \, \text{den}_{\lambda_0, \pi_n} \cap \text{supp} \, \nu_{\lambda_0, \pi_n}) $。我们有 $ a \in O(s, h, g) $ 当且仅当 $ \text{num}_{\lambda_0, \pi_0}(a, s, h, g) > 0 $。根据引理6的第1点，这等价于 $ \text{num}_{\lambda_0, \pi_n}(a, s, h, g) > 0 $，因此
$$
\pi_{n+1}(a | s, h, g) = \frac{\text{num}_{\lambda_0, \pi_n}(a, s, h, g)}{\text{den}_{\lambda_0, \pi_n}(s, h, g)} > 0 \quad \text{当且仅当 } a \in O(s, h, g).
$$
因此，$ \pi_{n+1}(\cdot | s, h, g) $ 的支持集与最优动作集相同。

2. 设 $ n \geq 1 $，$ (s, h, g) \in \bar{S}_{\lambda_0} \subset \text{supp} \, \text{den}_{\lambda_0, \pi_0} $。与引理9的证明类似，$ V^*(s, h, g) = 1 $，从第1点可知，$ \pi_n $ 在 $ \bar{S}_{\lambda_0} $ 上是最优的。因此，$ V_{\pi_n}(s, h, g) = 1 $。

3. 设 $ n \geq 1 $，$ (s, h, g) \in \bar{S}_{\lambda_0} $。首先假设 $ a \in O(s, h, g) $。根据引理6的第4点，存在 $ s' \in S $ 使得 $ \lambda_0(s' | s, a) = 1 $ 且 $ (s', h - 1, g) \in \bar{S}_{\lambda_0} $。根据第2点，$ V_{\pi_n}(s', h - 1, g) = 1 $，因此
$$
Q_{\pi_n}((s, h, g), a) = \sum_{s'' \in S} V_{\pi_n}(s'', h - 1, g) \lambda_0(s'' | s, a) = 1.
$$
如果 $ a \notin O(s, h, g) $，则根据引理6的第5点，结论成立。

### 5.3 在段子空间中的eUDRL的收敛性

在CE的上下文中，最优动作集 $ O(\bar{s}) $，$ \bar{s} \in \text{supp} \, \text{den}_{\lambda_0, \pi_0} $ 通过量 $ \text{num}_{\lambda_0, \pi_0} $ 和 $ \text{den}_{\lambda_0, \pi_0} $ 与段空间 $ Seg $ 上的分布相关，参见引理9。然而，从（5.1）可以得出，对于所有 $ \bar{s} \in \bar{S}_{\lambda_0} $，
$$
O(\bar{s}) = \text{supp} \, \text{num}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0}(\cdot, \bar{s}),
$$
即使算法在 $ Seg_{\text{diag}} $ 或 $ Seg_{\text{trail}} $ 上运行，例如在RWR的情况下，$ \text{supp} \, \text{num}_{\text{diag}/\text{trail}}^{\lambda_0, \pi_0}(\cdot, \bar{s}) $ 仍然是最优动作集。引理10在这些段子空间中的变体仍然有效，只是需要进行相应的修改。

**引理11（在 $ Seg_{\text{diag}} $ 或 $ Seg_{\text{trail}} $ 中确定性核的eUDRL策略的最优性）**：在引理10的设置中，考虑eUDRL递归在 $ Seg_{\text{diag}} $ 或 $ Seg_{\text{trail}} $ 上的限制。在将 $ \pi_n $ 改为 $ \pi_{\text{diag}/\text{trail}, n} $（参见（3.3），（3.2））后，引理的结论仍然有效。

**证明**：证明过程与引理10的证明类似，只是应用了引理7（而不是引理6）和方程（4.5），（4.4）。


### 第6节：UDRL、GCSL和ODTs的收敛性和稳定性

#### 6.1 eUDRL在确定性转移核下的连续性

**内容**：
本节讨论了eUDRL（一种强化学习算法）生成的策略和目标达成目标函数在有限迭代次数下的连续性问题。特别关注了在确定性转移核边界上的行为。

**关键点**：
1. **非连续性问题**：eUDRL生成的策略和目标达成目标函数在转移核的边界上通常不是连续的，尤其是在确定性转移核处。
2. **图2的说明**：通过图2展示了在确定性和非确定性转移核边界附近，策略和目标达成目标函数的连续性和不连续性行为。
3. **连续性证明的难点**：由于策略在最优性附近的非唯一性，导致了连续性证明的复杂性。

**公式内容及意义**：
- **公式（6.1）**：
  $$
  \pi_{n+1}(a|s, h, g) = \frac{u(a)}{\sum_{a' \in A} u(a')}
  $$
  其中，
  $$
  u(a) = \sum_{h' \ge h, g' \in G} P_\lambda(\rho(S_h) = g | A_0 = a, S_0, H_0 = h', G_0 = g', l(\Sigma) = h; \pi_n) \cdot \pi_n(a|s, h', g') \cdot P_\lambda(H_0 = h', G_0 = g' | S_0, l(\Sigma) = h; \pi_n)
  $$
  这个公式描述了eUDRL策略更新的递归关系，$u(a)$表示在给定状态下选择动作$a$的得分，而$\pi_{n+1}(a|s, h, g)$是根据这些得分归一化后的策略。


#### 6.2 eUDRL策略的相对连续性及相关值的连续性

**内容**：
本节引入了相对连续性的概念，并证明了eUDRL生成的策略在确定性转移核处具有相对连续性。同时，还证明了在有限迭代次数下，目标达成目标函数在确定性转移核处的连续性。

**关键点**：
1. **相对连续性定义**：定义了相对连续性的概念，用于描述在某些特定条件下，函数的极限行为。
2. **定理16**：证明了eUDRL生成的策略在确定性转移核处具有相对连续性，并且目标达成目标函数在这些点上是连续的。
3. **技术难点**：证明过程中需要处理策略在最优性附近的非唯一性问题。

**公式内容及意义**：
- **定理16（相对连续性）**：
  1. 对于所有$\bar{s} \in \bar{S}_{\lambda_0}$，策略$\pi_{n,\lambda}(\cdot|\bar{s})$在$\lambda_0$处相对于$O(\bar{s})$是相对连续的。
  2. 对于所有$\bar{s} \in \bar{S}_{\lambda_0}$和所有动作$a \in A$，值函数$Q^{\pi_{n,\lambda}}(\bar{s}, a)$和$V^{\pi_{n,\lambda}}(\bar{s})$在$\lambda_0$处是连续的。
  3. 存在一个序列的邻域$(U_{\delta_n}(\lambda_0))_{n \ge 0}$，使得对于所有$n \ge 0$，策略$\pi_{n,\lambda}(a|\bar{s})$在$(a, \bar{s}, \lambda) \in (\text{supp num}_{\lambda_0,\pi_0} \cap (A \times \text{supp } \nu_{\lambda_0,\pi_0})) \times U_{\delta_n}(\lambda_0)$上与0保持距离。


#### 6.3 将连续性结果扩展到其他片段子空间

**内容**：
本节将定理16的结果扩展到其他片段子空间，例如Segtrail和Segdiag，以覆盖更多算法（如ODT和RWR）。

**关键点**：
1. **定理17**：证明了在片段子空间Segtrail和Segdiag下，eUDRL生成的策略和值函数在确定性转移核处的相对连续性。
2. **技术方法**：通过修改定理16的证明，将结果推广到其他片段子空间。

**公式内容及意义**：
- **定理17（片段子空间的相对连续性）**：
  - 对于片段子空间Segtrail和Segdiag，定理16的结果仍然成立。具体来说，策略$\pi_{\text{diag/trail}, n}$在确定性转移核处具有相对连续性，值函数在这些点上是连续的。

#### 6.4 有限迭代次数下目标达成目标函数的连续性

**内容**：
本节讨论了eUDRL生成的目标达成目标函数在有限迭代次数下的连续性，并证明了其在确定性转移核处的连续性。

**关键点**：
1. **推论18**：证明了eUDRL生成的目标达成目标函数在确定性转移核处是连续的。
2. **推论19**：表明在确定性转移核附近，eUDRL生成的策略的目标达成目标函数接近最优值。

**公式内容及意义**：
- **推论18（目标达成目标函数的连续性）**：
  $$
  J^{\pi_{n,\lambda}}_\lambda = \sum_{\bar{s} \in \bar{S}_T} \bar{\mu}(\bar{s}) V^{\pi_{n,\lambda}}_\lambda(\bar{s})
  $$
  这个公式表示目标达成目标函数是初始状态分布$\bar{\mu}$和值函数$V^{\pi_{n,\lambda}}_\lambda$的加权和。推论证明了这个函数在确定性转移核处是连续的。

- **推论19（近最优行为）**：
  对于任意固定的$n$和所有$\epsilon > 0$，存在$\delta > 0$，使得当$\lambda \in U_\delta(\lambda_0)$时，
  $$
  |J^{\pi_{n,\lambda}}_\lambda - J^{\pi^*}_{\lambda_0}| < \epsilon
  $$
  这表明在确定性转移核的邻域内，eUDRL生成的策略的目标达成目标函数接近最优值。



### 总结
本节通过引入相对连续性的概念，证明了eUDRL生成的策略和目标达成目标函数在确定性转移核处的连续性。这些结果为理解eUDRL算法在不同环境下的行为提供了理论基础，并为进一步研究其收敛性和稳定性奠定了基础。



## 第7节：eUDRL在确定性转移核下的连续性（迭代次数趋于无穷）

### 7.1 eUDRL在确定性转移核下的渐近连续性

**内容**：
本节研究了eUDRL生成的策略、值函数和目标达成目标函数在迭代次数趋于无穷时的相对连续性。特别关注了在确定性转移核附近的渐近行为。尽管目前尚不清楚eUDRL是否总是收敛到一个极限策略，但作者通过分析策略的积累点集，证明了其相对连续性。

**关键点**：
1. **渐近连续性分析**：通过引入“策略积累点集的相对连续性”概念，研究eUDRL在确定性转移核附近的渐近行为。
2. **特殊初始分布**：当初始分布的支持集包含$\bar{S}_{\lambda_0}$时，证明了策略积累点集的相对连续性。
3. **技术方法**：通过限制讨论到两种特殊情况（初始分布包含$\bar{S}_{\lambda_0}$和最优策略在$\bar{S}_{\lambda_0}$上唯一），推导出eUDRL策略递归的上下界。


### 7.1.1 预备引理

**内容**：
本小节提供了几个预备引理，用于后续的连续性分析。

**关键点**：
1. **引理20（值函数的连续性）**：证明了在确定性转移核附近，值函数的差异可以通过转移核的差异进行量化。
   - **公式内容**：
     $$
     \text{如果 } V^*_{{\lambda_0}}(\bar{s}) = 0, \text{ 则 } V^\pi_{\lambda}(\bar{s}) \leq \frac{\delta^h}{2} \leq \frac{\delta^N}{2}.
     $$
   - **意义**：该引理为值函数的连续性提供了显式的量化估计。

2. **引理21（动作值函数的连续性）**：在相对连续性假设下，证明了动作值函数$Q^\pi_{\lambda}$在商拓扑下的连续性。
   - **公式内容**：
     $$
     |Q^\pi_{\lambda}(\bar{s}, a) - Q^*_{{\lambda_0}}(\bar{s}, a)| \leq \|\lambda(\cdot|s, a) - \lambda_0(\cdot|s, a)\|_1 + \max_{\bar{s}' \in \bar{S}_{\lambda_0}, a' \in O(\bar{s}')} |Q^\pi_{\lambda}(\bar{s}', a') - Q^*_{{\lambda_0}}(\bar{s}', a')|.
     $$
   - **意义**：该引理为后续证明提供了动作值函数的递归估计。

3. **引理22（访问概率的下界）**：证明了在初始分布支持集包含$\bar{S}_{\lambda_0}$的情况下，状态访问概率有正的下界。
   - **公式内容**：
     $$
     P_{\lambda}(H_{\Sigma_0} = h, G_{\Sigma_0} = g | S_{\Sigma_0} = s, l(\Sigma) = h; \pi) > \alpha,
     $$
     其中$\alpha = \frac{2}{N(N + 1)} \min_{\bar{s} \in \bar{S}_{\lambda_0}} \bar{\mu}(\bar{s}) > 0$。
   - **意义**：该引理为后续分析提供了访问概率的正下界，确保了策略递归的稳定性。



### 7.1.2 主定理

**内容**：
本小节证明了eUDRL生成的策略积累点集在确定性转移核附近的相对连续性。

**关键点**：
1. **定理25（策略积累点集的相对连续性）**：当初始分布的支持集包含$\bar{S}_{\lambda_0}$时，证明了eUDRL生成的策略积累点集在确定性转移核附近的相对连续性。
   - **公式内容**：
     $$
     \liminf_{n \to \infty} \pi_{n,\lambda}(O(\bar{s})|\bar{s}) \geq x^*(\gamma),
     $$
     其中$x^*(\gamma) = 1 - \gamma$，且$\gamma = \frac{\tilde{\beta}}{(1 - \beta)\alpha} < 1$。
   - **意义**：该定理表明，随着迭代次数的增加，策略在最优动作上的概率接近1，且这一性质在确定性转移核附近是连续的。

2. **技术方法**：
   - 通过分析由有理函数$f_{\gamma}(x) = \frac{x}{x + \gamma}$诱导的动态系统，证明了策略积累点集的收敛性。
   - 利用引理24（$f$-引理）的性质，证明了策略积累点集的渐近行为。


### 7.1.3 扩展到其他片段子空间

**内容**：
本小节将定理25的结果扩展到其他片段子空间（如Segdiag和Segtrail），以覆盖更多算法（如ODT和RWR）。

**关键点**：
1. **定理26（片段子空间的相对连续性）**：定理25的结果在片段子空间Segdiag和Segtrail下仍然成立。
   - **技术方法**：通过修改定理25的证明，将结果推广到其他片段子空间。


### 7.1.4 策略积累点的估计

**内容**：
本小节提供了eUDRL生成的策略、值函数和目标达成目标函数的显式估计，量化了这些量在确定性转移核附近的渐近行为。

**关键点**：
1. **推论27（策略积累点的估计）**：提供了策略、动作值函数、状态值函数和目标达成目标函数的显式估计。
   - **公式内容**：
     - 策略估计：
       $$
       \limsup_{n \to \infty} \max_{\bar{s} \in \bar{S}_{\lambda_0}} 2(1 - \pi_{n,\lambda}(O(\bar{s})|\bar{s})) \leq \kappa_N,
       $$
       其中$\kappa_N = 2(1 - x^*(\gamma_N))$。
     - 动作值函数估计：
       $$
       \limsup_{n \to \infty} \max_{\bar{s} \in \bar{S}_{\lambda_0}, a \in A} |Q^{\pi_{n,\lambda}}_{\lambda}(\bar{s}, a) - Q^*_{{\lambda_0}}(\bar{s}, a)| \leq \beta_N,
       $$
       其中$\beta_N$是关于$\delta$的函数。
     - 目标达成目标函数估计：
       $$
       \limsup_{n \to \infty} |J^{\pi_{n,\lambda}}_{\lambda} - J^*_{{\lambda_0}}| \leq \frac{N\delta}{2} + \beta_N + \kappa_N.
       $$
   - **意义**：这些估计表明，随着转移核接近确定性转移核，eUDRL生成的策略和值函数的误差趋于0，从而证明了其相对连续性。



### 总结
本节通过分析eUDRL在确定性转移核附近的渐近行为，证明了策略积累点集的相对连续性。这些结果为理解eUDRL在迭代次数趋于无穷时的稳定性提供了理论基础，并为“强化学习即监督学习”算法的理论发展做出了贡献。

## 7.2节内容概述

### 7.2节主题
本节主要研究了在假设最优策略唯一且确定性的情况下，eUDRL（增强型不确定性动态强化学习）生成策略的累积点的连续性。通过分析动态系统的性质，证明了eUDRL生成策略在确定性核附近的连续性，并提供了关于策略、值函数和目标达成目标的误差估计。

### 关键点
1. **假设条件**：假设最优策略在集合$\bar{S}_{\lambda_0}$上是唯一且确定性的，即对于每个$\bar{s} \in \bar{S}_{\lambda_0}$，最优动作集合$O(\bar{s})$中只有一个元素。
2. **动态系统的引入**：通过研究由特定有理函数诱导的动态系统，分析其不动点性质，从而推导出eUDRL策略迭代的累积点行为。
3. **主要结论**：证明了在确定性核附近，eUDRL生成策略的累积点是连续的，并提供了策略、值函数和目标达成目标的误差上界估计。
4. **数值实验**：通过简单的网格世界模型和二臂老虎机模型，展示了理论结果的实际表现。

### 公式内容与意义

#### Lemma 30 (h-lemma)
定义一个有理函数$h_b: [0, 1] \to [0, 1]$：
$$
h_b(x) = \frac{x^{2N}}{x^{2N} + b}
$$
其中$b \in (0, b_0)$，$b_0 = \frac{1}{2N} \left( \frac{2N-1}{2N} \right)^{2N-1}$。

**关键结论**：
1. $h_b$在$(0, 1]$上是递增的，且在$[0, 1]$上是非递减的。
2. $h_b$有三个不动点：$0$，$x_l(b)$，$x_u(b)$，满足$0 < x_l(b) < x_u(b) < 1$。
3. 对于$x \in (x_l(b), 1]$，$h_b(x) > x$；对于$x \in (0, x_l(b))$，$h_b(x) < x$。
4. 如果序列$(y_n)$满足$y_0 > x_l(b)$且$y_{n+1} \geq h_b(y_n)$，则$\liminf_{n \to \infty} y_n \geq x_u(b)$。
5. 存在唯一的连续、严格单调的扩展函数$\bar{x}_l$和$\bar{x}_u$，使得$\bar{x}_l(b) = x_l(b)$和$\bar{x}_u(b) = x_u(b)$。

**意义**：通过分析$h_b$的性质，为后续的策略累积点分析提供了数学工具。

#### Theorem 31
假设$\lambda_0$是确定性核，且在$\bar{S}_{\lambda_0}$上最优策略唯一，则对于任意初始策略$\pi_0 > 0$，存在$\delta \in (0, 2)$，使得对于所有$\lambda \in U_\delta(\lambda_0)$（即$\lambda$与$\lambda_0$的距离小于$\delta$），有：
$$
\liminf_{n \to \infty} \pi_{n, \lambda}(O(\bar{s})|\bar{s}) \geq x_u(b)
$$
其中$b = \frac{\delta^{N} 2^{N + 1}}{4(1 - \delta/2)^{2N} \min_{\bar{s} \in \text{supp} \bar{\mu}} \bar{\mu}(\bar{s})}$，且$x_u(b) \to 1$当$\delta \to 0$。

**意义**：证明了在确定性核附近，eUDRL生成策略的累积点是连续的，并且随着$\delta$的减小，累积点趋近于最优策略。

#### Corollary 33
在Theorem 31的条件下，定义$\delta_0$为方程$b(\delta) = b_0$的唯一解，其中$b_0 = \frac{1}{2N} \left( \frac{2N-1}{2N} \right)^{2N-1}$。则存在两个连续的严格单调函数$x_l(\delta)$和$x_u(\delta)$，满足以下性质：
1. $\liminf_{n \to \infty} \pi_{n, \lambda}(O(\bar{s})|\bar{s}) \geq x_u(\delta) \to 1$，当$\delta \to 0$。
2. $\limsup_{n \to \infty} |V_{\pi_n}^\lambda(\bar{s}) - V^*_{\lambda_0}(\bar{s})| \leq 1 - (1 - \frac{\delta}{2})^N x_u^N(\delta)$。
3. $\limsup_{n \to \infty} |J_{\pi_n}^\lambda - J^*_{\lambda_0}| \leq \frac{N \delta}{2} + (1 - (1 - \frac{\delta}{2})^N x_u^N(\delta))$。
4. 存在一个序列$(\pi^*_{\lambda_0, n})$，使得对于所有$n$，有$\|\pi_{n, \lambda}(\cdot|\bar{s}) - \pi^*_{\lambda_0, n}(\cdot|\bar{s})\|_1 \leq 2(1 - h_b^n(x_0))$，其中$x_0 = \min_{\bar{s} \in \bar{S}_{\lambda_0}} \pi_0(O(\bar{s})|\bar{s})$。

**意义**：提供了策略、值函数和目标达成目标的误差上界估计，并展示了这些误差随着$\delta$的减小而趋近于零。

### 数值实验
1. **二臂老虎机模型**：展示了在不同$\delta$下，eUDRL生成策略的累积点与理论估计的对比。
2. **网格世界模型**：
   - 图10a展示了累积点和理论估计随$\delta$的变化。
   - 图10b展示了策略随迭代次数$n$的收敛行为。
   - 图10c展示了不同$\delta$下策略的累积点。
   - 图10d展示了网格世界的布局。

### 结论
本节通过动态系统分析，证明了在最优策略唯一且确定性的情况下，eUDRL生成策略的累积点在确定性核附近是连续的，并提供了详细的误差估计。这些结果为理解eUDRL算法的收敛性和稳定性提供了理论支持。


## 8节内容概述(Regularized Recursion)

### 8节主题
本节研究了eUDRL（增强型不确定性动态强化学习）中的正则化（Regularization）问题。正则化的主要动机有两个：一是标准的ODT（Optimal Deterministic Target）架构中使用了熵正则化，而之前的模型没有考虑正则化；二是从数学角度看，正则化可以显著简化连续性讨论，例如在保证所有状态$\bar{s} \in \bar{S}_{\lambda_0}$在eUDRL迭代过程中均匀访问时，正则化可以提供状态访问概率的下界。

### 关键点
1. **正则化策略的定义**：引入了一种简单的正则化形式，类似于$\epsilon$-贪婪探索，通过算法当前策略和均匀策略的凸组合来实现。
2. **正则化对连续性的影响**：正则化避免了在证明中维护策略支持项的复杂性，但代价是$\epsilon$-eUDRL在$\epsilon > 0$时不会收敛到最优策略。
3. **主要结论**：证明了在确定性核附近，$\epsilon$-eUDRL生成策略的累积点是相对连续的，并提供了策略、值函数和目标达成目标的误差估计。
4. **数值实验**：通过二臂老虎机模型和网格世界模型展示了正则化对累积点和策略收敛的影响。

### 公式内容与意义

#### $\epsilon$-eUDRL的定义
正则化后的eUDRL（$\epsilon$-eUDRL）的迭代公式为：
$$
\pi_{n+1, \epsilon}(a|s, h, g) = (1 - \epsilon) \frac{\text{num}_{\lambda, \pi_n, \epsilon}(a, s, h, g)}{\text{den}_{\lambda, \pi_n, \epsilon}(s, h, g)} + \epsilon \frac{1}{|A|}
$$
其中，$\epsilon \in (0, 1)$是正则化参数，且当$(s, h, g) \notin \text{supp} \, \text{den}_{\lambda, \pi_n, \epsilon}$时，$\pi_{n+1, \epsilon}(a|s, h, g) = \frac{1}{|A|}$。

**意义**：通过引入正则化，确保了策略在所有状态下都有非零的概率，从而简化了连续性证明。

#### Lemma 40 ($\epsilon$-eUDRL版本的Lemma 6)
描述了$\epsilon$-eUDRL生成量的支持集在兼容MDP族中的行为。

#### Lemma 41 ($\epsilon$-eUDRL版本的Lemma 10)
描述了$\epsilon$-eUDRL在确定性转移核$\lambda_0$下的行为。定义最优策略集合为：
$$
\Pi^*_{\lambda_0, \epsilon} = \left\{ (1 - \epsilon)\pi^*_{\lambda_0} + \epsilon \frac{1}{|A|} \mid \pi^*_{\lambda_0} \text{是} \lambda_0 \text{在} \bar{S}_{\lambda_0} \text{上的最优策略} \right\}
$$
该引理断言$\epsilon$-eUDRL的累积点包含在$\Pi^*_{\lambda_0, \epsilon}$中。

**意义**：提供了$\epsilon$-eUDRL在确定性核附近的累积点的性质。

#### Lemma 42 ($\epsilon$-eUDRL版本的Lemma 22)
提供了状态访问概率的下界：
$$
\alpha(\delta, \epsilon) = \frac{2}{N(N-1)} \left( \min_{\bar{s} \in \bar{S}_{\lambda_0}} \bar{\mu}(\bar{s}) \right) \left( \frac{\epsilon}{|A|} \right)^N \left( 1 - \frac{\delta}{2} \right)^N
$$

**意义**：正则化确保了状态访问概率的下界，从而支持了连续性分析。

#### Theorem 34 ($\epsilon$-eUDRL极限策略的相对连续性)
假设$\{M_\lambda\}$和$\{\bar{M}_\lambda\}$是兼容族，$\lambda_0$是确定性核，$(\pi_{n, \lambda, \epsilon})_{n \geq 0}$是由$\epsilon$-eUDRL生成的策略序列，初始条件为$\pi_0 > 0$，转移核为$\lambda$，正则化参数为$\epsilon \in (0, 1)$。则对于所有$\pi_0 > 0$，以下结论成立：
1. **累积点的相对连续性**：设$L(\pi_0, \lambda, \epsilon)$表示$(\pi_{n, \lambda, \epsilon})_{n \geq 0}$的累积点集合，则任意函数$u: (\pi_0, \lambda, \epsilon) \mapsto u(\pi_0, \lambda, \epsilon) \in L(\pi_0, \lambda, \epsilon)$在$\lambda_0$和$\epsilon_0$处相对于$\bar{S}_{\lambda_0}$是相对连续的，即：
   $$
   [u(\pi_0, \lambda, \epsilon_0)](·|\bar{s}) \xrightarrow{O(\bar{s})} \pi^*_{\lambda_0, \epsilon_0}(·|\bar{s}) \quad \text{当} \quad \lambda \to \lambda_0
   $$
2. **累积点的误差估计**：设$\alpha(\delta, \epsilon)$如Lemma 42所述，$\beta, \tilde{\beta}, \epsilon \in (0, 1)$满足$1 > \gamma + \epsilon$，其中$\gamma = \tilde{\beta} \frac{((1-\epsilon)N - \beta)}{\alpha(\delta, \epsilon)}$，且
   $$
   U_\delta(\lambda_0) = \left\{ \lambda \mid \max_{(s,a) \in S \times A} \| \lambda(·|s, a) - \lambda_0(·|s, a) \|_1 < \delta \right\}
   $$
   则存在$\delta > 0$，使得对于所有$\lambda \in U_\delta(\lambda_0)$和所有$\bar{s} \in \bar{S}_{\lambda_0}$，有：
   $$
   \liminf_{n} \pi_{n, \lambda, \epsilon}(O(\bar{s})|\bar{s}) \geq x^*(\gamma, \epsilon, |O(\bar{s})|)
   $$
   其中$x^*(\gamma, \epsilon, |O(\bar{s})|)$在$(\beta, \tilde{\beta}, \alpha, \epsilon) \to (0, 0, \alpha(0, \epsilon_0), \epsilon_0)$时趋近于$1 - \epsilon_0 \left(1 - \frac{|O(\bar{s})|}{|A|}\right)$。

**意义**：证明了在确定性核附近，$\epsilon$-eUDRL生成策略的累积点是相对连续的，并提供了累积点的误差估计。

#### Corollary 36 ($\epsilon$-eUDRL累积点的估计)
在Theorem 34的条件下，假设$\delta \in (0, 1)$，定义：
$$
\alpha = \frac{2}{N(N-1)} \left( \min_{\bar{s} \in \bar{S}_{\lambda_0}} \bar{\mu}(\bar{s}) \right) \left( \frac{\epsilon}{|A|} \right)^N \left( 1 - \frac{\delta}{2} \right)^N
$$
$$
\tilde{\beta} = \frac{N \delta}{2}, \quad x^*(\gamma, \epsilon, M) = \frac{\hat{x}^* + \sqrt{(\hat{x}^*)^2 + 4 \gamma \epsilon M / |A|}}{2}
$$


定义量$\tilde{\beta}=\frac{N\delta}{2}$，$x^{*}(\gamma,\epsilon,M)=\frac{\hat{x}^{*}+\sqrt{(\hat{x}^{*})^{2}+\frac{4\gamma\epsilon M}{|\mathcal{A}|}}}{2}$，其中$\hat{x}^{*}=1 - \epsilon(1-\frac{M}{\mathcal{A}})-\gamma$（参见引理45）。对于时间范围$h$，$1\leq h\leq N$ ，有：
$$
\beta_{h}=
\begin{cases}
\max\{\delta,\tilde{\beta}\}, &\text{当 } h = 1时,\\
\delta+\kappa_{h - 1}+\beta_{h - 1}, &\text{当 } h\geq2时,
\end{cases}
$$
$$
\gamma_{h}=\frac{\tilde{\beta}}{((1 - \epsilon)^{N}-\beta_{h})\alpha},
$$
$$
\kappa_{h}=\underset{\bar{s}=(s,h',g)\in\tilde{\mathcal{S}}_{\lambda_{0}},h' = h}{\max}2\left(1-\epsilon\left(1-\frac{|\mathcal{O}(\bar{s})|}{|\mathcal{A}|}\right)-x^{*}(\gamma_{h},\epsilon,\mathcal{O}(\bar{s}))\right).
$$ 




假设$\beta_h, \gamma_h \in (0, 1)$且$1 > \gamma_h + \epsilon$，注意到$\beta_h, \kappa_h, \gamma_h$随$h$增加而增加，并且在$\delta \to 0$时趋近于0。则对于所有$\lambda \in U_\delta(\lambda_0)$和所有$\pi_0 > 0$，以下结论成立：
1. $$
   \limsup_{n} \max_{\bar{s} \in \bar{S}_{\lambda_0}} \left| 1 - \epsilon \left(1 - \frac{|O(\bar{s})|}{|A|}\right) - \pi_{n, \epsilon}(O(\bar{s})|\bar{s}) \right| \leq \kappa_N
   $$
2. $$
   (\exists (\pi^*_{n, \epsilon}), \pi^*_{n, \epsilon} \in \Pi^*_{\lambda_0, \epsilon}) : \limsup_{n} \max_{\bar{s}, a \in \bar{S}_{\lambda_0} \times A} |Q_{\pi_{n, \epsilon}}^\lambda(\bar{s}, a) - Q_{\pi^*_{n, \epsilon}}^{\lambda_0}(\bar{s}, a)| \leq \beta_N
   $$
3. $$
   (\exists (\pi^*_{n, \epsilon}), \pi^*_{n, \epsilon} \in \Pi^*_{\lambda_0, \epsilon}) : \limsup_{n} \max_{\bar{s} \in \bar{S}_{\lambda_0}} |V_{\pi_{n, \epsilon}}^\lambda(\bar{s}) - V_{\pi^*_{n, \epsilon}}^{\lambda_0}(\bar{s})| \leq \beta_N + \kappa_N
   $$
4. $$
   (\exists (\pi^*_{n, \epsilon}), \pi^*_{n, \epsilon} \in \Pi^*_{\lambda_0, \epsilon}) : \limsup_{n} |J_{\pi_{n, \epsilon}}^\lambda - J_{\pi^*_{n, \epsilon}}^{\lambda_0}| \leq \frac{N \delta}{2} + \beta_N + \kappa_N
   $$
5. 对于所有$\epsilon' > 0$，存在$n_0$和$(\pi^*_{n, \epsilon})_{n \geq 0}$，$\pi^*_{n, \epsilon} \in \Pi^*_{\lambda_0, \epsilon}$，使得对于所有$n \geq n_0$和所有$\bar{s} \in \bar{S}_{\lambda_0}$，有：
   $$
   \|\pi_{n, \epsilon}(·|\bar{s}) - \pi^*_{n, \epsilon}(·|\bar{s})\|_1 \leq 2 \left(1 - \epsilon \left(1 - \frac{|O(\bar{s})|}{|A|}\right) - z^{(n-n_0)}_{\gamma', \epsilon, |O(\bar{s})|}(x_0)\right)
   $$
   其中$x_0 = \epsilon \frac{|O(\bar{s})|}{|A|}$，$\beta' > \beta_N + \epsilon'$，且$\epsilon'$满足$1 > \gamma' + \epsilon'$，$\gamma' = \tilde{\beta} \frac{((1-\epsilon)N - \beta')}{\alpha} > 0$。

**意义**：提供了$\epsilon$-eUDRL生成策略、值函数和目标达成目标的误差估计，并展示了这些误差随$\delta \to 0$而趋近于零。

### 数值实验
1. **二臂老虎机模型**：
   - 图11a展示了$\epsilon$-eUDRL的累积点和估计值随距离$\delta$的变化。
   - 图11b对比了$\epsilon$-eUDRL和eUDRL的累积点估计。
2. **网格世界模型**：
   - 图12a展示了正则化ODT递归的累积点和估计值。
   - 图12b展示了网格世界的布局。

### 结论
本节通过引入正则化，简化了连续性分析，并证明了在确定性核附近，$\epsilon$-eUDRL生成策略的累积点是相对连续的。正则化虽然简化了分析，但也导致策略偏离最优策略。因此，正则化参数$\epsilon$的选择需要在简化分析和策略性能之间进行权衡。

## 附录A：片段分布及其分解

### 内容概述
附录A主要讨论了片段（Segment）分布的定义和性质。片段是从轨迹（Trajectory）中提取的连续部分，用于描述在强化学习中，特别是在eUDRL（增强型不确定性动态强化学习）框架下的状态和动作序列。本节详细定义了片段分布的构造方式，并分析了其归一化常数和相关概率计算。

### 关键点
1. **片段的定义**：片段是从轨迹中提取的连续部分，包含初始状态、目标、剩余时间步长以及一系列状态和动作。
2. **片段分布的构造**：通过轨迹分布的边缘分布构造片段分布，并引入归一化常数$c$以确保片段分布的正确性。
3. **片段分布的概率计算**：通过边缘分布和条件概率计算片段分布的概率。
4. **归一化常数$c$的界限**：推导了归一化常数$c$的上界，以确保片段分布的定义是合理的。

### 公式内容与意义

#### 片段的定义
片段$\sigma$由以下元组定义：
$$
\sigma = (l(\sigma), s^\sigma_0, h^\sigma_0, g^\sigma_0, a^\sigma_0, s^\sigma_1, a^\sigma_1, \dots, s^\sigma_{l(\sigma)})
$$
其中：
- $l(\sigma)$是片段的长度（即转移次数）。
- $s^\sigma_0, h^\sigma_0, g^\sigma_0$分别是片段开始时的状态、剩余时间步长和目标。
- $a^\sigma_i$和$s^\sigma_i$分别是片段中的动作和状态。

#### 片段分布的构造
片段分布$d^\pi_\Sigma$通过轨迹分布的边缘分布构造：
$$
P(\Sigma = \sigma; \pi) = c^{-1} \sum_{t \leq N - l(\sigma)} P(S_t = s^\sigma_0, H_t = h^\sigma_0, G_t = g^\sigma_0, A_t = a^\sigma_0, \dots, S_{t+l(\sigma)} = s^\sigma_{l(\sigma)}; \pi)
$$
其中$c$是归一化常数：
$$
c = \sum_{\sigma \in \text{Seg}} \sum_{t \leq N - l(\sigma)} P(S_t = s^\sigma_0, H_t = h^\sigma_0, G_t = g^\sigma_0, A_t = a^\sigma_0, \dots, S_{t+l(\sigma)} = s^\sigma_{l(\sigma)}; \pi) > 0
$$

#### 片段分布的概率计算
片段分布的概率可以分解为：
$$
P(\Sigma = \sigma; \pi) = c^{-1} \left( \prod_{i=1}^{l(\sigma)} \lambda(s^\sigma_i | a^\sigma_{i-1}, s^\sigma_{i-1}) \right) \left( \prod_{i=0}^{l(\sigma)-1} \pi(a^\sigma_i | \bar{s}^\sigma_i) \right) P(S_t = s^\sigma_0, H_t = h^\sigma_0, G_t = g^\sigma_0; \pi)
$$

#### 归一化常数$c$的界限
归一化常数$c$的上界为：
$$
0 < c \leq \sum_{k=1}^N \sum_{t \leq N - k} 1 = \frac{N(N + 1)}{2}
$$
**意义**：通过限制$c$的上界，确保片段分布是定义良好的，并且归一化常数是有限的。

## 附录B：动机示例

### 内容概述
附录B通过一系列示例展示了eUDRL生成的策略和目标达成目标（Goal-reaching Objective）在某些情况下可能出现的不连续性。这些示例旨在说明eUDRL在特定点（如边界点或确定性核）处的不连续性，并探讨这些不连续性是否可以被移除。

### 关键点
1. **示例1**：展示了在边界点处目标达成目标的不可移除的不连续性。
2. **示例2**：展示了在确定性点处策略的不可移除的不连续性。
3. **示例3-7**：通过随机游走、二臂老虎机和网格世界等示例，进一步探讨eUDRL生成量的连续性和不连续性。

### 公式内容与意义

#### 示例1：目标达成目标的不连续性
考虑一个具有三个状态和三个动作的MDP，其扩展环境（CE）的初始分布固定初始剩余时间为1，目标分布为$P(G_0 = 0) = P(G_0 = 2) = \frac{1}{2}$。定义参数化的转移核$\lambda_\alpha$：
$$
\lambda_\alpha(g|a) = \begin{cases}
1 - \alpha & \text{if } g = 0, a = 0 \\
\alpha/4 & \text{if } g = 1, a = 0 \\
3\alpha/4 & \text{if } g = 2, a = 0 \\
3\alpha/4 & \text{if } g = 0, a = 1 \\
1 - \alpha & \text{if } g = 1, a = 1 \\
\alpha/4 & \text{if } g = 2, a = 1 \\
1/2 & \text{if } g = 0, a = 2 \\
1/2 & \text{if } g = 1, a = 2 \\
0 & \text{if } g = 2, a = 2
\end{cases}
$$
当$\alpha \to 0^+$时，$\lambda_\alpha$趋近于一个确定性核$\lambda_0$。

通过计算eUDRL策略的递归公式，发现目标达成目标$J_{\lambda, \pi_2, 0^+} = \frac{9}{19}$，而当$\alpha = 0$时，$J_{\lambda, \pi_2, 0} = \frac{7}{16}$。这表明在$\alpha = 0$处存在不连续性。

#### 示例2：策略的不连续性
考虑与示例1相同的MDP，但初始目标分布为均匀分布。定义新的参数化转移核：
$$
\lambda_\alpha(g|a) = \begin{cases}
1 - \alpha & \text{if } g = 0, a = 0 \\
\alpha & \text{if } g = 1, a = 0 \\
0 & \text{if } g = 2, a = 0 \\
0 & \text{if } g = 0, a = 1 \\
1 - \alpha & \text{if } g = 1, a = 1 \\
\alpha & \text{if } g = 2, a = 1 \\
0 & \text{if } g = 0, a = 2 \\
1 & \text{if } g = 1, a = 2 \\
0 & \text{if } g = 2, a = 2
\end{cases}
$$
计算结果表明，策略在$\alpha = 0$处存在不连续性，但目标达成目标在某些情况下是连续的。

#### 示例3-7：其他示例
这些示例进一步探讨了eUDRL生成量的连续性和不连续性，包括随机游走、二臂老虎机和网格世界等场景。这些示例展示了eUDRL在不同环境下的行为，并验证了理论分析的结果。


## 总结
附录A和B通过定义片段分布和构造示例，详细分析了eUDRL生成量的连续性和不连续性。这些分析为理解eUDRL在不同环境下的行为提供了理论支持，并为后续章节的理论证明提供了动机。


## **附录C：有限迭代次数下的内核连续性**

### **内容概述**
本附录旨在证明在所有转移核（transition kernels）集合的内点处，由eUDRL（一种强化学习方法）生成的策略及相关量的连续性。内点被定义为满足$λ > 0$的转移核$λ$。文中通过两个主要引理（Lemma 37和Lemma 38）来探讨策略的稳定性，并进一步讨论了在边界点（包括确定性核和非确定性核）处连续性失效的原因。

### **关键点与公式**

#### **1. Lemma 37（支撑集稳定性）**
**内容**：对于兼容的MDP族$\{M_λ : λ ∈ (∆S)^{S×A}\}$和$\{\overline{M}_λ : λ ∈ (∆S)^{S×A}\}$，假设$λ, λ' ∈ (∆S)^{S×A}$是满足$λ, λ' > 0$的转移核，且$π, π' ∈ (∆A)^S$是策略，则有
$$
\text{supp den}_{λ,π} = \text{supp den}_{λ',π'}
$$
**证明思路**：
- 假设$\bar{s} = (s, h, g) ∈ \text{supp den}_{λ,π}$，则存在轨迹$τ$和时间步$t$，使得$P_{λ,π}(T = τ) > 0$，且满足$l(τ) ≥ t + h$，$s_τ^t = s$，$ρ(s_τ^{t+h}) = g$。
- 通过修改轨迹$τ$中的动作，构造新的轨迹$τ'$，使得$τ'$中的动作属于$π'$的支撑集。
- 由于$λ' > 0$，因此$P_{λ',π'}(T = τ') > 0$，从而$\bar{s} ∈ \text{supp den}_{λ',π'}$。
**意义**：该引理表明，在所有转移核的内点处，支撑集$\text{supp den}_{λ,π}$是恒定的，这与在确定性核（或边界点）附近的行为形成鲜明对比。



#### **2. Lemma 38（eUDRL策略和值的连续性）**
**内容**：对于兼容的MDP族$\{M_λ : λ ∈ (∆S)^{S×A}\}$和$\{\overline{M}_λ : λ ∈ (∆S)^{S×A}\}$，假设$(π_n,λ), π_0 ∈ (∆A)^S$是由eUDRL生成的策略序列，且$λ_0 > 0$是内点$λ_0 ∈ ((∆S)^{S×A})^\circ$。则对于所有$n ≥ 0$：
1. 对于所有$\bar{s} = (s, h, g) ∈ \overline{S}_T$，策略$π_{n+1,λ}(·|\bar{s})$在$λ_0$处关于$λ$是连续的。
2. 对于所有$\bar{s} = (s, h, g) ∈ \overline{S}_T$，值函数$V_{π_n}^λ(\bar{s})$和$Q_{π_n}^λ(\bar{s}, ·)$在$λ_0$处关于$λ$是连续的。此外，目标达成目标$J_{π_n}^λ$在$λ_0$处关于$λ$也是连续的。
**证明思路**：
- **归纳法**：
  - **基础情况（$n = 0$）**：$π_0$是关于$λ$的常数，因此在$λ_0$处连续。
  - **归纳假设**：假设对于$n ≥ 0$，结论成立，证明对于$n+1$也成立。
    - 若$\bar{s} \notin \text{supp den}_{λ_0,π_n}$，则根据引理37，$\bar{s} \notin \text{supp den}_{λ,π_n}$对于所有$λ ∈ U_δ(λ_0)$。因此，$π_{n+1,λ}(·|\bar{s}) = \frac{1}{|A|}$，从而在$λ_0$处连续。
    - 若$\bar{s} ∈ \text{supp den}_{λ_0,π_n}$，则对于所有$λ ∈ U_δ(λ_0)$，有$\bar{s} ∈ \text{supp den}_{λ,π_n}$。此时，$π_{n+1,λ}(a|\bar{s}) = \frac{\text{num}_{λ,π_n}(\bar{s}, a)}{\text{den}_{λ,π_n}(\bar{s})}$。根据引理12，$\text{num}_{λ,π_n}(\bar{s}, ·)$和$\text{den}_{λ,π_n}(\bar{s})$在$(λ, π_n)$上是连续的，且根据归纳假设，$π_{n,λ}(·|\bar{s}')$在$λ_0$处关于$λ$是连续的。因此，$π_{n+1,λ}(a|\bar{s})$在$λ_0$处关于$λ$是连续的。
- **值函数的连续性**：对于$n > 0$，固定$\bar{s} ∈ \overline{S}_T$，值函数$V_{π_n}^λ(\bar{s})$和$Q_{π_n}^λ(\bar{s}, a)$在$(λ, π_n)$上是连续的（根据引理12）。由于$π_{n,λ}$在$λ_0$处关于$λ$是连续的，因此$V_{π_n}^λ(\bar{s})$和$Q_{π_n}^λ(\bar{s}, a)$在$λ_0$处关于$λ$是连续的。目标达成目标$J_{π_n}^λ$在$λ_0$处关于$λ$的连续性则由值函数$V_{π_n}^λ(\bar{s})$的连续性推导而来。
**意义**：该引理证明了在所有转移核的内点处，eUDRL生成的策略和值函数是连续的，这为eUDRL在这些点的稳定性提供了理论支持。



#### **3. 边界点的不连续性**
**内容**：讨论了在边界点（包括确定性核和非确定性核）处，连续性证明失效的原因。
**关键点**：
- **边界点的定义**：边界点是指转移核$λ_0$不满足$λ_0 > 0$的点。
- **不连续性的原因**：
  - 在边界点处，支撑集$\text{supp den}_{λ,π}$可能在邻域内发生突变，导致策略$π_{n,λ}$在$λ_0$处不连续。
  - 具体来说：
    - 在迭代$n = 1$时，策略$π_{1,λ}$可能在$\overline{S}_T \setminus \text{supp den}_{λ_0,π_0}$中出现不连续性。
    - 在迭代$n = 2$时，除了上述不连续性外，还可能在$\overline{S}_{λ_0}$中出现不连续性，这会导致值函数和目标达成目标的不连续性。
**意义**：该部分揭示了eUDRL在边界点处可能出现的不连续性，这对于理解eUDRL在不同转移核下的行为具有重要意义。



#### **4. Lemma 39（边界点的稳定性）**

**内容**：将引理6推广到一般转移核$λ_0$，允许$λ_0$为非确定性核。

**结论**：引理6的结论在以下修改下仍然成立：
- 将“确定性转移核$λ_0$”替换为“转移核$λ_0$”。
- 将点1、2、3中的“对于所有$n ≥ 0$和所有$λ ∈ U_2(λ_0)$”替换为“存在$2 > δ > 0$，使得对于所有$n ≥ 0$和所有$λ ∈ U_δ(λ_0)$”。
- 
**证明思路**：证明过程与引理6相同，关键在于找到合适的$δ > 0$，使得对于所有$λ ∈ U_δ(λ_0)$，有$\text{supp } λ_0 ⊂ \text{supp } λ$。这是由于恒等映射$λ \mapsto λ$的连续性。

**意义**：该引理为讨论边界点处的稳定性提供了理论基础。

### **总结**
附录C通过引理37和引理38
证明了eUDRL生成的策略和值函数在所有转移核的内点处是连续的。然而，在边界点（包括确定性核和非确定性核）处，连续性可能失效，这主要是由于支撑集$\text{supp den}_{λ,π}$在邻域内的突变导致的。这些结论对于理解eUDRL在不同转移核下的行为具有重要意义。

## 附录D内容概述

附录D主要讨论了正则化递归（ϵ-eUDRL）算法的收敛性和稳定性，通过引入和证明一系列引理和定理，展示了ϵ-eUDRL算法在确定性转移核下的性质，以及其在不同条件下的收敛行为和稳定性。附录分为四个部分：初步引理、主要定理、对其他片段子空间的扩展以及对累积点位置的估计。

### D.1 初步引理

#### Lemma 40 (ϵ-eUDRL版本的支持稳定性)
**内容**：
- 给定两个兼容的家族$\{M_\lambda : \lambda \in (\Delta S)^{S \times A}\}$和$\{\bar{M}_\lambda : \lambda \in (\Delta S)^{S \times A}\}$，以及由ϵ-eUDRL迭代生成的策略序列$(\pi_{n,\lambda,\epsilon})_{n \geq 0}$，在确定性转移核$\lambda_0$下，满足以下条件：
  1. 对所有$n \geq 0$和所有$\lambda \in U_\delta(\lambda_0)$，支持集的包含关系成立：
     $$
     \text{supp}(\text{num}_{\lambda_0, \pi_0}) \cap (A \times \text{supp}(\nu_{\lambda_0, \pi_0})) \subset \text{supp}(\text{num}_{\lambda, \pi_{n,\epsilon}}) \cap (A \times \text{supp}(\nu_{\lambda, \pi_{n,\epsilon}}))
     $$
     当$\lambda = \lambda_0$时，包含关系变为等式。
  2. 对所有$n \geq 0$和所有$\lambda \in U_\delta(\lambda_0)$，有$\bar{S}_{\lambda_0} \subset \text{supp}(\text{den}_{\lambda, \pi_{n,\epsilon}}) \cap \text{supp}(\nu_{\lambda, \pi_{n,\epsilon}})$，当$\lambda = \lambda_0$时，包含关系变为等式。
  3. 对所有$n \geq 0$和所有$\lambda \in U_\delta(\lambda_0)$，有$P_\lambda(S^\Sigma_0 = s, l(\Sigma) = h, \rho(S^\Sigma_h) = g, H^\Sigma_0 = h, G^\Sigma_0 = g; \pi_{n,\epsilon}) > 0$，对所有$(s, h, g) \in \bar{S}_{\lambda_0}$。

**意义**：
- 该引理证明了ϵ-eUDRL算法在确定性转移核附近的支持集稳定性，即策略的支持集不会因小的扰动而发生显著变化，为后续的收敛性分析奠定了基础。


#### Lemma 41 (ϵ-eUDRL策略的最优性)
**内容**：
- 对于确定性转移核$\lambda_0$，由ϵ-eUDRL迭代生成的策略序列$(\pi_{n,\epsilon})_{n \geq 0}$满足：
  1. 对所有$n \geq 0$，策略$\pi_{n+1,\epsilon}$在$\bar{S}_{\lambda_0}$上具有形式：
     $$
     \pi_{n+1,\epsilon} = (1 - \epsilon)\pi^*_n + \frac{\epsilon}{|A|}
     $$
     其中$\pi^*_n$是基于“num/den”比率定义的最优策略。
  2. 对所有$n \geq 1$和所有$\bar{s} = (s, h, g) \in \bar{S}_{\lambda_0}$，值函数满足：
     $$
     V^{\pi_{n,\epsilon}}(\bar{s}) \geq (1 - \epsilon)^h \geq (1 - \epsilon)^N
     $$
  3. 对所有$n \geq 1$和所有$\bar{s} = (s, h, g) \in \bar{S}_{\lambda_0}$，动作值函数满足：
     $$
     Q^{\pi_{n,\epsilon}}(\bar{s}, a) \geq (1 - \epsilon)^{h-1} \geq (1 - \epsilon)^N \quad \text{对于} \ a \in O(\bar{s})
     $$
     否则为0。

**意义**：
- 该引理证明了ϵ-eUDRL策略在确定性转移核下的最优性，并给出了值函数和动作值函数的下界估计，为分析算法的收敛性和稳定性提供了关键依据。


#### Lemma 42 (ϵ-eUDRL的访问概率下界)
**内容**：
- 对于确定性转移核$\lambda_0$，由ϵ-eUDRL迭代生成的策略序列$(\pi_{n,\lambda,\epsilon})_{n \geq 0}$，对所有$n > 0$、所有$\bar{s} = (s, h, g) \in \bar{S}_{\lambda_0}$和所有$\lambda \in U_2(\lambda_0)$，访问概率满足：
  $$
  P_\lambda(H^\Sigma_0 = h, G^\Sigma_0 = g | S^\Sigma_0 = s, l(\Sigma) = h; \pi_{n,\epsilon}) \geq \alpha(\delta, \epsilon)
  $$
  其中
  $$
  \alpha(\delta, \epsilon) = \frac{2}{N(N + 1)} \left( \min_{\bar{s}' \in \text{supp} \bar{\mu}} \bar{\mu}(\bar{s}') \right) \left( \frac{\epsilon}{|A|} \right)^N \left( 1 - \frac{\delta}{2} \right)^N > 0
  $$

**意义**：
- 该引理提供了ϵ-eUDRL策略在确定性转移核附近的访问概率的下界估计，表明策略在状态空间中的探索能力，为后续的收敛性分析提供了支持。


#### Lemma 44 (ϵ-eUDRL的连续性)
**内容**：
- 对于确定性转移核$\lambda_0$，如果策略$\pi_\epsilon$与$\Pi^*_{\lambda_0,\epsilon}$足够接近，且$\lambda$与$\lambda_0$足够接近，则动作值函数$Q^{\pi_\epsilon}_\lambda$与$\{Q^{\pi^*_\epsilon}_{\lambda_0} | \pi^*_\epsilon \in \Pi^*_{\lambda_0,\epsilon}\}$足够接近。具体而言，对所有$\epsilon' > 0$，存在$\delta > 0$，使得当$\lambda \in U_\delta(\lambda_0)$且
  $$
  2(1 - \epsilon(1 - \frac{|O(\bar{s})|}{|A|}) - \pi_\epsilon(O(\bar{s})|\bar{s})) < \delta
  $$
  时，有
  $$
  |Q^{\pi_\epsilon}_\lambda(\bar{s}, \cdot) - Q^{\pi^*_\epsilon}_{\lambda_0}(\bar{s}, \cdot)| < \epsilon'
  $$
  其中$\pi^*_\epsilon \in \Pi^*_{\lambda_0,\epsilon}$。

**意义**：
- 该引理证明了ϵ-eUDRL策略的动作值函数在商拓扑下的连续性，表明策略在确定性转移核附近的稳定性，为分析算法的收敛性提供了关键支持。


### D.2 主要定理

#### Theorem 34 (ϵ-eUDRL极限策略的相对连续性)
**内容**：
- 对于确定性转移核$\lambda_0$，由ϵ-eUDRL迭代生成的策略序列$(\pi_{n,\lambda,\epsilon})_{n \geq 0}$，满足以下性质：
  1. 对所有初始条件$\pi_0 > 0$，累积点集合$L(\pi_0, \lambda, \epsilon)$中的任意函数$u(\pi_0, \lambda, \epsilon)$在$\lambda$和$\epsilon$上是相对连续的，即
     $$
     [u(\pi_0, \lambda, \epsilon_0)](\cdot|\bar{s}) \xrightarrow{\text{O}(\bar{s})} \pi^*_{\lambda_0,\epsilon_0}(\cdot|\bar{s}) \quad \text{当} \ \lambda \to \lambda_0
     $$
  2. 对所有$\bar{s} \in \bar{S}_{\lambda_0}$，有
     $$
     \liminf_{n} \pi_{n,\lambda,\epsilon}(O(\bar{s})|\bar{s}) \geq x^*(\gamma, \epsilon, |O(\bar
{s})|)
     $$
     其中$x^*(\gamma, \epsilon, |O(\bar{s})|)$是引理45中的不动点，并且
     $$
     x^*(\gamma, \epsilon, |O(\bar{s})|) \to 1 - \epsilon_0 \left(1 - \frac{|O(\bar{s})|}{|A|}\right) \quad \text{当} \ (\beta, \tilde{\beta}, \alpha, \epsilon) \to (0, 0, \alpha(0, \epsilon_0), \epsilon_0)
     $$

**意义**：
- 该定理证明了ϵ-eUDRL算法生成的策略序列在确定性转移核下的相对连续性，表明算法在小的扰动下能够收敛到接近最优策略的策略，为算法的稳定性和收敛性提供了理论支持。


### D.3 对其他片段子空间的扩展

#### Theorem 35 (ϵ-eUDRL在Segdiag/trail上的相对连续性)
**内容**：
- 定理34在Segdiag/trail子空间上仍然成立，即对于限制在Segdiag/trail上的ϵ-eUDRL算法，其生成的策略序列在确定性转移核下的相对连续性仍然成立。

**意义**：
- 该定理将ϵ-eUDRL的收敛性和稳定性结果扩展到更一般的片段子空间，表明算法在不同的约束条件下仍然具有良好的收敛性和稳定性。

### D.4 累积点位置的估计

#### Corollary 48 (累积点位置的估计)
**内容**：
- 在定理34的条件下，假设$\delta \in (0, 1)$，定义以下参数：
  - $\alpha = \frac{2}{N(N - 1)} \left( \min_{\bar{s} \in \bar{S}_{\lambda_0}} \bar{\mu}(\bar{s}) \right) \left( \frac{\epsilon}{|A|} \right)^N \left( 1 - \frac{\delta}{2} \right)^N$
  - $\tilde{\beta} = \frac{N\delta}{2}$
  - $x^*(\gamma, \epsilon, M) = \frac{\hat{x}^* + \sqrt{(\hat{x}^*)^2 + 4\gamma\epsilon M}}{2}$，其中$\hat{x}^* = 1 - \epsilon(1 - \frac{M}{|A|}) - \gamma$
  - 对于地平线$h$，$1 \leq h \leq N$，定义
    $$
    \beta_h =
    \begin{cases}
    \max\{\delta, \tilde{\beta}\}, & \text{如果 } h = 1 \\
    \delta + \kappa_{h-1} + \beta_{h-1}, & \text{如果 } h \geq 2
    \end{cases}
    $$
    $$
    \gamma_h = \tilde{\beta} \left( \frac{(1 - \epsilon)^N - \beta_h}{\alpha} \right)
    $$
    $$
    \kappa_h = \max_{\bar{s} = (s, h', g) \in \bar{S}_{\lambda_0}, h' = h} 2 \left( 1 - \epsilon \left(1 - \frac{|O(\bar{s})|}{|A|}\right) - x^*(\gamma_h, \epsilon, O(\bar{s})) \right)
    $$

  在这些条件下，对所有$\lambda \in U_\delta(\lambda_0)$和所有$\pi_0 > 0$，有以下结论：
  1. $$
     \limsup_{n} \max_{\bar{s} \in \bar{S}_{\lambda_0}} 2 \left( 1 - \epsilon \left(1 - \frac{|O(\bar{s})|}{|A|}\right) - \pi_{n,\epsilon}(O(\bar{s})|\bar{s}) \right) \leq \kappa_N
     $$
  2. 存在$(\pi^*_n, \epsilon)$，$\pi^*_n, \epsilon \in \Pi^*_{\lambda_0, \epsilon}$，使得
     $$
     \limsup_{n} \max_{\bar{s}, a \in \bar{S}_{\lambda_0} \times A} |Q^{\pi_{n,\epsilon}}_\lambda(\bar{s}, a) - Q^{\pi^*_n, \epsilon}_{\lambda_0}(\bar{s}, a)| \leq \beta_N
     $$
  3. 存在$(\pi^*_n, \epsilon)$，$\pi^*_n, \epsilon \in \Pi^*_{\lambda_0, \epsilon}$，使得
     $$
     \limsup_{n} \max_{\bar{s} \in \bar{S}_{\lambda_0}} |V^{\pi_{n,\epsilon}}_\lambda(\bar{s}) - V^{\pi^*_n, \epsilon}_{\lambda_0}(\bar{s})| \leq \beta_N + \kappa_N
     $$
  4. 存在$(\pi^*_n, \epsilon)$，$\pi^*_n, \epsilon \in \Pi^*_{\lambda_0, \epsilon}$，使得
     $$
     \limsup_{n} |J^{\pi_{n,\epsilon}}_\lambda - J^{\pi^*_n, \epsilon}_{\lambda_0}| \leq \frac{N\delta}{2} + \beta_N + \kappa_N
     $$
  5. 对所有$\epsilon' > 0$，存在$n_0$和$(\pi^*_n, \epsilon)_{n \geq 0}$，$\pi^*_n, \epsilon \in \Pi^*_{\lambda_0, \epsilon}$，使得对所有$n \geq n_0$和所有$\bar{s} \in \bar{S}_{\lambda_0}$，有
     $$
     \|\pi_{n,\epsilon}(\cdot|\bar{s}) - \pi^*_n, \epsilon(\cdot|\bar{s})\|_1 \leq 2 \left( 1 - \epsilon \left(1 - \frac{|O(\bar{s})|}{|A|}\right) - z^{(n - n_0)}_{\gamma', \epsilon, |O(\bar{s})|}(x_0) \right)
     $$
     其中$x_0 = \frac{\epsilon|O(\bar{s})|}{|A|}$，$\beta' > \beta_N + \epsilon'$，且$\epsilon'$满足$1 > \gamma' + \epsilon'$，$\gamma' = \tilde{\beta} \left( \frac{(1 - \epsilon)^N - \beta'}{\alpha} \right) > 0$。

**意义**：
- 该推论提供了ϵ-eUDRL算法生成的策略序列的累积点位置的估计，表明在确定性转移核附近，策略序列的值函数、动作值函数和策略本身都能收敛到接近最优策略的值，进一步验证了算法的收敛性和稳定性。

### 总结
附录D通过一系列引理和定理，详细分析了ϵ-eUDRL算法在确定性转移核下的收敛性和稳定性。这些结果不仅证明了算法在小扰动下的鲁棒性，还为算法在不同约束条件下的应用提供了理论支持。




# 原文翻译（部分）

**关于上下颠倒强化学习、目标条件监督学习和在线决策变换器的收敛性和稳定性**

Miroslav Štrupl<sup>1</sup> miroslav.strupl@idsia.ch  
Oleg Szehr<sup>1</sup> oleg.szehr@idsia.ch  
Francesco Faccio<sup>1,2</sup> francesco.faccio@idsia.ch  
Dylan R. Ashley<sup>1,2</sup> dylan.ashley@idsia.ch  
Rupesh Kumar Srivastava rupspace@gmail.com  
Jürgen Schmidhuber<sup>1,2,3</sup> juergen.schmidhuber@kaust.edu.sa  

<sup>1</sup> Dalle Molle人工智能研究所（IDSIA）- USI/SUPSI，卢加诺，瑞士  
<sup>2</sup> 沙特阿拉伯阿卜杜拉国王科技大学卓越生成人工智能中心，图瓦尔  
<sup>3</sup> NNAISENSE，卢加诺，瑞士  

**摘要**  
本文对Episodic Upside-Down Reinforcement Learning（eUDRL）、Goal-Conditioned Supervised Learning（GCSL）和Online Decision Transformers（ODT）的收敛性和稳定性进行了严格的分析。这些算法在各种基准测试中表现出色，从游戏到机器人任务，但其理论理解仅限于特定的环境条件。本研究为通过监督学习或序列建模解决强化学习问题的算法奠定了理论基础。研究的核心在于分析环境条件，这些条件使得算法能够识别最优解。我们还评估了在环境受到微小噪声干扰时，这些解是否保持稳定。具体而言，我们研究了基于底层马尔可夫决策过程的转移核，命令条件下的策略、值函数和目标到达目标的连续性和渐近收敛性。我们证明了，如果转移核位于确定性核的足够小的邻域内，则可以实现近似最优的行为。这些量在确定性核处是连续的（相对于特定的拓扑结构），无论是渐近地还是在有限次学习周期后。开发的方法使我们能够首次明确估计策略和值的收敛性和稳定性，以底层转移核为依据。在理论方面，我们引入了一些新的概念，如在段空间中工作、在商拓扑中研究连续性以及应用动态系统的不动点理论。理论研究伴随着对示例环境的详细调查和数值实验。关键词：强化学习、上下颠倒强化学习、目标条件监督学习、在线决策变换器、收敛性和稳定性  

©2022 Miroslav Štrupl, Oleg Szehr, Francesco Faccio, Dylan R. Ashley, Rupesh Kumar Srivastava, 和 Jürgen Schmidhuber.  
许可：CC-BY 4.0，见<https://creativecommons.org/licenses/by/4.0/>.  
arXiv:2502.05672v1 [stat.ML] 2025年2月8日  


## 1 引言

强化学习（RL）算法旨在学习在与环境交互时选择最优动作的策略。环境不会揭示最优动作，而是为采取更好的动作提供更高的奖励。这与监督学习（SL）形成直接对比，后者为每个输入提供正确的输出以供学习。尽管如此，仍有一系列算法被提出，试图使用纯监督学习技术解决强化学习任务。上下颠倒强化学习（UDRL）通过将期望的回报/目标映射到动作，将动作预测作为一个监督学习问题来解决，从而颠覆了传统的RL过程[^Schmidhuber2019^][^Srivastava2019^]。目标条件监督学习（GCSL）利用目标信息指导模型的学习过程[^Ghosh2021^]，在线决策变换器（ODT）则利用变换器架构对整个轨迹进行建模，将过去的状体、动作和奖励视为序列，以预测最优动作[^Zheng2022^]。实验表明，这些算法不仅由于依赖于SL而具有惊人的简单性和可扩展性，而且在几个RL基准测试中也能取得良好的结果（例如Vizdoom[^Kempka2016^]、机器人操作[^Ahn2020^]和运动控制[^Fu2020^]）。然而，它们的理论理解仅限于启发式方法和对限制性特殊情况的研究。通过严格分析收敛性和稳定性，本研究开始为通过SL或序列建模解决RL问题的算法奠定理论基础。我们的研究由以下两个问题引导：1）假设给出了底层马尔可夫环境的明确模型（转移核），UDRL、GCSL和ODT的收敛性如何？在无限资源的极限下，典型的感兴趣对象（如策略、状态和动作值）的行为如何？2）在环境模型存在扰动或误差的情况下，这些量的稳定性如何？确保算法能够可靠地识别最优解并在不同条件下保持稳定的保证是它们在现实世界系统中实际部署的基础[^Bertsekas1996^][^Nocedal2006^]。在建立一些基本背景后，值得注意的是，UDRL、GCSL和ODT是非常相似的算法。尽管架构细节有所不同，但它们的核心是关于学习代理获取信息的共同理念。它们都专注于直接根据轨迹中的奖励信号预测动作，而不是学习值函数。关键的共同要素是将奖励、观察和规划范围解释为从其中计算代理命令的任务定义输入。然后通过SL更新代理的行动规则（策略），将之前的轨迹观察和命令映射到动作中，完成学习过程。更正式地说，假设一个遵循某种行动规则$\pi_{\text{old}}$的学习代理收集了一系列轨迹样本。给定一个从状态-动作对$(s, a)$开始、长度为$h$且目标$g$是从段的特征（如状态和奖励的序列）计算得出的轨迹段，可以认为动作$a$对于从$s$在$h$步内实现$g$是有用的。那么，将$(h, g)$解释为代理的命令，并使用SL将新的行动规则$\pi_{\text{new}}$拟合到分布$a|s, h, g$中是合理的，

$$
\pi_{\text{new}} = \arg\max_{\pi} \mathbb{E}[\text{loss}(\pi(a|s, h, g))],
$$

其中$\text{loss}$是适当的损失函数，期望是在轨迹样本中的所有段上计算的。通过将$\pi_{\text{old}}$替换为$\pi_{\text{new}}$，从$\pi_{\text{old}}$中采样新的轨迹，计算实现的范围和目标$(h, g)$，最后使用这些信息更新$\pi_{\text{new}}$，学习过程迭代进行。我们在episodic UDRL（eUDRL）的框架内对UDRL、GCSL和ODT算法进行了分析，eUDRL具有目标结构的特定形式。具体来说，eUDRL假设对段的终止状态应用目标映射$\rho$来评估目标是否达成，即$g = \rho(s)$。GCSL可以被视为eUDRL的一个稍微受限的版本，因为它只关注状态到达任务，并且使用固定的范围。决策变换器（DT）本质上对应于针对离线RL的eUDRL的一次迭代[^Chen2021^]，ODT可以被视为具有熵正则化的eUDRL的一种形式；详见背景部分2的详细信息。与标准RL策略不同，后者根据给定状态定义代理的动作概率，“策略”$\pi_{\text{old}}$和$\pi_{\text{new}}$还根据命令$(h, g)$对这些概率进行条件化。这导致了命令扩展（参见定义1）的形式主义，这是马尔可夫决策过程的一个特殊类别，其中命令被包含为状态的一部分，$\bar{s} = (s, h, g)$。在技术方面，本文开发了命令扩展的数学，为通过SL探索RL提供了坚实的基础。

eUDRL的学习过程只需要一个SL步骤，不需要值函数或策略代理。尽管eUDRL算法的简单性和高实验性能令人瞩目，但只有在确定性环境中，eUDRL才能识别出最优策略。这一限制已经在原始工作中被承认[^Schmidhuber2019^]，在那里为非确定性环境提出了另一种方法（例如，通过使用期望回报而不是实际回报）。在实践中，eUDRL的简单性也促使其被应用于非确定性环境。部分原因是基于许多非确定性环境只表现出轻微的非确定性，即它们可以被视为受到扰动的确定性环境。这种方法在eUDRL[^Srivastava2019^]和GCSL[^Ghosh2021^]的案例中得到了探索，展示了该算法在各个领域的实际效用，包括几个MuJoCo任务。然而，这些文章并没有提供坚实的收敛性保证。尽管Ghosh等人[^Ghosh2021^]表明GCSL优化了目标到达目标的下界，但关于其紧密性的保证是有限的。事实上，即使在接近确定性的环境中，eUDRL的目标到达目标也可能对转移核的扰动敏感。我们将在本文中多次讨论这种行为，并通过具体示例和计算进行说明。考虑图1，以说明特定环境中UDRL生成的目标到达目标的不连续性（见第6节详细信息）。该图显示了目标到达目标的值（用$J_{\alpha}$表示）沿着两个连续的单参数环境转移核族，这些族由$\alpha$参数化，并在$\alpha = 0$处相交。在图1a的情况下，单参数族在确定性环境中相交。在图1b的情况下，族在一个特定的非确定性环境中相交。尽管图1a显示出连续的行为，但目标到达目标在图1b中似乎不连续。尽管eUDRL具有显著的特性，但该算法的稳定性显然令人担忧。

### 图1：eUDRL生成的目标到达目标的不连续性

![](https://via.placeholder.com/500x300)  
**图1**：沿着两个连续的单参数射线（环境的转移核，标记为A和C）的eUDRL生成的目标到达目标的不连续性。水平轴显示射线参数$\alpha$的值；交点为$\alpha = 0$。$\alpha = 0$处相应量的精确值由水平线（标记为B）表示。


### eUDRL在非确定性环境中的不稳定性

eUDRL（特别是UDRL、GCSL和DT）在非确定性环境中的不稳定性已被Štrupl等人[^Strupl2022a^]和Paster等人[^Paster2022^]强调。前者分析了这种不稳定性背后的原则，而后者则针对DT存在的问题，并旨在实现算法的完全功能变体。由于ODT可以被视为具有在线微调功能的DT的一种变体，因此观察到的不稳定性也存在于ODT中。随后的研究，例如Yang等人[^Yang2023^]，提出了对UDRL、GCSL和DT的各种改进。尽管Brandfonbrener等人[^Brandfonbrener2022^]对离线强化学习中的第一次eUDRL迭代进行了深入分析，但对UDRL、GCSL和ODT在有限次迭代和渐近极限下的行为的全面处理仍然是一个开放问题。  
本文并不提出另一种确保eUDRL在非确定性环境中有效的方法，而是试图理解其原始设计，并证明其经验上的成功，正如在Srivastava等人[^Srivastava2019^]和Ghosh等人[^Ghosh2021^]中所见。我们的目标是确定，是否通过多次迭代eUDRL可以实现期望的渐近行为，重点关注严格的收敛性保证和渐近误差界限。为了将确定性转移核的收敛性保证扩展到接近确定性核的近似最优行为，我们采用了拓扑方法。这种方法引入了强化学习领域的新数学工具，例如在商拓扑中引入连续性概念，利用紧集上的测度弱收敛（所谓的相对连续性）。我们证明了在任何有限次迭代后，确定性核处策略的相对连续性和目标到达目标的连续性。尽管这一主题具有重要意义，但由于其复杂性，这一领域尚未得到充分探索。正如我们将展示的，eUDRL策略从第二次迭代开始变得不连续，特别是在转移核集合的边界处（包括确定性核）。这表明，第一次迭代与任何有限次迭代一样有效，但它并没有阐明重复迭代是否能将误差减少到最优行为。通过在特定条件下建立eUDRL递归的界限，我们展示了eUDRL生成策略的累积点集的相对连续性，并证明了相关量（如目标到达目标）的连续性，以及有用的界限和q-线性收敛速率。这些条件概述了两个重要的特殊情况：第一种涉及初始分布的支持条件，第二种以确定性环境中最优策略的唯一性为特征，详见第7节。尽管eUDRL的连续性和稳定性的一般讨论仍然是一个开放问题，但我们发现本文开发的理论足以解决正则化的eUDRL递归。正则化的eUDRL递归与利用熵正则化的ODT非常接近。我们对正则化eUDRL递归的分析，将其表述为更新策略与均匀分布之间的凸组合（类似于ϵ-greedy策略），以完全一般性的方式呈现，而不限制于任何特殊情况。我们讨论的所有内容都限于有限（离散）环境，忽略函数近似和样本量有限的问题。


### 1.1 大纲

第2节提供理解本文所需的背景知识。第3节描述了eUDRL递归在特定段子空间中的形式，并讨论了其与奖励加权回归算法的联系。第4节描述了在所有迭代过程中，eUDRL在环境扰动下稳定的态集合，这为后续讨论eUDRL生成量的连续性提供了有力工具。第5节证明了在确定性环境中，eUDRL收敛到最优策略。第6节研究了在有限次迭代下，确定性环境中eUDRL生成量的连续性。第7节研究了在确定性环境中，eUDRL生成量的累积点集的连续性。第8节在一般情况下研究了正则化eUDRL生成量的累积点集的连续性。第9节讨论了相关工作，第10节总结了本文。为了使本文对更广泛的读者群体易于理解，我们还包含了几个附录。附录A详细介绍了段分布的构建。附录B包含了正文中提到的示例的详细内容。附录C研究了在有限次迭代下，所有转移核内部点处eUDRL生成量的连续性。附录D包含了第8节中正则化eUDRL递归的所有引理和证明的详细内容。

## 2 背景

本节提供了理解本文所需的背景知识。它为我们的工作在“理论景观”中的定位提供了指导，介绍了马尔可夫决策过程的命令扩展类型和段空间分布等基本理论概念。此外，本节还概述了我们的工作如何融入现有文献，描述了ODT如何融入eUDRL框架。本文中介绍的创新性发展将在随后的章节中介绍。

我们将仅处理有限的实值随机变量。它们被定义为映射$X: \Omega \to \mathbb{R}$，具有测度空间$(\Omega, \mathcal{F}, P)$，其中$\Omega$是一个有限集合，$\mathcal{F}$是$\Omega$上的σ代数，$P$表示概率测度。

### **2.1 马尔可夫决策过程**

马尔可夫决策过程（MDPs）是用于不确定动态环境中序贯决策问题的数学框架[^Puterman2014^]。形式上，MDP是一个五元组$M = (S, A, \lambda, \mu, R)$，其中$S$是允许状态的集合，$A$是可能动作的集合，转移概率核$\lambda$定义了在给定状态下采取动作后进入新状态的概率，初始状态的分布为$\mu$，通过这种转移获得的奖励为$R$（随机的）。在MDP中，代理与环境在一系列时间步$t$中进行迭代交互（我们为随机变量添加下标$t$以强调它们属于特定时间点）。设随机变量$S_t: \Omega \to S$描述MDP的状态，随机变量$A_t: \Omega \to A$描述代理选择的动作。从初始状态分布$\mu(s) = P(S_0 = s)$开始，代理在MDP的每一步中观察当前状态$s \in S$，并根据策略$\pi(a|s) = P(A_t = a|S_t = s)$选择相应动作$a \in A$，随后环境根据概率$\lambda(s'|s, a) = P(S_{t+1} = s'|S_t = s, A_t = a)$转移到$s'$。如果对于每个$s, a \in S \times A$，转移核$\lambda(\cdot|s, a)$是确定性的，即存在一个状态$s'_{s,a}$使得$\lambda(s'_{s,a}|s, a) = 1$，则称转移核$\lambda$是确定性的。类似地，如果对于每个$s \in S$，策略$\pi(\cdot|s)$是确定性的，则称策略$\pi$是确定性的。连续的状态-动作转移序列通常被称为MDP的轨迹。在本文中，我们始终假设奖励是确定性的，即奖励由$R_{t+1}(S_{t+1}, S_t, A_t) = r(S_{t+1}, S_t, A_t)$给出，其中$r$是一个确定性函数。从时间$t$开始的回报$G_t = \sum_{k \in \mathbb{N}_0} R_{t+k+1}$是整个MDP剧集期间累积的奖励，我们不对未来奖励进行折扣。遵循策略$\pi$的代理的性能可以通过状态值函数$V^\pi(s) = \mathbb{E}[G_t | S_t = s; \pi]$和动作值函数$Q^\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a; \pi]$来衡量。存在唯一的最优状态值函数$V^* = \max_\pi V^\pi$和唯一的最优动作值函数$Q^* = \max_\pi Q^\pi$，其中最大化是在所有策略的集合上进行的。一个策略$\pi^*$，如果$V^{\pi^*} = V^*$，则必然也有$Q^{\pi^*} = Q^*$，被称为最优的。在接下来的内容中，我们将使用符号$\pi$、$V^\pi$、$Q^\pi$来表示命令扩展类型的MDP
的相应量。


### **2.2 命令扩展类型的马尔可夫决策过程**

eUDRL训练过程的目标是使代理更好地执行命令。在eUDRL中，命令以“目标”和“范围”的形式提供给代理。目标指定了代理需要达到的状态，而范围指定了何时需要达到该状态。目标映射$\rho: S \to G$将用于评估命令的目标是否在命令的范围内达成，即如果$\rho(s)$等于命令指定的目标，则认为目标已达成。目标映射的值域$G$是所有可能目标的集合，无论这些目标是否在特定轨迹中达成。由于目标映射$\rho$定义在整个状态空间$S$上，因此在范围到达时，每个状态$s$都对应一个有效的目标$\rho(s)$。然而，这个目标可能不是所选命令指定的目标。目标映射的引入允许我们在统一的形式化框架内研究典型的eUDRL任务：状态到达任务可以通过$G = S$、$\rho = \text{id}_S$（$S$上的恒等映射）来覆盖，而与原始MDP $M$相关的回报目标可以通过扩展状态，增加一个累积奖励的组件，并定义$\rho$为该组件的投影来覆盖。除了MDP状态外，eUDRL代理还接收一个额外的命令输入。设随机变量$G_t: \Omega \to G$描述eUDRL代理的目标，随机变量$H_t: \Omega \to \bar{N}_0 = \{0, 1, \dots, N\}$，$N \geq 1$，描述剩余范围。可以将eUDRL代理视为在扩展状态空间中运行的普通代理，该扩展状态空间通过命令扩展。在这种情况下，需要提供一个关于目标和范围的初始分布$P(H_0 = h, G_0 = g | S_0 = s)$，并且随着扩展MDP的每次转移，剩余范围减少1，直到范围达到0。此时，扩展MDP进入吸收状态（见下文定义），不再发生进一步的演变。如果在范围变为0时达到指定目标，则会获得奖励。总结如下，我们给出命令扩展的定义：

**定义1（命令扩展）** 一个马尔可夫决策过程$M = (S, A, \lambda, \mu, r)$的命令扩展（CE）是一个形式为$\bar{M} = (\bar{S}, A, \bar{\lambda}, \bar{\mu}, \bar{r}, \rho)$的马尔可夫决策过程，其中：

- 一个命令是一个对$(g, h) \in G \times \bar{N}_0$，其中$G \subset \mathbb{Z}^n_G$，$\bar{N}_0 = \{0, 1, \dots, N\}$。$N \geq 1$表示最大范围，$G$表示CE的目标集。目标映射$\rho: S \to G$用于评估目标$g \in G$是否达成。
- 扩展状态空间是$\bar{S} = S \times \bar{N}_0 \times G$，扩展状态是一个三元组$\bar{s} = (s, h, g) \in \bar{S}$，由原始MDP状态$s$和命令$(g, h)$组成。
- 扩展MDP的初始分布由命令分布和$M$的初始分布的乘积给出
$$
\bar{\mu}(\bar{s}) = P(H_0 = h, G_0 = g | S_0 = s)\mu(s).
$$
- 转移核$\bar{\lambda}$对所有$(s, h, g) \in \bar{S}$，所有$s' \in S$和$a \in A$定义为
$$
\bar{\lambda}((s', h - 1, g) | (s, h, g), a) = \lambda(s' | s, a) \quad \text{如果 } h > 0,
$$
$$
\bar{\lambda}((s', h, g) | (s, h, g), a) = \delta_{ss'} \quad \text{如果 } h = 0.
$$
- 奖励函数$\bar{r}$对所有$\bar{s}' = (s', h', g') \in \bar{S}$，$\bar{s} = (s, h, g) \in \bar{S}$和所有$a \in A$定义为
$$
\bar{r}((s', h', g'), (s, h, g), a) =
\begin{cases}
1\{\rho(s') = g\} & \text{如果 } h = 1, h' = 0, g' = g, \\
0 & \text{其他情况}.
\end{cases}
$$
我们将通常将$M$称为原始MDP，将$\bar{M}$称为扩展MDP。类似地，$s$是原始状态，$\bar{s}$是扩展状态。

在随后的讨论中，我们几乎完全围绕扩展MDP的策略展开，而不是原始MDP的策略。为了简洁，我们保留符号$\pi$（而不是$\bar{\pi}$）用于CE代理的策略。奖励的定义旨在引导代理实现预期的命令。选择二元奖励函数允许我们将期望奖励值解释为概率。总结如下，状态和动作值函数的计算结果是，对于所有$(s, h, g) \in \bar{S}$，$h > 1$，以及所有动作$a \in A$，
$$
Q^\pi((s, h, g), a) = \mathbb{P} \left[ \rho(S_{t+h}) = g \middle| S_t = s, H_t = h, G_t = g, A_t = a; \pi \right],
$$
$$
V^\pi(s, h, g) = \mathbb{P} \left[ \rho(S_{t+h}) = g \middle| S_t = s, H_t = h, G_t = g; \pi \right]. \quad (2.1)
$$
有了这些状态和动作值函数，CE可以被解释为普通的有限范围RL问题。可以选择各种保证收敛到最优策略的算法，例如Sutton和Barto[^Sutton2018^]的专著。然而，本研究致力于研究eUDRL算法，它不仅仅是应用于CE类型的MDP的普通RL算法。相反，eUDRL是通过一个特定的迭代过程在CE上定义的（涉及轨迹采样、监督学习和策略更新），详见第2.4节的详细描述，尽管具有一些方便的特性，但并不能保证总是收敛到最优解。在eUDRL的研究中，使用Ghosh等人[^Ghosh2021^]的目标到达目标将非常方便，它源自CE值函数，可以写为
$$
J^\pi = \sum_{\bar{s} \in \bar{S}} \bar{\mu}(\bar{s}) V^\pi(\bar{s}). \quad (2.2)
$$
存在唯一的最优目标到达目标$J^* = \max_\pi J^\pi$，其中最大化是在所有策略的集合上进行的。很容易看出，对于任何最优策略$\pi^*$，都有$J^{\pi^*} = J^*$。为了方便起见，我们引入了CE的吸收状态的单独符号。回顾一下，对于MDP $M$，如果一旦达到吸收状态，底层过程总是保持在这个状态，即对于所有状态$s, s'$和动作$a$，核等于克罗内克δ函数$\lambda(s'|s, a) = \delta_{ss'}$，则称状态$s \in S$为吸收状态。

**注2（CE的吸收状态）** 设$\bar{M} = (\bar{S}, A, \bar{\lambda}, \bar{\mu}, \bar{r}, \rho)$是一个CE。形式为$(s, g, h) \in \bar{S}$的状态是吸收状态当且仅当$h = 0$。非吸收状态称为瞬态状态。我们分别用$\bar{S}_A \subset \bar{S}$和$\bar{S}_T \subset \bar{S}$表示吸收状态和瞬态状态的集合。

初始化一个CE进入吸收状态不会产生任何可学习的信息，因为CE将保持在这个状态，无论选择什么动作[^Strupl2022a^]。因此，我们将专注于那些未初始化到吸收状态的CE。换句话说，我们假设所研究的CE的初始分布为其赋予吸收状态的概率为0，即$\text{supp} \ \bar{\mu} \subset \bar{S}_T$[^Strupl2022a^]。这种CE被称为非退化的。此后，我们只研究非退化的CE，简称为CE。

## 2.3 CE的轨迹和段分布

在eUDRL算法中，采样CE轨迹的段起着关键作用。一个CE轨迹包含一系列连续的扩展状态和动作的转移。从一般意义上讲，CE轨迹可以表示为无限序列的形式：
$$
\tau = ((s_0, h_0, g_0), a_0, (s_1, h_0 - 1, g_0), a_1, \dots, (s_{h_0-1}, 1, g_0), a_{h_0-1}, (s_{h_0}, 0, g_0), \dots).
$$
然而，一旦CE达到最大范围，它就会进入吸收状态，之后不再发生进一步的演变[^Strupl2022a^]。因此，用长度为$N$的有限序列来表示轨迹就足够了。设$\text{Traj} \subset (\bar{S} \times A)^N \times \bar{S}$表示长度为$N$的有限轨迹的子集，其转移满足定义1中关于范围和目标的动态要求。在接下来的内容中，我们将用以下形式表示轨迹：
$$
\tau = ((s_0, h_0, g_0), a_0, (s_1, h_0 - 1, g_0), a_1, \dots, (s_{l(\tau)-1}, 1, g_0), a_{l(\tau)-1}, (s_{l(\tau)}, 0, g_0), \dots, (s_N, 0, g_0)) \in \text{Traj}.
$$
其中，$l(\tau)$表示轨迹$\tau$首次进入吸收状态之前的转移次数；它等于轨迹的初始范围。假设状态和动作空间是有限的，$\text{Traj}$是可测的，我们用$T: \Omega \to \text{Traj}$表示一个随机变量，其分量为$T = ((S_0, H_0, G_0), A_0, \dots, (S_N, H_N, G_N))$，其结果为CE轨迹。轨迹$\tau \in \text{Traj}$的概率由下式给出：
$$
P(T = \tau; \pi) = \left( \prod_{t=1}^{l(\tau)} \lambda(s_t | a_{t-1}, s_{t-1}) \right) \cdot \left( \prod_{t=0}^{l(\tau)-1} \pi(a_t | \bar{s}_t) \right) \cdot \bar{\mu}(\bar{s}_0).
$$

每个段包含来自给定轨迹的一段连续的状态-动作转移，以及段的初始范围、目标和长度。我们将用以下形式的元组表示段：
$$
\sigma = (l(\sigma), s_{\sigma}^0, h_{\sigma}^0, g_{\sigma}^0, a_{\sigma}^0, s_{\sigma}^1, a_{\sigma}^1, \dots, s_{\sigma}^{l(\sigma)}),
$$
其中，$a_{\sigma}^0, \dots, a_{\sigma}^{l(\sigma)-1}$是选择的动作，$s_{\sigma}^0, \dots, s_{\sigma}^{l(\sigma)}$是达到的原始MDP状态，下标$t$表示段内相应量的时间（即与原始MDP中状态或动作实际发生的时间无关）。量$l(\sigma)$表示段的长度（即在段的执行过程中执行的动作数量），$h_{\sigma}^0$和$g_{\sigma}^0$表示段的初始范围和目标。在我们的表示中，除了第一个扩展状态外，我们省略了所有扩展状态的范围和目标，因为它们已经由CE的范围和目标动态确定。我们用$\text{Seg}$表示所有这种形式的元组的集合。假设状态和动作空间是有限的，$\text{Seg}$是可测的，我们用$\Sigma: \Omega \to \text{Seg}$表示一个随机变量，其分量为$\Sigma = (l(\Sigma), S_{\Sigma}^0, H_{\Sigma}^0, G_{\Sigma}^0, A_{\Sigma}^0, S_{\Sigma}^1, A_{\Sigma}^1, \dots, S_{\Sigma}^{l(\Sigma)})$，其结果为CE轨迹的段。我们在附录A中提供了关于$\Sigma$分布构造的所有形式细节。在这里，我们只总结了对于所有$\sigma \in \text{Seg}$，有：
$$
P(\Sigma = \sigma; \pi) = c^{-1} \sum_{t \leq N - l(\sigma)} P(S_t = s_{\sigma}^0, H_t = h_{\sigma}^0, G_t = g_{\sigma}^0, A_t = a_{\sigma}^0, \dots, S_{t + l(\sigma)} = s_{\sigma}^{l(\sigma)}; \pi),
$$
其中$c$是归一化常数：
$$
c = \sum_{\sigma \in \text{Seg}} \sum_{t \leq N - l(\sigma)} P(S_t = s_{\sigma}^0, H_t = h_{\sigma}^0, G_t = g_{\sigma}^0, A_t = a_{\sigma}^0, \dots, S_{t + l(\sigma)} = s_{\sigma}^{l(\sigma)}; \pi).
$$
在附录中，我们证明了归一化常数的以下界限：
$$
0 < c \leq \frac{N(N + 1)}{2}. \quad (2.4)
$$
我们还将在附录中证明$\Sigma$分布的以下“马尔可夫性”：对于任意长度为$k$的段和任意$i \leq k$，有
$$
P(A_{\Sigma}^i = a_i | l(\Sigma) = k, S_{\Sigma}^0 = s^0, H_{\Sigma}^0 = h^0, G_{\Sigma}^0 = g^0, A_{\Sigma}^0 = a^0, \dots, S_{\Sigma}^i = s^i; \pi) = \pi(a_i | s^i, h - i, g), \quad (2.5)
$$
$$
P(S_{\Sigma}^i = s^i | l(\Sigma) = k, S_{\Sigma}^0 = s^0, H_{\Sigma}^0 = h^0, G_{\Sigma}^0 = g^0, A_{\Sigma}^0 = a^0, \dots, S_{\Sigma}^{i-1} = s^{i-1}, A_{\Sigma}^{i-1} = a^{i-1}; \pi) = P(S_{\Sigma}^i = s^i | S_{\Sigma}^{i-1} = s^{i-1}, A_{\Sigma}^{i-1} = a^{i-1}) = \lambda(s^i | s^{i-1}, a^{i-1}). \quad (2.6)
$$

### **限制到尾随段**
有时将分析限制到$\text{Seg}$的特定子集是有用的。例如，考虑所谓的“尾随段”会很有用。一个段$\sigma = (l(\sigma), s_{\sigma}^0, h_{\sigma}^0, g_{\sigma}^0, a_{\sigma}^0, \dots, s_{\sigma}^{l(\sigma)})$是尾随的，如果它与轨迹的末端对齐，即如果$l(\sigma) = h_{\sigma}^0$。由所有尾随段组成的$\text{Seg}$的子空间记为$\text{Seg}_{\text{trail}} := \{\sigma \in \text{Seg} | l(\sigma) = h_{\sigma}
^0\}$。ODT和有时eUDRL操作在尾随段上。在eUDRL的情况下，这种限制是出于加快学习过程的考虑，反映了问题的阶段性[^Strupl2022a^]。

### **限制到对角段**
满足$g_{\sigma}^0 = \rho(s_{\sigma}^{l(\sigma)})$的段称为目标对角段。这样的段的特点是段的初始目标实际上在段的末尾达成。一个既是尾随的又是目标对角的段称为对角段。由所有对角段组成的$\text{Seg}$的子空间记为$\text{Seg}_{\text{diag}} := \{\sigma \in \text{Seg} | l(\sigma) = h_{\sigma}^0, g_{\sigma}^0 = \rho(s_{\sigma}^{l(\sigma)})\}$。我们在讨论与奖励加权回归（Peters和Schaal, 2007）的关系时会考虑这些段，见第3.2节[^Peters2007^]。



### 2.4 eUDRL算法

假设给定一个CE $\bar{M}$。eUDRL从初始策略$\pi_0$开始，通过迭代过程生成策略序列$(\pi_n)_{n \geq 0}$。每次eUDRL迭代包括以下步骤：首先，根据当前策略$\pi_n$从$\bar{M}$中生成一批轨迹；其次，根据$\Sigma$的分布$d_{\pi_n}^{\Sigma}$从这批轨迹中采样轨迹段$\sigma$；最后，使用监督学习将后续策略$\pi_{n+1}$拟合到采样的轨迹段上。在实践中，通常假设一个参数化的策略类别，$\pi_{n+1}$将是该类别中最小化对数似然损失的策略之一。在这里，我们研究eUDRL在原则上可以实现什么。我们假设没有特定的策略表示，并假设$\pi_{n+1}$是使用交叉熵计算的：
$$
\pi_{n+1} = \arg\max_{\pi} \mathbb{E}_{\sigma \sim d_{\pi_n}^{\Sigma}} \left[ \log \pi(a_{\sigma}^0 | s_{\sigma}^0, l(\sigma), \rho(s_{\sigma}^{l(\sigma)})) \right], \quad (2.7)
$$
这反映了无限样本大小或对分布$d_{\pi_n}^{\Sigma}$的完全了解。换句话说，新策略$\pi_{n+1}$被拟合到条件概率上：
$$
\pi_{n+1}(a | (s, h, g)) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^{l(\sigma)}) = g; \pi_n). \quad (2.8)
$$
方程(2.7)中的条件选择可以通过与Andrychowicz等人的Hindsight Experience Replay（HER）算法[^Andrychowicz2017^]进行比较来解释。eUDRL使用段样本的方式与HER相似，不仅使用实现预期目标的轨迹进行学习，还使用未实现预期目标的轨迹及其实际结果。按照类似的推理，如果观察到一个样本$\sigma \sim d_{\pi_n}^{\Sigma}$，可以假设第一个动作$a_{\sigma}^0$是执行实现命令$(l(\sigma), \rho(s_{\sigma}^{l(\sigma)}))$的一个好选择，无论这个段实际选择的命令$(h_{\sigma}^0, g_{\sigma}^0)$是什么。许多遵循“通过监督学习实现强化学习”范式的算法，包括Srivastava等人的eUDRL实际实现[^Srivastava2019^]、Ghosh等人的目标条件监督学习（GCSL）[^Ghosh2021^]和Zheng等人的在线决策变换器（ODT）[^Zheng2022^]，都符合本文的框架。应该指出的是，我们假设的无限样本大小意味着与Srivastava等人[^Srivastava2019^]或Ghosh等人[^Ghosh2021^]讨论的eUDRL变体相比有所简化，我们省略了回放缓冲区（因为它不是必需的）。Srivastava等人的算法使用按回报排序的轨迹组织回放缓冲区，其中目标与轨迹回报相关。此外，根据回放缓冲区中回报分布的最佳分位数采样初始命令。这意味着初始分布会随着时间变化，而在这里我们假设它是固定的。Ghosh等人的GCSL算法可以被视为Srivastava等人算法的一个简化版本，GCSL执行纯状态到达任务。GCSL省略了命令中的范围分量，假设其任务的范围是固定的。与Srivastava等人一样，GCSL使用回放缓冲区，但没有优先级。eUDRL与ODT的关系在下面的第2.5节中描述。

1. 注意到$\pi_{n+1}$由(2.8)定义当且仅当$P(S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^{l(\sigma)}) = g; \pi_n) > 0$，否则我们暂时将其留作未定义。稍后，为了方便起见，如果$P(S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^{l(\sigma)}) = g; \pi_n) = 0$，我们选择$\pi_{n+1} = 1/|A|$。


### 2.5 eUDRL与ODT递归的关系

将成功的Transformer架构（参见[^Schmidhuber1992^], [^Vaswani2017^], [^Schlag2021^]）与UDRL相结合的想法导致了在线决策变换器（ODT）的概念，它将强化学习解释为一个序列建模问题。所讨论的序列由状态、动作以及MDP的“剩余回报”组成的元组$(s'_t, a'_t, g'_t)$。剩余回报$g'_t$在轨迹完成后计算，由从时间$t$到轨迹结束时累积的奖励给出[^Zheng2022^]。

在ODT中，Transformer模型接收状态序列和剩余回报，以预测后续动作。为了在每个时间步$t$近似$a'_t$的分布，Transformer（上下文长度为$K$）接收长度为$\min\{t, K\}$的先前状态序列$s'_{-K,t} := s'_{\min\{0,t-K\}:t}$和剩余回报$g'_{-K,t} := g'_{\min\{0,t-K\}:t}$。与UDRL类似，这个分布被用作基于剩余回报的策略。与原始Transformer架构（使用位置编码）不同，ODT使用时间嵌入。此外，ODT仅在尾随段上操作。与前身决策Transformer（DT）相比，ODT引入了一系列改进，允许在线微调。首先，ODT使用随机策略（而DT仅允许确定性策略），其次，这些策略使用最大似然准则进行训练（而DT依赖于均方误差）。这些特性也以类似的形式出现在eUDRL算法中。此外，ODT引入了熵正则化，这促使我们研究eUDRL的正则化技术。在第8节中，我们将考虑使用均匀分布的凸组合对eUDRL进行策略正则化，这使我们的研究接近ODT算法[^Zheng2022^]。

像Srivastava等人的eUDRL实现[^Srivastava2019^]和Ghosh等人的GCSL[^Ghosh2021^]一样，ODT也需要一些简化，以符合CE框架。与之前一样，我们省略了回放缓冲区，并假设固定范围，这意味着CE初始化范围为$N$。这是为了确保剩余范围$h$和当前时间$t$之间的一一对应关系，通过$t = N - h$。因此，在ODT执行时间嵌入后，这种差异变得无关紧要。在这些简化下，ODT在CE框架内匹配目标到达场景，其中奖励在剧集结束时授予[^Zheng2022^]。对应关系如下：首先，ODT要解决的MDP的状态$s'_t$被组合成过去的序列$s_t = s'_{-K,t}$。底层CE的原始MDP操作在这些组合状态上。目标映射的值域被选择为集合$G = \{\text{所有可能的回报}\}$，其中回报仅依赖于终端状态$s_N$。策略更新使用最大似然匹配到条件$a_t|s_t, l(\sigma), \rho(s_N)$。使用ODT的尾随段假设、固定范围$l(\sigma) = h_t$以及$\
rho(s_N)$计算与$s'_N$对应的回报，这等价于匹配$a_t|s'_{-K,t}, h_t, \rho(s_N)$。一般情况下，只能部分地用CE框架的语言来描述这种情况，通过将回报作为状态的一部分进行累积。我们假设状态由$s_t = (s'_{-K,t}, z_{-K,t})$给出，$z_t$对应于底层MDP的累积回报。目标映射被选择为$\rho(s_t) = z_t$。在这种情况下，策略更新是通过最大似然匹配到条件$a_t|(s'_{-K,t}, z_{-K,t}), h_t, z_N$进行的，其中从$z_N - z_{-K,t}$可以恢复剩余回报。这并不完全符合CE框架，因为我们不仅条件于期望的剩余回报$z_N - z_{-K,t}$，还条件于$(z_N, z_{-K,t})$。原则上，这可能会在学习过程中引入一些噪声。然而，这并不是问题，因为我们假设样本数量没有限制。在ODT的CE框架中进行公式化没有缺点，因为噪声的影响在这种情况下变得可以忽略不计[^Zheng2022^]。

## 3 eUDRL递归、奖励加权回归及实际实现

标准的eUDRL递归是在整个段空间上进行的。在本节中，我们将展示在尾随段和对角段子空间中进行递归时的公式。这将使我们能够将eUDRL与奖励加权回归（RWR）算法联系起来[^Dayan1997^][^Peters2007^]。RWR已知的收敛性[^Strupl2022b^]将为我们在后续章节中的证明提供直观的指导。

### **3.1 特定段子空间中的eUDRL递归**

以下引理指定了当递归仅使用来自特定子集的段进行拟合时，eUDRL递归如何表示（见上文关于尾随段和对角段的讨论）。回顾一下，若$l(\sigma) = h_{\sigma}^0$，则称段$\sigma = (l(\sigma), s_{\sigma}^0, h_{\sigma}^0, g_{\sigma}^0, a_{\sigma}^0, \dots, s_{\sigma}^{l(\sigma)}) \in \text{Seg}$为尾随段；若该段为尾随段且$\rho(s_{\sigma}^{l(\sigma)}) = g_{\sigma}^0$，则称其为对角段。为了节省空间，避免写出策略分布中的冗长归一化因子（这些因子可以很容易地恢复），我们将在引理（及其证明）中经常使用比例符号$\propto$而不是等号。因此，这里的“比例性”仅指动作维度。

**引理3** 考虑由方程(2.7)描述的eUDRL中的递归策略更新。

1. **假设递归在整个集合$\text{Seg}$上进行**。那么对于所有状态$(s, h, g) \in \bar{S}_T$和所有动作$a \in A$，有
   $$
   \pi_{n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g; \pi_n)
   $$
   $$
   \propto \sum_{h' \geq h, g' \in G} P(\rho(S^h) = g | A_0 = a, H_0 = h', G_0 = g', S_0 = s; \pi_n)
   $$
   $$
   \cdot \pi_n(a | s, h', g') P(H_{\Sigma}^0 = h', G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n). \quad (3.1)
   $$

2. **假设递归在集合$\text{Seg}_{\text{trail}}$上进行**。那么对于所有状态$(s, h, g) \in \bar{S}_T$和所有动作$a \in A$，有
   $$
   \pi_{\text{trail}, n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
   $$
   $$
   \propto \sum_{g' \in G} P(\rho(S^h) = g | A_0 = a, H_0 = h, G_0 = g', S_0 = s; \pi_{\text{trail}, n})
   $$
   $$
   \cdot \pi_{\text{trail}, n}(a | s, h, g') P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{trail}, n}). \quad (3.2)
   $$

3. **假设递归在集合$\text{Seg}_{\text{diag}}$上进行**。那么对于所有状态$(s, h, g) \in \bar{S}_T$和所有动作$a \in A$，有
   $$
   \pi_{\text{diag}, n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, \rho(S_{\Sigma}^h) = G_{\Sigma}^0, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
   $$
   $$
   \propto P(\rho(S^h) = g | A_0 = a, H_0 = h, G_0 = g, S_0 = s; \pi_{\text{diag}, n})
   $$
   $$
   \cdot \pi_{\text{diag}, n}(a | s, h, g) P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{diag}, n}). \quad (3.3)
   $$
   此外，策略可以用Q函数表示为
   $$
   \pi_{\text{diag}, n+1}(a | s, h, g) \propto Q^{\pi_{\text{diag}, n}}((s, h, g), a) \pi_{\text{diag}, n}(a | s, h, g)
   $$
   $$
   \cdot P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{diag}, n}),
   $$
   $$
   \pi_{\text{diag}, n+1}(a | s, h, g) \propto Q^{\pi_{\text{diag}, n}}((s, h, g), a) \pi_{\text{diag}, n}(a | s, h, g).
   $$
   注意，$\text{Seg}$、$\text{Seg}_{\text{trail}}$和$\text{Seg}_{\text{diag}}$中的递归公式有一个基本的相似性，即尽管$\pi_{\text{diag}/\text{trail}, n+1}(a | s, h, g)$依赖于条件概率$P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)$，但它并不依赖于定义$\text{Seg}_{\text{diag}}$和$\text{Seg}_{\text{trail}}$的条件$l(\Sigma) = H_{\Sigma}^0$和$\rho(S_{\Sigma}^{l(\sigma)}) = G_{\Sigma}^0$。这些递归公式仅通过求和所涉及的集合不同而有所区别。稍后，这一特性将使我们能够为所有三种递归推导出共同的界限（即用$\pi_n$来界定$\pi_{n+1}$），从而得到更简单的证明。

**证明**  
**引理3，第1点**：等式直接来自(2.8)。我们首先应用贝叶斯法则，并假设$P(S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g; \pi_n) > 0$。随后，我们对$H_{\Sigma}^0$和$G_{\Sigma}^0$进行边缘化处理，前提是$H_{\Sigma}^0 \geq l(\Sigma) = h$（根据定义，段总是包含在轨迹中），并应用乘法规则。我们得到：
$$
\pi_{n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g; \pi_n)
$$
$$
\propto P(\rho(S_{\Sigma}^h) = g, A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)
$$
$$
= \sum_{h' \geq h, g' \in G
} P(\rho(S_{\Sigma}^h) = g, A_{\Sigma}^0 = a, H_{\Sigma}^0 = h', G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)
$$
$$
= \sum_{h' \geq h, g' \in G} P(\rho(S_{\Sigma}^h) = g | A_{\Sigma}^0 = a, H_{\Sigma}^0 = h', G_{\Sigma}^0 = g', S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)
$$
$$
\cdot P(A_{\Sigma}^0 = a | H_{\Sigma}^0 = h', G_{\Sigma}^0 = g', S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)
$$
$$
\cdot P(H_{\Sigma}^0 = h', G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n)
$$
$$
= \sum_{h' \geq h, g' \in G} P(\rho(S^h) = g | A_0 = a, H_0 = h', G_0 = g', S_0 = s; \pi_n)
$$
$$
\cdot \pi_n(a | s, h', g') P(H_{\Sigma}^0 = h', G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_n),
$$
其中在最后一个等式中，我们应用了段分布的性质（2.5）和（2.6）。

**引理3，第2点**：我们有
$$
\pi_{\text{trail}, n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
$$
$$
= P(A_{\Sigma}^0 = a, H_{\Sigma}^0 = h | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n}).
$$
第一个等式是(2.7)的直接结果，其中添加了条件$l(\Sigma) = H_{\Sigma}^0$，以将方程限制在$\text{Seg}_{\text{trail}}$上。对于第二个等式，我们利用了蕴含关系$(l(\Sigma) = h) \land (l(\Sigma) = H_{\Sigma}^0) \Rightarrow H_{\Sigma}^0 = h$。按照上述推理，我们应用贝叶斯法则，并假设$P(S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n}) > 0$。随后，我们对$H_{\Sigma}^0$和$G_{\Sigma}^0$进行边缘化处理，应用乘法规则，并利用蕴含关系$(l(\Sigma) = h) \land (H_{\Sigma}^0 = h) \Rightarrow l(\Sigma) = H_{\Sigma}^0$来消除事件$l(\Sigma) = H_{\Sigma}^0$。我们得到：
$$
\pi_{\text{trail}, n+1}(a | s, h, g) \propto P(\rho(S_{\Sigma}^h) = g, A_{\Sigma}^0 = a, H_{\Sigma}^0 = h | S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
$$
$$
= \sum_{g' \in G} P(\rho(S_{\Sigma}^h) = g, A_{\Sigma}^0 = a, H_{\Sigma}^0 = h, G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
$$
$$
= \sum_{g' \in G} P(\rho(S_{\Sigma}^h) = g | A_{\Sigma}^0 = a, H_{\Sigma}^0 = h, G_{\Sigma}^0 = g', S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
$$
$$
\cdot P(A_{\Sigma}^0 = a | H_{\Sigma}^0 = h, G_{\Sigma}^0 = g', S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{trail}, n})
$$
$$
\cdot P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{trail}, n})
$$
$$
= \sum_{g' \in G} P(\rho(S^h) = g | A_0 = a, H_0 = h, G_0 = g', S_0 = s; \pi_{\text{trail}, n})
$$
$$
\cdot \pi_{\text{trail}, n}(a | s, h, g') P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g' | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{trail}, n}).
$$

**引理3，第3点**：我们有
$$
\pi_{\text{diag}, n+1}(a | s, h, g) = P(A_{\Sigma}^0 = a | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, \rho(S_{\Sigma}^h) = G_{\Sigma}^0, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
$$
$$
= P(A_{\Sigma}^0 = a, H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h, \rho(S_{\Sigma}^h) = g, \rho(S_{\Sigma}^h) = G_{\Sigma}^0, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
$$

第一个等式是公式(2.7)的结果，其中添加了条件$l(\Sigma)=H_{0}^{\Sigma}$和$\rho(S_{0}^{\Sigma}) = G_{0}^{\Sigma}$，以便将该等式限制在$\mathrm{Seg}^{\mathrm{diag}}$ 上。对于第二个等式，我们利用了蕴含关系$(l(\Sigma)=h)\land(l(\Sigma)=H_{0}^{\Sigma})\Longrightarrow H_{0}^{\Sigma}=h$ 以及$(\rho(S_{0}^{\Sigma}) = g)\land(\rho(S_{0}^{\Sigma})=G_{0}^{\Sigma})\Longrightarrow G_{0}^{\Sigma}=g$ 。按照上述推理，我们应用贝叶斯规则，其中我们假设$\mathbb{P}(S_{0}^{\Sigma}=s, l(\Sigma)=h,\rho(S_{0}^{\Sigma}) = g,\rho(S_{h}^{\Sigma})=G_{0}^{\Sigma},l(\Sigma)=H_{0}^{\Sigma};\pi^{\mathrm{diag}})>0$ 。随后，我们对$H_{0}^{\Sigma}$ 和$G_{0}^{\Sigma}$ 进行边缘化，应用乘积规则，并利用蕴含关系$(\rho(S_{0}^{\Sigma}) = g)\land(G_{0}^{\Sigma}=g)\Longrightarrow\rho(S_{h}^{\Sigma})=G_{0}^{\Sigma}$ 以及$(l(\Sigma)=h)\land(H_{0}^{\Sigma}=h)\Longrightarrow l(\Sigma)=H_{0}^{\Sigma}$ 来消除事件$\rho(S_{h}^{\Sigma}) = G_{0}^{\Sigma}$ 和$l(\Sigma)=H_{0}^{\Sigma}$ 。我们得到：

$$
\pi_{\text{diag}, n+1}(a | s, h, g) \propto P(\rho(S_{\Sigma}^h) = g, \rho(S_{\Sigma}^h) = G_{\Sigma}^0, A_{\Sigma}^0 = a, H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
$$
$$
= P(\rho(S_{\Sigma}^h) = g | A_{\Sigma}^0 = a, H_{\Sigma}^0 = h, G_{\Sigma}^0 = g, S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
$$
$$
\cdot P(A_{\Sigma}^0 = a | H_{\Sigma}^0 = h, G_{\Sigma}^0 = g, S_{\Sigma}^0 = s, l(\Sigma) = h, l(\Sigma) = H_{\Sigma}^0; \pi_{\text{diag}, n})
$$
$$
\cdot P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g, l(\Sigma) = H_{\Sigma}^0 | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{diag}, n}).
$$
最后，利用动作价值函数的表达式（2.1），我们得到：
$$
\pi_{\text{diag}, n+1}(a | s, h, g) \propto Q^{\pi_{\text{diag}, n}}((s, h, g), a) \pi_{\text{diag}, n}(a | s, h, g) P(H_{\Sigma}^0 = h, G_{\Sigma}^0 = g | S_{\Sigma}^0 = s, l(\Sigma) = h; \pi_{\text{diag}, n})
$$
$$
\propto Q^{\pi_{\text{diag}, n}}((s, h, g), a) \pi_{\text{diag}, n}(a | s, h, g).
$$

## 9 相关工作 (Related Work)

在本节中，我们首先讨论了使用迭代监督学习解决强化学习问题的早期关键工作（第9.1节）。然后，我们探讨了与目标条件强化学习（Goal-conditioned Reinforcement Learning）相关的研究（第9.2节）。接着，我们重点关注与上下倒置强化学习（Upside-Down Reinforcement Learning, UDRL）直接相关的工作（第9.3节）。最后，我们讨论了与决策变换器（Decision Transformers）和在线决策变换器（Online Decision Transformers）中使用的Transformer架构相关的重要工作（第9.4节）。


### 9.1 奖励加权回归 (Reward-Weighted Regression)

使用迭代监督学习解决强化学习问题的想法最早由Peters和Schaal（2007）以**奖励加权回归（Reward-Weighted Regression, RWR）**的形式进行了研究，该研究基于Dayan和Hinton（1997）提出的期望最大化（Expectation-Maximization）框架。然而，他们研究的场景非常有限：Dayan和Hinton（1997）仅考虑了**单臂老虎机（bandit）场景**（只有一个状态和有限的动作），而Peters和Schaal（2007）则将研究限制在**即时奖励问题**上。

将RWR扩展到完整的**逐集（episodic）设置**（考虑整个回报而非仅即时奖励）的工作由Wierstra等人（2008）和Kober与Peters（2011）完成。Hachiya等人（2009, 2011）随后讨论了RWR中的高效离线策略方案。最后，Peng等人（2019）将深度神经网络近似器引入RWR。

关于RWR的理论研究相对较少。Dayan和Hinton（1997）以及Peters和Schaal（2007）分别在他们的设置中证明了RWR的**单调性**，这些证明都基于期望最大化范式。十多年后，Strupl等人（2022b）证明了RWR在**紧致状态和动作空间**中收敛到全局最优（假设样本数量无限且不使用函数近似）。

RWR与本工作特别相关，因为eUDRL在**对角片段（Segdiag）**上的表现与RWR一致（如第3节所述）。这种联系激发了我们在确定性核处对eUDRL生成量的连续性证明的方法。

### 9.2 无重标记的目标条件强化学习 (Goal-conditioned Reinforcement Learning Without Relabeling)

据我们所知，Schmidhuber和Huber（1991, 1990）首次提出了目标条件强化学习的概念，当时是在学习选择性注意力的背景下。在这项工作中，存在额外的目标定义输入模式，用于编码各种任务，以便RL机器知道接下来要执行哪个任务。每个任务结束时会给予奖励。通过一个单独的环境模型网络来解决通过非可微环境传播学习信号的需求。本质上，这是一种具有稀疏奖励的基于模型的强化学习，与监督学习（SL）等价，只是样本是在线收集的。但请注意，与UDRL不同，这项工作以及其他本段提到的工作都缺少**片段/轨迹重标记**。

Schmidhuber和Huber（1991）的工作也为关于注意力的文献做出了贡献，为当今的Transformer架构奠定了基础。Sutton等人（1999）提出的**选项框架（Options Framework）**也可以被视为目标条件强化学习的一种形式（一个选项决定了一个策略）。向策略网络提供目标可以实现对未见目标的泛化，当用深度网络表示策略时（价值函数类似）。基于Sutton等人（2011）的工作，Schaul等人（2015）提出了目标条件价值函数的便捷近似器。例如，Faccio等人（2023）最近的一篇论文引入了用于深度策略网络的目标条件策略生成器。


### 9.3 上下倒置强化学习 (UDRL)

UDRL（Schmidhuber, 2019年12月5日）通过使用轨迹片段和实现的目标/回报进行重标记，扩展了迭代监督学习的思想，以学习命令条件策略。这背后的动机是希望在表示UDRL策略时通过深度神经网络实现对目标/回报的泛化，并且更高效地利用数据（给定轨迹中的片段数量多于状态数量）。轨迹重标记在Andrychowicz等人（2017）提出的**后见之明经验回放（Hindsight Experience Replay, HER）**中已经出现，我们将在下文更详细地讨论这项工作。

Schmidhuber（2019年12月5日）设计的特定算法（我们在此称为eUDRL）最初仅用于**确定性环境**（尽管Schmidhuber（2019年12月5日）对随机环境的变体进行了一些推测）。然而，许多有用的随机环境接近于确定性。基于这一动机，Srivastava等人（2019）展示了（回报到达变体的）eUDRL成功解决了许多非确定性环境（例如MuJoCo任务），这些环境的转移核仅表现出较小的非确定性。此后，Ghosh等人（2021）在GCSL（本质上是限制在固定时间步长和状态到达任务的eUDRL）的背景下，进一步展示了令人印象深刻的基准性能。与此同时，Kumar等人（2019）也提出了本质上是eUDRL的算法。

将eUDRL与Transformer架构（Schmidhuber, 1992; Vaswani等人, 2017; Schlag等人, 2021）结合作为策略近似器的想法成功地在Chen等人（2021）提出的**决策变换器（Decision Transformer, DT）**架构中得到展示。然而，Chen等人（2021）仅研究了**离线强化学习（offline RL）**设置（相当于单次eUDRL迭代）。在DT架构提出后不久，Furuta等人（2021）提出了其通用版本，带来了非平凡的性能提升。

Strupl等人（2022a）和Paster等人（2022）描述了eUDRL在具有随机转移动态的环境中收敛的问题和原因。虽然Strupl等人（2022a）重新编写了eUDRL的递归公式以识别收敛问题，但Paster等人（2022）直接在DT的背景下提出了这一问题的解决方案。此后，Yang等人（2023）和Faccio等人（2023）分别在DT和eUDRL的背景下提出了解决随机环境中eUDRL问题的方法。

与本工作更相关的是关于eUDRL及相关算法的理论研究。这里，与本工作最相关的实践和理论工作是Brandfonbrener等人（2022）、Ghosh等人（2021）、Kumar等人（2019）、Andrychowicz等人（2017）和Paster等人（2022）的工作。我们在下面对这些工作进行详细讨论。

- **Brandfonbrener等人（2022）**：该研究在离线强化学习的背景下讨论了eUDRL（及其他相关算法），即仅研究eUDRL的单次迭代，旨在获得样本复杂度的界限。相比之下，我们关注于开发连续性结果和界限，既包括任何有限次迭代的情况，也包括样本数量无限的渐近情况。因此，尽管本工作与Brandfonbrener等人（2022）在某些初步结果上存在重叠（例如，首次迭代的目标到达目标的连续性，如Brandfonbrener等人（2022）的定理1所示），但首次迭代的连续性很容易获得，因为初始策略不依赖于转移核$\lambda$。主要困难在于策略是$\lambda$的函数。这导致在$\lambda_0$（确定性核）处，策略在$\lambda$中从第二次迭代开始就出现不连续性。为了克服这一困难，必须使用某种较弱的连续性概念。在这里，我们使用了相对连续性的概念（见我们在定理16的证明中的归纳步骤）。开发渐近结果是另一个复杂层次（见第7节和第8节）。

- **Ghosh等人（2021）**：Ghosh等人（2021）引入了以下目标到达目标的下界（见Ghosh等人（2021）的定理3.1）：
  $$
  J^\pi \geq J^{\text{GCSL}}(\pi) - 4T(T - 1)\alpha'^2 + C,
  $$
  其中$J^\pi$表示策略$\pi$的目标到达目标，$J^{\text{GCSL}}(\pi)$表示策略$\pi$的GCSL目标（与eUDRL目标（2.7）相同，用于尾
部片段），$T$表示固定时间步长，$\alpha' := \max_{\bar{s}} \text{DTV}(\pi(\cdot|\bar{s})|\pi^{\text{old}}(\cdot|\bar{s}))$表示策略$\pi$与用于收集轨迹的策略$\pi^{\text{old}}$之间的总变差距离，$C$是与策略$\pi$相关的常数。

  当尝试将这一结果应用于我们的工作时，例如评估目标到达目标的连续性（或目标到达目标的$\delta$依赖误差界限），存在（至少）两个问题。第一个问题是，要以有意义的方式使用这一界限，需要最小化$\alpha'$。这可以通过假设GCSL（eUDRL）策略序列存在极限来实现（此时$\alpha' \to 0$）。然而，证明这一点是困难的。为了保持一定的严谨性，我们避免了极限假设，而是研究所有可能的累积点。从这个角度来看，这一结果并不实用，因为在有限次迭代中，$\alpha'$可能很大。它对于渐近情况也不实用，因为我们不能承受极限假设。第二个问题是，没有考虑对$\delta$（转移核$\lambda$与确定性核$\lambda_0$的距离）的依赖。为了使用这一界限（例如，评估在$\lambda_0$处的连续性），需要确定右侧所有三项对$\delta$的依赖关系。这似乎比直接确定$J^\pi$对$\delta$的依赖关系更具挑战性（如我们在本文中所做的），并且可能仍然不够精确（见上述观点）。

- **Kumar等人（2019）**：Kumar等人（2019）提出了一种用于学习回报条件策略的算法，本质上是固定时间步长的eUDRL在尾部片段（Segtrail）上的应用。然而，尽管作者对eUDRL的性质进行了一些理论讨论，但这些讨论并未涉及MDP转移核与确定性核的距离（确定性核的作用完全没有被研究）。因此，关于我们在本文中特别感兴趣的转移核的连续性，这项工作几乎没有提供任何信息。

- **Andrychowicz等人（2017）**：Andrychowicz等人（2017）提出了**后见之明经验回放（Hindsight Experience Replay, HER）**，这是经验回放的流行扩展，已应用于许多离线强化学习算法，如DQN和DDPG。HER在某些方面与eUDRL密切相关。与eUDRL一样，HER涉及目标、目标映射和目标依赖策略（注意，通过将其纳入状态表示中，可以考虑对时间步长/剩余时间步长的依赖）。HER适用于固定时间步长，并且包含折扣因子（也包括无折扣的情况）。HER扩展的原理是不仅将代理实际遇到的轨迹填充到回放缓冲区中，还将其“重标记”版本填充进去，其中原始目标被实际达到的最终状态替换，相应的奖励也进行了替换。然后，HER使用离线算法从回放缓冲区中的数据学习价值函数和策略。

  eUDRL与HER的主要区别在于：（1）HER不仅使用重标记的轨迹，还使用原始轨迹（即具有原始目标的轨迹）；（2）HER并不直接将下一个策略拟合到某些动作条件上（至少当使用DQN或DDPG作为原始HER论文中的离线算法时）。这两个问题是将HER与eUDRL递归公式联系起来的主要障碍。通过使用一个方便的强化学习算法（例如RWR），可以解决第二个问题，但因为大部分数据未被重标记，我们仍然会得到RWR和eUDRL递归公式之间的某种混合。这一观察结果可能已经暗示了HER扩展RWR的一些性质。需要注意的是，HER与eUDRL共享的一些问题已经在文献中被报告、分析和/或解决。我们建议感兴趣的读者参考Lanka和Wu（2018）以及Schramm等人（2023）的工作。

- **Paster等人（2022）**：Paster等人（2022）描述了DT在随机环境中发散的原因，并提出了一个解决方案。为了解决随机领域中的收敛问题，Paster等人（2022）提出不基于回报进行条件化，而是基于统计量$I(\tau)$（其中$\tau$表示轨迹）进行条件化，该统计量与环境的随机性无关。在他们的工作中，$I$由神经网络表示，并通过对抗性方案进行学习。然后根据$I$值对轨迹进行聚类，并将$I$值映射到其对应聚类的平均回报。结果是，通过将策略条件化为$I$值，可以一致地达到相应的平均回报，即在期望中达到。这与基于一般依赖于环境随机性的统计量（例如轨迹回报或eUDRL中的某些一般抽象目标）进行条件化形成对比。关于“与环境随机性无关”的具体含义，请参考Paster等人（2022）的原文。


### 9.4 Transformer架构 (Transformers)

未归一化的线性化自注意力Transformer（也称为ULTRAs）至少早在20世纪90年代就已出现，当时在Schmidhuber（1992, 1991）的作品中被称为**快速权重编程器（Fast-Weight Programmers, FWPs）**（Schlag等人, 2021）。FWPs的目标是通过使用慢速和快速网络处理每个输入来获得比循环网络更高效的存储替代方案，其中慢速网络的输出用于调整快速网络的权重。注意力术语随后由Schmidhuber（1993）引入。现代Transformer架构由Vaswani等人（2017）引入，并且其计算复杂度与输入大小呈二次方关系。最近，一些Vaswani等人（2017）架构的变体重新使用线性化注意力（例如，参见Katharopoulos等人（2020）和Choromanski等人（2020）的工作），因为这样可以实现线性扩展。


## 10 结论 (Conclusion)

我们的贡献在于首次对一些关键的“通过监督学习实现强化学习”的方法的**收敛性和稳定性**进行了严格的理论分析。具体来说，我们的研究既关注了**命令扩展型MDP（CE型MDP）**的数学表述，也关注了**episodic Upside-Down Reinforcement Learning（eUDRL）**算法。这些框架的结合为常见的“通过SL实现RL”的训练方案提供了足够的通用性，包括GCSL、ODT和RWR。我们展示了如何将GCSL、ODT和RWR表示为CE框架的特例，并将其解释为eUDRL的变体。这通过将技术分析限制在特定的片段空间**Segtrail和Segdiag**中实现。本质上，GCSL、ODT和RWR的训练迭代可以被理解为eUDRL在**Seg、Segtrail或Segdiag**上的训练迭代实例，对于某个方便选择的CE。为了适应这些训练方案，我们将收敛性和稳定性的分析扩展到片段空间**Segtrail和Segdiag**，并为**Seg**证明了相关结果。我们的发现大致可以分为三个类别：（1）有限次迭代时训练方案的稳定性分析；（2）训练方案的渐近性质研究；（3）补充讨论针对算法的更广泛和完整的数学图景的结果。



### 10.1 有限次迭代时的稳定性研究 (Investigation of Stability at a Finite Number of Iterations)

在本工作中，我们展示了eUDRL在转移核空间的边界处的不稳定性。具体来说，我们提供了具有非确定性核的环境示例，这些核位于边界上，即使对核进行微小的扰动，也可能导致训练方案生成的值或目标到达目标发生显著且任意的变化。示例1和示例2分别代表非确定性和确定性转移核，说明了在转移核空间的边界处，eUDRL生成的策略在迭代次数$n \geq 2$时存在不可移除的不连续性。相比之下，在核空间的内部，策略和值始终保持连续性。然而，在确定性核处，虽然值是连续的，但策略可能存在不可移除的不连续性。为了解决这一问题，我们引入了**相对连续性**的概念，它对应于一种关于商拓扑的连续性形式。利用这一概念，我们证明了eUDRL策略在任何有限次迭代中在确定性核处都是相对连续的。这意味着目标到达目标的连续性以及eUDRL类训练方案在任何有限次迭代中在确定性核处的稳定性。结合eUDRL在确定性核处收敛到最优性的事实，这表明在接近确定性的转移核处，目标到达目标接近最优。



### 10.2 无限次迭代时的收敛性和稳定性研究 (Convergence and Investigation of Stability at an Infinite Number of Iterations)

在本工作中，我们研究了eUDRL生成的量在迭代次数趋于无穷时的渐近行为。我们建立了eUDRL策略的累积点集合的相对连续性以及相关目标到达目标在确定性核处的连续性，特别是对于两个重要的特殊情况。这些情况由额外的假设特征化，这些假设有助于连续性的分析：

1. **CE的初始分布的支持集包含“关键状态”集合** $\bar{S}_{\lambda_0} \subset \text{supp} \, \bar{\mu}$。简单来说，$\bar{S}_{\lambda_0}$由那些对于证明目标到达目标在给定确定性核$\lambda_0$处的连续性至关重要的状态组成。特别地，CE无法到达的状态被排除在$\bar{S}_{\lambda_0}$之外。

2. **最优策略在$\bar{S}_{\lambda_0}$上是唯一的**，即对于所有状态（$\forall \bar{s} \in \bar{S}_{\lambda_0}$）：$|O(\bar{s})| = 1$。这一条件减轻了研究相对连续性的复杂性，显著简化了稳定性和收敛性分析。

因此，这两个条件都意味着在渐近极限下，接近确定性的核处的目标到达目标表现出接近最优的行为，这与任何有限次迭代时的情况相同。我们将这些结果完全明确化，推导出基于转移核与确定性核距离的最优性误差估计。我们还推导了eUDRL策略误差的界限，并评估了其$q$-线性收敛速率。尽管我们相信所概述的条件涵盖了广泛的实用场景，但关于eUDRL生成策略在确定性核处的累积点集合的相对连续性的完全一般性讨论仍然是一个开放问题。

受“通过SL实现RL”训练方案中广泛使用正则化的启发（例如，在ODT中），我们研究了eUDRL策略迭代的$\epsilon$-贪婪正则化。我们在完全一般性的情况下建立了$\epsilon$-eUDRL生成策略的累积点集合的相对连续性。与之前一样，这涉及到相应目标到达目标的累积点集合的连续性。我们还证明了在渐近极限下，$\epsilon$-eUDRL在接近确定性的核处的目标到达目标表现出接近最优的行为。与无正则化的讨论类似，我们提供了基于转移核与确定性核距离的最优性误差估计。然而，需要注意的是，渐近分析是在完全一般性的情况下进行的，不依赖于前面情况中使用的特殊假设。


### 10.3 进一步的结果 (Further Results)

我们建立了eUDRL生成策略和目标到达目标在任何有限次迭代中对于位于所有核空间内部的转移核的连续性。与确定性核不同，这并不直接意味着接近最优性。我们包含这一结果是为了完善数学图景，并为后续工作提供基础。我们还包含了eUDRL在确定性核处的最优性证明：尽管这一结果可能在文献中被隐含地理解，但我们为了完整性而将其写了出来。数学分析通过一系列详细的示例加以补充，这些示例可能对这一研究领域的持续发展有所帮助。具体来说，我们通过涉及**双臂老虎机（2-armed bandit）**、**$\mathbb{Z}^3$上的随机游走**和**$3 \times 3$网格世界**的示例来说明我们的结果。用于计算示例和生成相关图表的代码可在以下链接获取：[https://github.com/struplm/eUDRL-GCSL-ODT-Convergence-public](https://github.com/struplm/eUDRL-GCSL-ODT-Convergence-public)。


## 致谢与资金支持说明 (Acknowledgments and Disclosure of Funding)

本研究得到了以下机构的支持：欧洲研究委员会（ERC，高级资助编号742870）、瑞士国家超级计算中心（CSCS，项目编号s1090）以及瑞士国家科学基金会（资助编号200021 192356，项目名称NEUSYM）。我们还要感谢NVIDIA公司捐赠的DGX-1设备（作为AI研究先锋奖的一部分）以及IBM捐赠的Minsky机器。

