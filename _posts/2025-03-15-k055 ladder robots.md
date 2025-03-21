---
layout: mypost
title: k055 Robust Ladder Climbing with a Quadrupedal Robot
categories: [具身智能]
---

# 论文重点难点讲解

## 论文重点

`1.` **研究目标**

   - 使四足机器人能够可靠地攀爬工业环境中的梯子，以扩展其在危险或难以到达区域的检查任务能力。

   - 提出一种基于强化学习（RL）的控制策略和钩状末端执行器设计，以实现快速、鲁棒且通用的梯子攀爬能力。

`2.` **主要贡献**

   - 提出了一种基于RL的控制策略，能够实现四足机器人在不同梯子配置下的鲁棒攀爬。

   - 设计了一种钩状末端执行器，通过其形状优势，为机器人提供稳定的锚定力，使其能够在陡峭梯子上攀爬并抵抗外部干扰。

   - 在仿真中验证了该方法在多种梯子参数（长度、宽度、间距、横档半径、倾角）下的有效性，并在真实硬件上实现了零样本迁移（zero-shot transfer），成功率达到90%。

   - 实现了比现有技术快232倍的攀爬速度，展示了该方法在实际应用中的高效性和鲁棒性。

`3.` **实验结果**

   - **仿真结果**：机器人在仿真中对不同梯子配置的平均攀爬成功率为96%，即使在存在噪声和外部干扰的情况下也能保持较高的成功率。

   - **真实世界结果**：在真实硬件上，机器人在70°和80°的梯子倾角下均实现了100%的成功率，在90°倾角下成功率为66.7%。平均攀爬速度为0.44 m/s，远高于现有技术。

   - **抗干扰能力**：机器人能够应对未建模的干扰（如拉扯机器人腿部的绳子），通过钩状末端执行器生成张力力，保持稳定。

`4.` **未来工作方向**

   - 实现四足机器人上下梯子的能力。

   - 将深度相机图像等感知模态集成到训练流程中，以减少对运动捕捉系统的依赖，使其能够在实际工业环境中应用。

   - 探索机器人形态和控制策略的共同优化，以实现更复杂的任务。

### 论文难点

`1.` **梯子攀爬的复杂性**

   - 梯子攀爬需要机器人具备全身协调能力，以在高倾角下稳定重心并向上移动。这与传统四足机器人的运动模式有很大不同，需要新的控制策略。

   - 梯子的参数（如长度、宽度、间距、横档形状和倾角）变化较大，要求控制策略具有很强的泛化能力。

`2.` **强化学习与硬件的结合**

   - 将基于强化学习的控制策略从仿真迁移到真实硬件上是一个挑战。仿真环境中无法完全模拟真实世界的复杂性和不确定性，例如机器人与梯子之间的摩擦力、外部干扰等。

   - 钩状末端执行器的设计需要与控制策略紧密结合，以确保在真实环境中能够提供足够的锚定力和稳定性。

`3.` **鲁棒性和速度的平衡**

   - 在实现快速攀爬的同时，还需要保证机器人在面对未建模干扰时的鲁棒性。这需要在控制策略中同时考虑速度和稳定性，避免在高速攀爬时因干扰导致失败。

`4.` **感知和控制的集成**

   - 当前方法依赖于运动捕捉系统来获取梯子的精确姿态和位置信息。未来需要将深度相机等感知模态集成到控制策略中，以减少对高精度外部感知系统的依赖，使其能够在更复杂的实际环境中应用。

# 论文讲解

## 1 研究背景与动机

四足机器人因其在复杂地形上的高机动性和稳定性，逐渐成为工业环境中执行危险或难以到达区域检查任务的理想选择。然而，四足机器人在攀爬梯子这一常见任务上仍面临挑战。攀爬梯子不仅需要机器人具备全身协调能力以稳定重心，还需要适应梯子的多种参数（如长度、宽度、横档间距、横档形状、倾角等）。此外，梯子攀爬的鲁棒性对于减少职业风险和提高工业效率至关重要。本文旨在通过强化学习和钩状末端执行器的设计，使四足机器人能够快速、鲁棒地攀爬不同配置的梯子。

## 2 方法

### 2.1 控制策略

#### 2.1.1 教师策略（Teacher Policy）

教师策略的目标是利用无噪声的本体感知信息和特权状态信息（privileged state information）训练一个高性能的控制策略。教师策略的训练包括以下几个关键部分：

- **网络架构**：教师策略由四个多层感知机（MLP）组成，每个MLP使用`elu`激活函数。这些MLP分别对高度扫描图、特权状态信息和惯性测量单元（IMU）信息进行编码，然后将这些编码后的信息与本体感知信息拼接，最终通过一个较大的MLP生成控制动作。

- **训练算法**：教师策略使用IPO（Interior-point Policy Optimization）算法进行训练，该算法通过约束优化确保关节的物理限制（如位置、速度、扭矩）。

- **奖励函数**：奖励函数综合了多个方面，包括位置跟踪、姿态跟踪、关节限制、动作平滑性等。

- **训练环境**：教师策略在仿真环境中训练，仿真环境包括两种地形：复杂的自然地形（如盒子和斜坡）和不同参数的梯子（长度、宽度、横档间距、横档半径、倾角等）。训练过程中，梯子的参数会随着训练进度动态调整，以增加难度。

- **终止条件**：如果机器人基座的俯仰角或滚转角超过$100^\circ$，则终止当前训练回合。

#### 2.1.2 学生策略（Student Policy）

学生策略的目标是将教师策略的性能迁移到真实机器人上，同时仅依赖于噪声化的本体感知信息和梯子状态估计。学生策略的训练包括以下几个关键部分：

- **网络架构**：学生策略的网络架构与教师策略类似，但在输入部分仅包含噪声化的本体感知信息、IMU历史信息和梯子状态估计。学生策略通过一个循环信念编码器（belief encoder）重构教师策略的特权状态信息。

- **训练方法**：学生策略的训练采用行为克隆（behavior cloning）的方式，通过最小化教师策略和学生策略的动作差异来优化学生策略。同时，学生策略还需要通过重构损失（reconstruction loss）来学习教师策略的特权状态信息。

- **训练环境**：学生策略在与教师策略相同的仿真环境中训练，但输入信息中加入了噪声，以模拟真实世界中的不确定性。

### 2.2 钩状末端执行器设计

钩状末端执行器是实现鲁棒攀爬的关键部件，其设计特点如下：

- **结构设计**：钩状末端执行器由铝制核心和3D打印外壳组成，钩子的凹面设计使其能够稳定地附着在圆形横档上，同时钩子的突出部分允许机器人在横档上施加推力和拉力。

- **功能优势**：钩状末端执行器能够在攀爬过程中为机器人提供稳定的锚定力，使其能够在陡峭梯子上保持稳定，并抵抗外部干扰。

### 2.3 仿真与训练

- **仿真环境**：所有仿真均在LeggedGym中进行，训练时使用4096个并行环境，教师策略训练15000个周期，学生策略训练5000个周期。

- **训练过程**：训练过程中，梯子的参数（长度、宽度、横档间距、横档半径、倾角等）会随机生成，并随着训练进度动态调整。此外，训练过程中还会加入随机外部干扰（如基座上的随机力和力矩）和噪声，以增强策略的鲁棒性。

## 3 实验与结果

### 3.1 仿真结果

- **性能评估**：学生策略在仿真环境中对不同梯子配置的平均攀爬成功率为96%，即使在存在噪声和外部干扰的情况下也能保持较高的成功率。

- **钩状末端执行器的优势**：与传统的球形足相比，钩状末端执行器在陡峭梯子上的攀爬性能显著提升，尤其是在大倾角和大横档半径的情况下。

- **噪声和干扰下的表现**：在仿真中，学生策略能够在存在噪声和外部干扰（如基座上的随机力）的情况下成功攀爬梯子，展示了策略的鲁棒性。

### 3.2 真实世界结果

#### 3.2.1 零样本迁移（Zero-shot Transfer）

将训练好的学生策略直接部署到ANYmal D机器人上，无需进一步微调。实验结果如下：

- **不同倾角下的表现**：

  - 在$70^\circ$和$80^\circ$的梯子倾角下，机器人实现了100%的成功率。

  - 在$90^\circ$倾角下，成功率为66.7%。部分测试失败的原因是机器人基座与横档之间的未建模碰撞，导致机器人无法继续攀爬。

- **平均攀爬速度**：机器人平均攀爬速度为0.44 m/s，比现有技术快232倍。

#### 3.2.2 抗干扰能力

在攀爬过程中，通过拉扯机器人腿部的绳子进行干扰测试，机器人能够通过钩状末端执行器生成的张力力保持稳定，并展示出恢复和重试行为。

## 4 结论

本文通过设计钩状末端执行器和基于强化学习的控制策略，实现了四足机器人在不同配置梯子上的快速、鲁棒攀爬。仿真和真实世界实验验证了该方法的有效性。钩状末端执行器为机器人提供了稳定的锚定力，使其能够在陡峭梯子上攀爬并抵抗外部干扰。未来的工作将集中在实现机器人上下梯子的能力，以及将深度相机等感知模态集成到训练流程中，以减少对运动捕捉系统的依赖，进一步提升该方法在实际工业环境中的适用性。

### 论文方法部分详细讲解

## 1 控制策略

### 1.1 教师策略（Teacher Policy）

教师策略的目标是利用无噪声的本体感知信息和特权状态信息（privileged state information）训练一个高性能的控制策略。其主要组成部分如下：

- **网络架构**：

  教师策略由四个多层感知机（MLP）组成，每个MLP使用`elu`激活函数。这些MLP分别对以下输入进行编码：

  - 高度扫描图（height scan）

  - 特权状态信息（privileged state information）

  - 惯性测量单元（IMU）历史信息

  编码后的信息与本体感知信息（proprioceptive observations）拼接，最终通过一个较大的MLP生成控制动作。

- **训练算法**：

  教师策略使用IPO（Interior-point Policy Optimization）算法进行训练，该算法通过约束优化确保关节的物理限制（如位置、速度、扭矩）。

- **奖励函数**：

  奖励函数综合了多个方面，包括位置跟踪、姿态跟踪、关节限制、动作平滑性等。

- **训练环境**：

  教师策略在仿真环境中训练，仿真环境包括两种地形：

  - 复杂的自然地形（如盒子和斜坡）

  - 不同参数的梯子（长度、宽度、横档间距、横档半径、倾角等）

  训练过程中，梯子的参数会随着训练进度动态调整，以增加难度。

- **终止条件**：

  如果机器人基座的俯仰角或滚转角超过$100^\circ$，则终止当前训练回合。

### 1.2 学生策略（Student Policy）

学生策略的目标是将教师策略的性能迁移到真实机器人上，同时仅依赖于噪声化的本体感知信息和梯子状态估计。其主要组成部分如下：

- **网络架构**：

  学生策略的网络架构与教师策略类似，但在输入部分仅包含噪声化的本体感知信息、IMU历史信息和梯子状态估计。学生策略通过一个循环信念编码器（belief encoder）重构教师策略的特权状态信息。

- **训练方法**：

  学生策略的训练采用行为克隆（behavior cloning）的方式，通过最小化教师策略和学生策略的动作差异来优化学生策略。同时，学生策略还需要通过重构损失（reconstruction loss）来学习教师策略的特权状态信息。

- **训练环境**：

  学生策略在与教师策略相同的仿真环境中训练，但输入信息中加入了噪声，以模拟真实世界中的不确定性。

## 2 钩状末端执行器设计

钩状末端执行器是实现鲁棒攀爬的关键部件，其设计特点如下：

- **结构设计**：

  钩状末端执行器由铝制核心和3D打印外壳组成，钩子的凹面设计使其能够稳定地附着在圆形横档上，同时钩子的突出部分允许机器人在横档上施加推力和拉力。

- **功能优势**：

  钩状末端执行器能够在攀爬过程中为机器人提供稳定的锚定力，使其能够在陡峭梯子上保持稳定，并抵抗外部干扰。

## 3 仿真与训练

### 3.1 仿真环境

所有仿真均在LeggedGym中进行，训练时使用4096个并行环境，教师策略训练15000个周期，学生策略训练5000个周期。

### 3.2 训练过程

训练过程中，梯子的参数（长度、宽度、横档间距、横档半径、倾角等）会随机生成，并随着训练进度动态调整。此外，训练过程中还会加入随机外部干扰（如基座上的随机力和力矩）和噪声，以增强策略的鲁棒性。



