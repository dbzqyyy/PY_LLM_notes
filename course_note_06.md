# 书生·浦语大模型实战营第二期培训总结

## 第六课-概览

Lagent&AgentLego 智能体应用搭建

**大模型局限性**

- 幻觉
- 时效性
- 可靠性

**智能体定义**

- 感知环境动态条件
- 采取动作影响环境
- 运用推理能力采取行动

**智能体组成**

- 大脑
- 感知
- 动作

**经典智能体范式**

![image-20240420163358415](course_note_06.assets/image-20240420163358415.png)

- AutoGPT

![image-20240420163428639](course_note_06.assets/image-20240420163428639.png)

将任务输入系统后，将任务列表发送给对应智能体，将结果存入记忆，并发送另一个智能体分配新的任务，循环直到完成任务。

- ReWoo

![image-20240420163630574](course_note_06.assets/image-20240420163630574.png)

将输入拆分多步发送给worker执行，将2部分输出都发给solver得到结果。

- ReACT

![image-20240420164539337](course_note_06.assets/image-20240420164539337.png)

通过选择工具进行执行，完成后由模型思考是否选用下一个工具，直到达到结束条件。

## Lagent&AgentLego

### Lagent

一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体，
支持多种智能体范式。(如 AutoGPT、ReWoo、ReAct)
支持多种工具。(如谷歌搜索、Python解释器等)

![image-20240420164712049](course_note_06.assets/image-20240420164712049.png)

### AgentLego

一个多模态工具包，旨在像乐高积木，可以快速简便地拓展自定义工具，从而组装出自己的智能体。支持多个智能体框架。(如 Lagent、LangChain、Transformers Agents)

提供大量视觉、多模态领域前沿算法

![image-20240420164824760](course_note_06.assets/image-20240420164824760.png)

### 两者关系

![image-20240420164844791](course_note_06.assets/image-20240420164844791.png)

## 作业

![image-20240422234428886](course_note_06.assets/image-20240422234428886.png)

查询天气的好像网络不行。

![image-20240423001749537](course_note_06.assets/image-20240423001749537.png)

直接使用AgentLego

![image-20240424233500466](course_note_06.assets/image-20240424233500466.png)

AgentLegoWebUI使用

![image-20240424232843685](course_note_06.assets/image-20240424232843685.png)

AgentLego自定义工具

![image-20240425000553333](course_note_06.assets/image-20240425000553333.png)