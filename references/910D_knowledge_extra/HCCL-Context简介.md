# HCCL Context简介<a name="ZH-CN_TOPIC_0000002523343574"></a>

本章节的接口用于在kernel侧设置/获取通算融合算子每个通信域对应的context（消息区）地址。需要同步在host侧调用[HcclGroup](HcclGroup.md)接口配置通信域名称后，才可以调用[GetHcclContext](GetHcclContext.md)获取对应的context地址。

> **说明：** 
>本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

