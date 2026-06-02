# SetAlgConfig

**页面ID:** atlasascendc_api_07_10040  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10040.html

---

#### 功能说明

设置通信算法。

#### 函数原型

```
uint32_t SetAlgConfig(const std::string &algConfig)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| algConfig | 输入 | 通信算法配置。string类型，支持的最大长度为128字节。          针对             Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，当前支持的取值为：                     - "AllReduce=level0:doublering"：AllReduce通信任务。           - "AllGather=level0:doublering"：AllGather通信任务。           - "ReduceScatter=level0:doublering"：ReduceScatter通信任务。           - "AlltoAll=level0:fullmesh;level1:pairwise"：AlltoAllV和AlltoAll通信任务。           - "BatchWrite=level0:fullmesh"：BatchWrite通信任务。                    针对             Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，该参数为预留字段，配置后不生效，默认仅支持FullMesh算法。FullMesh算法即NPU之间的全连接，任意两个NPU之间可以直接进行数据收发。详细的算法内容可参见集合通信算法介绍。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

本接口的调用示例请见调用示例。
