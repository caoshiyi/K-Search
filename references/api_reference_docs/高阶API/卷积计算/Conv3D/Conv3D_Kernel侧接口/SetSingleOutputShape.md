# SetSingleOutputShape

**页面ID:** atlasascendc_api_07_10075  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10075.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置单核上结果矩阵Output的形状。

Conv3D高阶API目前支持M合轴模式的输出方式。在M合轴模式下，Conv3D API内部将Wout和Hout视为同一轴处理，输出时先沿Wout方向输出，完成一整行Wout输出后，再进行下一行的Wout输出。

**图1 **M合轴模式示意图
<!-- img2text -->
```
                           Wout
                            ↑
                            │
           ┌──────────────────────────────────────┐
           │                                      │
           │  ───────────────────────────────→    │
           │                                      │
           │  ←───────────────────────────────    │
           │                                      │
           │  ───────────────────────────────→    │
 Hout      │                                      │   ⎫ 第一次输出块
           │  ←───────────────────────────────    │   ⎭
           │                                      │
           │  ───────────────────────────────→    │   ⎫ 第二次输出块
           │                                      │   ⎭
           │  ←───────────────────────────────    │
           │                                      │
           └──────────────────────────────────────┘

                           M合轴模式
```

#### 函数原型

```
__aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleDo, uint64_t singleM)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCo | 输入 | 单核上Output的C维度大小。 |
| singleDo | 输入 | 单核上Output的D维度大小。 |
| singleM | 输入 | 单核上Output的M维度大小，即H维度大小与W维度大小的乘积。 |

#### 约束说明

本接口当前仅支持设置Output的C维度、D维度和M维度（即H轴、W轴合并后的维度），不支持设置原始Output的大小。

#### 调用示例

```
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
```
