# Stage

**页面ID:** atlasgeapi_07_0152  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasgeapi_07_0152.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

设置自定义Pass执行阶段。

#### 函数原型

```
PassRegistrationData &Stage(const CustomPassStage stage)
```

#### 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| 表示自定义Pass执行阶段。                     - kBeforeInferShape：（默认值）自定义Pass在框架入口处InferShape之前执行。           - kAfterInferShape：自定义Pass在InferShape之后执行。             如果自定义Pass在InferShape之后执行，Pass中需要保证改图之后shape的连续性，可以通过InferShapeAndType接口保证：                                                                                                                                             ``` // 1. 获取输入节点node1的输出描述     TensorDesc output_desc;     node1.GetOutputDesc(0, output_desc);     // 2. 更新当前节点node2的输入描述     node2.UpdateInputDesc(0, output_desc);     // 3. 当前节点node2推导InferShape     operator2.InferShapeAndType(); ```                                                                                                    调用InferShape函数时，InferShape之前会将输入的original shape刷入到算子的shape上，InferShape之后会将算子的输出shape刷入到算子输出的original shape上，因此当为一个算子设置InputDesc时，需要设置original shape。           - kAfterAssignLogicStream：自定义Pass在逻辑流分配阶段之后执行。该阶段仅接收逻辑流分配的Pass（注册自定义的逻辑流分配Pass执行函数请参见CustomAllocateStreamPassFn），由于该阶段不允许改图，其他场景的改图Pass会校验报错。           - kAfterBuiltinFusionPass：自定义Pass在内置的原图融合Pass之后执行。 |  |  |

#### 返回值说明

返回PassRegistrationData对象。

#### 约束说明

无
