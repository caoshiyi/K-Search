# GetValue

**页面ID:** atlasascendc_api_07_00103  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00103.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

获取LocalTensor指定索引的数值。

**该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。**

#### 函数原型

```
__aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint32_t index) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| index | 输入 | LocalTensor索引，单位为元素。 |

#### 返回值说明

LocalTensor指定索引的数值，PrimType类型。

PrimType定义如下：

```
// PrimT用于从T中提取基础数据类型：T传入基础数据类型，直接返回数据类型；T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
using PrimType = PrimT<T>;
```

#### 约束说明

无

#### 调用示例

参考调用示例。
