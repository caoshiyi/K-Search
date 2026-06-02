# Extract

**页面ID:** atlasascendc_api_07_0841  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0841.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

处理Sort的结果数据，输出排序后的value和index。

#### 函数原型

```
template <typename T>
__aicore__ inline void Extract(const LocalTensor<T> &dstValue, const LocalTensor<uint32_t> &dstIndex, const LocalTensor<T> &sorted, const int32_t repeatTime)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| dstValue | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 |
| dstIndex | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 此源操作数固定为uint32_t数据类型。 |
| sorted | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。 |
| repeatTime | 输入 | 重复迭代次数，int32_t类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，每次迭代处理64个float类型或128个half类型元素。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，每次迭代处理64个float类型或128个half类型元素。 Atlas 推理系列产品AI Core，每次迭代完成16个Region Proposals的元素抽取并排布到16个元素里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。 取值范围：repeatTime∈[0,255]。 |

#### 约束说明

#### 调用示例

请参见MrgSort的调用示例。
