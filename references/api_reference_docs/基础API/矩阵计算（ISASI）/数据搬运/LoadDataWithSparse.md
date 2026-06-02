# LoadDataWithSparse

**页面ID:** atlasascendc_api_07_0244  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0244.html

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

用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。索引矩阵的数据类型为int2，需要拼成int8的数据类型，再传入接口。

索引矩阵在一个int8的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）为一个int8，0 1 2 0（对应索引矩阵后四位0 2 1 0）为一个int8。

索引矩阵的功能说明参考MmadWithSparse。

#### 函数原型

```
template <typename T = int8_t, typename U = uint8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparse(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<U>& idx, const LoadData2dParams& loadDataParam)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | dst、src的数据类型。 |
| U | idx的数据类型。 - 当dst、src、idx为基础数据类型时，T和U必须为uint8_t类型，否则编译失败。 - 当dst、src、idx为TensorTrait类型时，T和U的LiteType必须为int8_t类型，否则编译失败。 最后两个模板参数仅用于上述数据类型检查，用户无需关注。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为LocalTensor，支持的TPosition为B2，LocalTensor的起始地址需要512字节对齐。 支持的数据类型为int8_t。 数据连续排列顺序要求为小N大Z格式。 |
| src | 输入 | 源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。 支持的数据类型为int8_t。 |
| idx | 输入 | 源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。 支持的数据类型为int8_t。 |
| loadDataParam | 输入 | LoadData参数结构体，LoadData2DParams类型，详细说明参考LoadData2DParams结构体内参数说明。 |

#### 约束说明

- repeat=0表示不执行。
- 每次迭代中的startIndex不能小于零。
- 不支持转置功能。

#### 调用示例

详细用例请参考MmadWithSparse。
