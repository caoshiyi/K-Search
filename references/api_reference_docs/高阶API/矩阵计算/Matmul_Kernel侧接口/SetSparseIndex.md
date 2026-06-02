# SetSparseIndex

**页面ID:** atlasascendc_api_07_10112  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10112.html

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

设置稀疏矩阵稠密化过程生成的索引矩阵。

索引矩阵在稠密化中的作用请参考MmadWithSparse。

#### 函数原型

```
__aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| indexGlobal | 输入 | 索引矩阵在Global Memory上的首地址，类型为GlobalTensor。 索引矩阵的数据类型为int2，需要由用户拼成int8的数据类型，再传入本接口。索引矩阵的Format格式只支持NZ格式。 |

#### 约束说明

- 索引矩阵的Format格式要求为NZ格式。
- 本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。

#### 调用示例

```
#define ASCENDC_CUBE_ONLY // 使能纯Cube模式(只有矩阵计算)
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetSparseIndex(gm_index); //设置索引矩阵
mm.SetBias(gm_bias);  
mm.IterateAll(gm_c);
```
