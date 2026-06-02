# Concat

**页面ID:** atlasascendc_api_07_0839  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0839.html

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

对数据进行预处理，将要排序的源操作数src一一对应的合入目标数据concat中，数据预处理完后，可以进行Sort。

#### 函数原型

```
template <typename T>
__aicore__ inline void Concat(LocalTensor<T> &concat, const LocalTensor<T> &src, const LocalTensor<T> &tmp, const int32_t repeatTime)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| concat | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。 |
| tmp | 输入 | 临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供，临时空间大小的获取方式请参考GetConcatTmpSize。数据类型与源操作数保持一致。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 |
| repeatTime | 输入 | 重复迭代次数，int32_t类型，每次迭代处理16个元素，下次迭代跳至相邻的下一组16个元素。取值范围：repeatTime∈[0,255]。 |

#### 约束说明

#### 调用示例

请参见MrgSort的调用示例。
