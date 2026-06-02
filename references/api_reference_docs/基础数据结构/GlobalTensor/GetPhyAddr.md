# GetPhyAddr

**页面ID:** atlasascendc_api_07_00025  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00025.html

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

获取全局数据的地址。

#### 函数原型

- 获取全局数据的地址

```
__aicore__ inline const __gm__ PrimType* GetPhyAddr() const
```

- 获取全局数据（指定偏移offset个元素）的地址

```
__aicore__ inline __gm__ PrimType* GetPhyAddr(const uint64_t offset) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 偏移的元素个数，用于指定数据的位置。 |

#### 返回值说明

全局数据的地址。

#### 约束说明

无。
