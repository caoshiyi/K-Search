# GetStoreAtomicConfig(ISASI)

**页面ID:** atlasascendc_api_07_0287  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0287.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取原子操作使能位与原子操作类型的值，详细说明见表1。

#### 函数原型

```
__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| atomicType | 输出 | 原子操作使能位。 0：无原子操作 1：使能原子操作，进行原子操作的数据类型为float 2：使能原子操作，进行原子操作的数据类型为half 3：使能原子操作，进行原子操作的数据类型为int16_t 4：使能原子操作，进行原子操作的数据类型为int32_t 5：使能原子操作，进行原子操作的数据类型为int8_t 6：使能原子操作，进行原子操作的数据类型为bfloat16_t |
| atomicOp | 输出 | 原子操作类型。 0：求和操作 |

#### 约束说明

此接口需要与SetStoreAtomicConfig(ISASI)配合使用，用以获取原子操作使能位与原子操作类型的值。

#### 调用示例

```
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
uint16_t type = 0;       // 原子操作使能位
uint16_t op = 0;         // 原子操作类型
AscendC::GetStoreAtomicConfig(type, op);
```
