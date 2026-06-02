# InitBuffer

**页面ID:** atlasascendc_api_07_0125  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0125.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

调用TBufPool::InitBuffer接口为TQue/TBuf进行内存分配。

#### 函数原型

```
template <class T> __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
template <TPosition pos> __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 说明 |
| --- | --- |
| T | que参数的类型。 |
| pos | Buffer逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。关于TPosition的具体介绍请参考TPosition。 |

**表2 **InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| que | 输入 | 需要分配内存的TQue对象 |
| num | 输入 | 分配内存块的个数 |
| len | 输入 | 每个内存块的大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐 |

**表3 **InitBuffer(TBuf<pos>& buf, uint32_t len)原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| buf | 输入 | 需要分配内存的TBuf对象 |
| len | 输入 | 为TBuf分配的内存大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐 |

#### 约束说明

声明TBufPool时，可以通过bufIDSize指定可分配Buffer的最大数量，默认上限为4，最大为16。TQue或TBuf的物理内存需要和TBufPool一致。

#### 调用示例

参考InitBufPool
