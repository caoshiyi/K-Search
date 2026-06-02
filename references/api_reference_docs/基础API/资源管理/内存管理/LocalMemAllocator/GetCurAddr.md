# GetCurAddr

**页面ID:** atlasascendc_api_07_00098  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00098.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

返回当前物理位置空闲的起始地址。

#### 函数原型

```
template <Hardware hard>
__aicore__ inline uint32_t LocalMemAllocator<hard>::GetCurAddr() const
```

#### 参数说明

无

#### 返回值说明

当前物理位置空闲的起始地址，范围为[0，物理内存最大值)。

#### 约束说明

无

#### 调用示例

```
LocalMemAllocator allocator;
// 默认的物理位置为UB，由于从0地址开始分配，下面的打印结果为0
AscendC::printf("current addr is %u\n", allocator.GetCurAddr());
```
