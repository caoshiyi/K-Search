# Reset

**页面ID:** atlasascendc_api_07_0126  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0126.html

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

在切换TBufPool资源池时使用，结束当前TbufPool资源池正在处理的相关事件。调用后当前资源池及资源池分配的Buffer仍然存在，只是Buffer内容可能会被改写。可以切换回该资源池后，重新开始使用该Buffer，无需再次分配。

#### 函数原型

```
__aicore__ inline void Reset()
```

#### 约束说明

无

#### 调用示例

参考InitBufPool
