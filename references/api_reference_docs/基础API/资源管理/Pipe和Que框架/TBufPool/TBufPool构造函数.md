# TBufPool构造函数

**页面ID:** atlasascendc_api_07_0123  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0123.html

---

#### 功能说明

创建TBufPool对象时，初始化数据成员。

#### 函数原型

```
template <TPosition pos, uint32_t bufIDSize = defaultBufIDSize>
__aicore__ inline TBufPool();
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 说明 |
| --- | --- |
| pos | TBufPool逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。关于TPosition的具体介绍请参考TPosition。 |
| bufIDSize | TBufPool可分配Buffer数量，默认为4，不超过16。对于非共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的bufIDSize不能超过原TBufPool剩余可用的Buffer数量；对于共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的bufIDSize不能超过原TBufPool设置的Buffer数量。 |

#### 约束说明

无。
