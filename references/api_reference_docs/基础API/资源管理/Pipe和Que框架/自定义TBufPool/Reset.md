# Reset

**页面ID:** atlasascendc_api_07_0129  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0129.html

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

完成自定义TbufPool资源的释放与eventId等变量的复位对消。

#### 函数原型

```
__aicore__ inline void Reset()
```

#### 约束说明

切换自定义TBufPool资源池时调用该接口，调用后对应资源池及资源池分配的Buffer不能继续使用。

#### 调用示例

请参考调用示例。
