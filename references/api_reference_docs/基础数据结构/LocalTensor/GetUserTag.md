# GetUserTag

**页面ID:** atlasascendc_api_07_00109  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00109.html

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

获取指定Tensor块的Tag信息，用户可以根据Tag信息对Tensor进行不同操作。

#### 函数原型

```
__aicore__ inline TTagType GetUserTag() const
```

#### 参数说明

无

#### 返回值说明

指定Tensor块的Tag信息。TTagType定义如下：

```
using TTagType = int32_t
```

#### 约束说明

无

#### 调用示例

参考调用示例。
