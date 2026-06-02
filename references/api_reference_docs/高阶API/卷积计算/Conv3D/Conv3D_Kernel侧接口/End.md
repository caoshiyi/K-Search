# End

**页面ID:** atlasascendc_api_07_10078  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10078.html

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

在Conv3D计算完成后，必须调用一次End函数，以清除EventID并释放内部申请的临时内存。

#### 函数原型

```
void End()
```

#### 参数说明

无

#### 约束说明

End接口必须在IterateAll接口后调用。

#### 调用示例

```
conv3dApi.End();
```
