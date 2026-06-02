# GetSubBlockIdx(ISASI)

**页面ID:** atlasascendc_api_07_0281  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0281.html

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

获取AI Core上Vector核的ID。

#### 函数原型

```
__aicore__ inline int64_t GetSubBlockIdx()
```

#### 参数说明

无

#### 返回值说明

返回Vector核ID。

#### 约束说明

无

#### 调用示例

```
int64_t subBlockID = AscendC::GetSubBlockIdx();
```
