# SetInputStartPosition

**页面ID:** atlasascendc_api_07_10076  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10076.html

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

设置单核上特征矩阵Input载入数据的起始位置。

#### 函数原型

```
__aicore__ inline void SetInputStartPosition(int64_t diStartPos, int64_t mStartPos)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| diStartPos | 输入 | 单核上Din方向起始位置。 |
| mStartPos | 输入 | 单核上M方向起始位置。 |

#### 约束说明

无

#### 调用示例

```
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
```
