# SetPlacement

**页面ID:** atlasopapi_07_00761  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00761.html

---

#### 函数功能

设置Tensor的placement。

#### 函数原型

```
void SetPlacement(const TensorPlacement placement)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| placement | 输入 | 需要设置的Tensor的placement。 关于TensorPlacement类型的定义，请参见TensorPlacement。 |

#### 约束说明

无

#### 调用示例

```
TensorV2 tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape              
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format              
                kFollowing,                                  // placement              
                ge::DT_FLOAT16,                              //dt              
                nullptr};
tensor.SetPlacement(TensorPlacement::kOnHost);
auto placement = tensor.GetPlacement();
```
