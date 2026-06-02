# GetExpandDimsType

**页面ID:** atlasopapi_07_00217  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00217.html

---

#### 函数功能

获取shape的补维规则。

#### 函数原型

**ExpandDimsType GetExpandDimsType() const**

#### 参数说明

无。

#### 返回值说明

返回shape的补维规则。

关于ExpandDimsType类型的定义，请参见ExpandDimsType。

#### 约束说明

无。

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape              
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format              
              kFollowing,                                  // placement              
              ge::DT_FLOAT16,                              //dt              
              nullptr};
auto expand_dims_type = tensor.GetExpandDimsType();   // ExpandDimsType{}
```
