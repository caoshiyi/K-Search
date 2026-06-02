# SetExpandDimsType

**页面ID:** atlasopapi_07_00218  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00218.html

---

#### 函数功能

设置shape的补维规则。

#### 函数原型

**void SetExpandDimsType(const ExpandDimsType &expand_dims_type)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| expand_dims_type | 输入 | 需要设置的补维规则。 关于ExpandDimsType类型的定义，可参见ExpandDimsType。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape              
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format              
              kFollowing,                                  // placement              
              ge::DT_FLOAT16,                              //dt              
              nullptr};
ExpandDimsType type("1001");
tensor.SetExpandDimsType(type);
auto expand_dims_type = tensor.GetExpandDimsType();   // type
```
