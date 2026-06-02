# MutableFormat

**页面ID:** atlasopapi_07_00757  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00757.html

---

#### 函数功能

获取Tensor的数据格式，包含运行时数据格式和原始数据格式。

#### 函数原型

```
StorageFormat &MutableFormat()
```

#### 参数说明

无

#### 返回值说明

Format引用。

关于StorageFormat类型的定义，请参见StorageFormat。

#### 约束说明

无

#### 调用示例

```
TensorV2 tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape              
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format              
                kFollowing,                                  // placement              
                ge::DT_FLOAT16,                              //dt              
                nullptr};
auto fmt = tensor.MutableFormat();
```
