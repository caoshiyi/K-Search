# GetVersion

**页面ID:** atlasopapi_07_00764  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00764.html

---

#### 函数功能

获取Tensor中的version。

#### 函数原型

```
TensorVersion GetVersion () const
```

#### 参数说明

无

#### 返回值说明

获取Tensor的version。

关于TensorVersion类型的定义，请参见TensorVersion。

#### 约束说明

无

#### 调用示例

```
TensorV2 tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape                
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format                              
                kFollowing,                                  // placement                
                ge::DT_FLOAT16,                              //dt                
                nullptr};
auto version = tensor.GetVersion(); // kTensorV2
```
