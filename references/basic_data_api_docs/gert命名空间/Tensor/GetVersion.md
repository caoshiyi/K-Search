# GetVersion

**页面ID:** atlasopapi_07_00769  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00769.html

---

#### 函数功能

获取Tensor中的版本。kTensorV1：不携带view和offset信息，kTensorV2：携带view和offset信息

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

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
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape                
               {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format                              
               kFollowing,                                  // placement                
               ge::DT_FLOAT16,                              //dt                
               nullptr};
auto version = tensor.GetVersion(); // kTensorV1
```
