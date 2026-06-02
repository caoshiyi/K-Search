# GetAddr

**页面ID:** atlasopapi_07_00741  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00741.html

---

#### 函数功能

获取Tensor的数据地址。

#### 函数原型

```
const void *GetAddr() const
void *GetAddr()
```

#### 参数说明

无

#### 返回值说明

返回数据地址。

#### 约束说明

无

#### 调用示例

```
TensorV2 tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape                
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format                              
                kFollowing,                                  // placement                
                ge::DT_FLOAT16,                              //dt                
                nullptr};
auto addr = tensor.GetAddr(); // &tensor + 1
```
