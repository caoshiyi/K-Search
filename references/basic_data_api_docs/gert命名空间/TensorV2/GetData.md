# GetData

**页面ID:** atlasopapi_07_00739  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00739.html

---

#### 函数功能

获取Tensor的数据地址。

#### 函数原型

```
template<class T>  const T *GetData() const
template<class T>  auto GetData() -> T*
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| T | 输入 | 数据类型。 |

#### 返回值说明

数据地址。

#### 约束说明

无

#### 调用示例

```
TensorV2 tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape                
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format                              
                kFollowing,                                  // placement                
                ge::DT_FLOAT16,                              //dt                
                nullptr};
auto addr = tensor.GetData<int64_t>(); // reinterpret_cast<int64_t *>(&tensor + 1)
```
