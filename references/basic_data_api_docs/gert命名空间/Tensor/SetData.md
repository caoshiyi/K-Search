# SetData

**页面ID:** atlasopapi_07_00199  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00199.html

---

#### 函数功能

设置Tensor的数据。

#### 函数原型

```
void SetData(TensorData &&data)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data | 输入 | 需要设置的数据。 关于TensorData类型的定义，请参见TensorData。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
Tensor t{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape                
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format                
              kOnHost,                                // placement                
              ge::DT_FLOAT16,                              //dt                
              nullptr};
void *a = &t;
TensorData td(a, nullptr);
t.SetData(std::move(td));
```
