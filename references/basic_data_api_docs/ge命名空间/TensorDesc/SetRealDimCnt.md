# SetRealDimCnt

**页面ID:** atlasopapi_07_00451  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00451.html

---

#### 函数功能

向TensorDesc中设置Tensor的实际维度数目。

通过GetShape接口返回的Shape的维度可能存在补1的场景，因此可以通过该接口设置Shape的实际维度个数。

#### 函数原型

```
void SetRealDimCnt(const int64_t real_dim_cnt)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| real_dim_cnt | 输入 | 需设置的TensorDesc的实际数据维度数目信息。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
