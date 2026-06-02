# AddAutoMappingSubgraphIOIndexFunc

**页面ID:** atlasopapi_07_00297  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00297.html

---

#### 函数功能

注册的具体网络类型的自动映射函数。

#### 函数原型

```
void AddAutoMappingSubgraphIOIndexFunc(domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| framework | 输入 | 网络类型，FrameworkType类型定义请参考FrameworkType。 |
| fun | 输入 | 自动映射输入输出函数，函数类型详见AutoMappingSubgraphIndex。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
