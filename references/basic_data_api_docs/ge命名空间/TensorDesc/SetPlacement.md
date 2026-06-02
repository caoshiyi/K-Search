# SetPlacement

**页面ID:** atlasopapi_07_00450  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00450.html

---

#### 函数功能

设置Tensor的数据存放的位置。

#### 函数原型

```
void SetPlacement(Placement placement)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 需设置的数据地址的值。 枚举值定义如下： ``` enum Placement {   kPlacementHost = 0,   // host data addr   kPlacementDevice = 1, // device data addr }; ``` |  |  |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
