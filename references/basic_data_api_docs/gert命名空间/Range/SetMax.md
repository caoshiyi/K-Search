# SetMax

**页面ID:** atlasopapi_07_00133  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00133.html

---

#### 函数功能

设置最大的T对象指针。

#### 函数原型

```
void SetMax(T *max)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| max | 输入 | 最大的T对象指针。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
Range<int> range;
int max = 1024;
range.SetMax(&max);
```
