# GetMax

**页面ID:** atlasopapi_07_00135  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00135.html

---

#### 函数功能

获取最大的T对象指针。

#### 函数原型

```
const T *GetMax() const
T *GetMax()
```

#### 参数说明

无。

#### 返回值说明

返回最大的T对象指针。

#### 约束说明

无。

#### 调用示例

```
int min = -1;
int max = 1024;
Range<int> range(&min, &max);

auto ret = range.GetMax(); // ret指针指向max
```
