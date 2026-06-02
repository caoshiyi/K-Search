# MutableData

**页面ID:** atlasopapi_07_00271  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00271.html

---

#### 函数功能

获取首个元素的指针地址，MutableData(),  reinterpret_cast<T *>(MutableData())  + GetSize()) 中的数据即为当前容器中保存的数据。

与[GetData接口相比，指针地址指向的内容可以修改。

#### 函数原型

**T *MutableData()**

#### 参数说明

无。

#### 返回值说明

首个元素的指针地址。

#### 约束说明

无。

#### 调用示例

```
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity);
auto cv = reinterpret_cast<TypedContinuousVector *>(cv_holder.get());
auto cap = cv->MutableData<int64_t>();
```
