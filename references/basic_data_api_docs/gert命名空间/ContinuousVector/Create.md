# Create

**页面ID:** atlasopapi_07_00053  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00053.html

---

#### 函数功能

创建一个ContinuousVector实例，ContinuousVector不支持动态扩容。

#### 函数原型

```
template<typename T>  static std::unique_ptr<uint8_t[]> Create(size_t capacity, size_t &total_size)
template<typename T>  static std::unique_ptr<uint8_t[]> Create(const size_t capacity)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| T | 输入 | 实例中包含的元素类型。 |
| capacity | 输入 | 实例的最大容量。 |
| total_size | 输出 | 本实例的总长度。 |

#### 返回值说明

指向本实例的指针。

#### 约束说明

无。

#### 调用示例

```
size_t capacity = 100U;
auto cv_holder = ContinuousVector::Create<int64_t>(capacity); // 创建了一个可以存放100个int64_t数据的内存。
```
