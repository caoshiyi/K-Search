# Init

**页面ID:** atlasopapi_07_00054  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00054.html

---

#### 函数功能

使用最大容量初始化本实例。

#### 函数原型

```
void Init(const size_t capacity)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| capacity | 输入 | 实例的最大容量。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
size_t capacity = 100U;
size_t total_size = capacity * sizeof(int64_t) + sizeof(ContinuousVector);
auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
reinterpret_cast<ContinuousVector *>(holder.get())->Init(capacity); // 100U
```
