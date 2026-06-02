# Init

**页面ID:** atlasopapi_07_00046  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00046.html

---

#### 函数功能

初始化ContinuousVectorVector类。

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
size_t total_length = 1000U; // 需根据实际存放的数据量进行设置 
size_t capacity = 100U;
std::vector<uint8_t> buf(total_length);
auto cvv = new (buf.data()) ContinuousVectorVector();
cvv->Init(capacity);
```
