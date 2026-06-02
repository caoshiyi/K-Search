# CalcTotalSize

**页面ID:** atlasopapi_07_00260  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00260.html

---

#### 函数功能

通过最大容量计算TilingData实例所占用的内存空间。

#### 函数原型

**static ge::graphStatus CalcTotalSize(const size_t cap_size, size_t &total_size)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| cap_size | 输入 | 最大容量，单位为字节。 |
| total_size | 输出 | 内存空间，单位为字节。 |

#### 返回值说明

- 成功返回：ge::GRAPH_SUCCESS。
- 失败返回：ge::GRAPH_FAILED。

#### 约束说明

无。

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
size_t total_size = 0U;
auto ret = td->CalcTotalSize(td.GetCapacity, total_size); // total_size = 100 + sizeof(TilingData)
```
