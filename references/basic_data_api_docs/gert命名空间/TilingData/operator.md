# operator<<

**页面ID:** atlasopapi_07_00263  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00263.html

---

#### 函数功能

使用<<运算符重载的方式，实现向后添加tiling data的功能。若添加超过可容纳的最大长度，则忽略本次操作。

使用<<添加tiling data，可以实现和Append相同的功能，使用该运算符更加直观方便。

#### 函数原型

**template<typename T>**

**TilingData &operator<<(TilingData &out, const T &data)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| T | 输入 | 添加的tiling data的类型。 |
| out | 输出 | TilingData类实例。 |
| data | 输入 | 添加的tiling data的实例。 |

#### 返回值说明

追加完data的TilingData对象。

#### 约束说明

无。

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());

struct AppendData{
  int a = 10;
  int b = 100;
};
AppendData ad;
(*td) << ad;
auto data_size = td.GetDataSize(); // 2 * sizeof(int)
```
