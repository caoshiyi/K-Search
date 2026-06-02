# SetTilingKey

**页面ID:** atlasopapi_07_00234  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00234.html

---

#### 函数功能

设置TilingKey。

不同的kernel实现分支可以通过TilingKey来标识，host侧设置TilingKey后，可以选择对应的分支。例如，一个算子在不同的shape下，有不同的算法逻辑，kernel侧可以通过TilingKey来选择不同的算法逻辑，在host侧Tiling算法也有差异，host/kernel侧通过相同的TilingKey进行关联。

#### 函数原型

**ge::graphStatus SetTilingKey(const uint64_t tiling_key)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_key | 输入 | 需要设置的tiling key。 |

#### 返回值说明

成功时返回“ge::GRAPH_SUCCESS”。

关于graphStatus定义，请参见ge::graphStatus。

#### 约束说明

tiling_key的取值范围在uint64_t数据类型范围内，但不可以取值为UINT64_MAX。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ret = context->SetTilingKey(20);
  // ...
}
```
