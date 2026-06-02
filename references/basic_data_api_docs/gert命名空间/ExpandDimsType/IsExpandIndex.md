# IsExpandIndex

**页面ID:** atlasopapi_07_00066  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00066.html

---

#### 函数功能

基于补维后的shape，判断指定的index轴是否为补维轴。

#### 函数原型

```
bool IsExpandIndex(const AxisIndex index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 指定轴的索引。 |

#### 返回值说明

- true代表指定的轴为补维轴。
- false代表指定的轴为原始轴。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType type1("1001");
bool is_expand_index0 = type1.IsExpandIndex(0); // true
bool is_expand_index1 = type1.IsExpandIndex(1); // false
```
