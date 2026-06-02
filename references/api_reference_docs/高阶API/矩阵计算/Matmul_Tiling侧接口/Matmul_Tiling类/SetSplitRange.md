# SetSplitRange

**页面ID:** atlasascendc_api_07_0687  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0687.html

---

#### 功能说明

设置baseM/baseN/baseK的最大值和最小值。 目前Tiling暂时不支持该功能。

#### 函数原型

```
int32_t SetSplitRange(int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1, int32_t minBaseM = -1, int32_t minBaseN = -1, int32_t minBaseK = -1)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| maxBaseM | 输入 | 设置最大的baseM值，默认值为-1。-1表示不设置指定的baseM最大值，该值由Tiling函数自行计算。 |
| maxBaseN | 输入 | 设置最大的baseN值，默认值为-1。-1表示不设置指定的baseN最大值，该值由Tiling函数自行计算。 |
| maxBaseK | 输入 | 设置最大的baseK值，默认值为-1。-1表示不设置指定的baseK最大值，该值由Tiling函数自行计算。 |
| minBaseM | 输入 | 设置最小的baseM值，默认值为-1。-1表示不设置指定的baseM最小值，该值由Tiling函数自行计算。 |
| minBaseN | 输入 | 设置最小的baseN值，默认值为-1。-1表示不设置指定的baseN最小值，该值由Tiling函数自行计算。 |
| minBaseK | 输入 | 设置最小的baseK值，默认值为-1。-1表示不设置指定的baseK最小值，该值由Tiling函数自行计算。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

若baseM/baseN/baseK不满足C0_size对齐，计算Tiling时会将该值对齐到C0_size。提示，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。
