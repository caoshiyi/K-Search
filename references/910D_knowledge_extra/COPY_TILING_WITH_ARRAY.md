# COPY\_TILING\_WITH\_ARRAY<a name="ZH-CN_TOPIC_0000002523303558"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

拷贝指定大小的数组内容到目标数组中，并返回指向拷贝后数组的指针。适用于拷贝一个结构体的数组成员变量的场景。该宏将指定数组拷贝至栈上，适用于频繁访问Tiling数据的场景，能够加快数据访问速度。

## 函数原型<a name="zh-cn_topic_0000001526206862_section1630753514297"></a>

```
COPY_TILING_WITH_ARRAY(arr_type, arr_count, src_ptr, dst_ptr)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row472122142712"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p6721528274"><a name="p6721528274"></a><a name="p6721528274"></a>arr_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p17721152142712"><a name="p17721152142712"></a><a name="p17721152142712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p372120282711"><a name="p372120282711"></a><a name="p372120282711"></a>指定要拷贝的数组类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>arr_count</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a>指定要拷贝的数组大小。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001526206862_row1239183183016"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001526206862_p223953193015"><a name="zh-cn_topic_0000001526206862_p223953193015"></a><a name="zh-cn_topic_0000001526206862_p223953193015"></a>src_ptr</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001526206862_p7239938308"><a name="zh-cn_topic_0000001526206862_p7239938308"></a><a name="zh-cn_topic_0000001526206862_p7239938308"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001526206862_p72396320307"><a name="zh-cn_topic_0000001526206862_p72396320307"></a><a name="zh-cn_topic_0000001526206862_p72396320307"></a>指向tiling_struct结构体的指针。</p>
</td>
</tr>
<tr id="row17376172914010"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p1237615295409"><a name="p1237615295409"></a><a name="p1237615295409"></a>dst_ptr</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p6376162913400"><a name="p6376162913400"></a><a name="p6376162913400"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1537672974019"><a name="p1537672974019"></a><a name="p1537672974019"></a>返回拷贝后的指向类型为arr_type、大小为arr_count的数组指针。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

-   该宏需在算子Kernel代码处使用，并且传入的dst\_ptr参数无需声明类型。
-   该宏需要和[GET\_TILING\_DATA\_PTR\_WITH\_STRUCT](GET_TILING_DATA_PTR_WITH_STRUCT.md)配合使用，输入参数src\_ptr为GET\_TILING\_DATA\_PTR\_WITH\_STRUCT获取到的指针。
-   该宏获取到的dst\_ptr指针指向的数组是局部变量，请确保在合理作用域范围内使用。
-   暂不支持Kernel直调工程。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;

    GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);
 
    if ASCEND_IS_AIV {
        COPY_TILING_WITH_ARRAY(uint64_t, 2, tilingDataPtr->vectorTilingArray, vTilingArrayPtr);        
        op.Init(x, y, z, (*vTilingArrayPtr)[0], (*vTilingArrayPtr)[1]);
        op.Process();
    } else {
        COPY_TILING_WITH_ARRAY(uint64_t, 2, tilingDataPtr->cubeTilingArray, cTilingArrayPtr);
	op.Init(x, y, z, (*cTilingArrayPtr)[0], (*cTilingArrayPtr)[1]);
	op.Process();
    }
}
```

