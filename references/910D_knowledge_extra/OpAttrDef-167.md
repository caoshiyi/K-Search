# OpAttrDef<a name="ZH-CN_TOPIC_0000002523344796"></a>

## 功能说明<a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section36583473819"></a>

定义算子属性。

## 函数原型<a name="zh-cn_topic_0000001656780128_section3992421457"></a>

```
class OpAttrDef {
public:
  explicit OpAttrDef(const char *name);
  OpAttrDef(const OpAttrDef &attr_def);
  ~OpAttrDef();
  OpAttrDef &operator=(const OpAttrDef &attr_def);
  OpAttrDef &AttrType(Option attr_type);
  OpAttrDef &Bool(void);
  OpAttrDef &Bool(bool value);
  OpAttrDef &Float(void);
  OpAttrDef &Float(float value);
  OpAttrDef &Int(void);
  OpAttrDef &Int(int64_t value);
  OpAttrDef &String(void);
  OpAttrDef &String(const char *value);
  OpAttrDef &ListBool(void);
  OpAttrDef &ListBool(std::vector<bool> value);
  OpAttrDef &ListFloat(void);
  OpAttrDef &ListFloat(std::vector<float> value);
  OpAttrDef &ListInt(void);
  OpAttrDef &ListInt(std::vector<int64_t> value);
  OpAttrDef &ListListInt(void);
  OpAttrDef &ListListInt(std::vector<std::vector<int64_t>> value);
  OpAttrDef &Version(uint32_t version);
  ge::AscendString &GetName(void) const;
  bool IsRequired(void);
private:
  ...
};
```

## 函数说明<a name="zh-cn_topic_0000001656780128_section1340317244469"></a>

**表 1**  OpAttrDef类成员函数说明

<a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_table18149577913"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_row61411571196"><th class="cellrowborder" valign="top" width="16.650000000000002%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p2093713281104"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p2093713281104"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p2093713281104"></a>函数名称</p>
</th>
<th class="cellrowborder" valign="top" width="20.1%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p1593811282101"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p1593811282101"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p1593811282101"></a>入参说明</p>
</th>
<th class="cellrowborder" valign="top" width="63.24999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p393813285106"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p393813285106"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p393813285106"></a>功能说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_row8906103284616"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p1985231611167"><a name="zh-cn_topic_0000001656780128_p1985231611167"></a><a name="zh-cn_topic_0000001656780128_p1985231611167"></a>AttrType</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a>attr_type: 属性类型</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_p096733515614"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_p096733515614"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001526594958_p096733515614"></a>设置算子属性类型，取值为：OPTIONAL（可选）、REQUIRED（必选）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_row433315311399"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p414793501615"><a name="zh-cn_topic_0000001656780128_p414793501615"></a><a name="zh-cn_topic_0000001656780128_p414793501615"></a>Bool</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p05124438166"><a name="zh-cn_topic_0000001656780128_p05124438166"></a><a name="zh-cn_topic_0000001656780128_p05124438166"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p8444068494"><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p8444068494"></a><a name="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_p8444068494"></a>设置算子属性数据类型为Bool</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_row833317313398"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p2991838121618"><a name="zh-cn_topic_0000001656780128_p2991838121618"></a><a name="zh-cn_topic_0000001656780128_p2991838121618"></a>Bool</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p1135123414252"><a name="zh-cn_topic_0000001656780128_p1135123414252"></a><a name="zh-cn_topic_0000001656780128_p1135123414252"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p197371148163118"><a name="zh-cn_topic_0000001656780128_p197371148163118"></a><a name="zh-cn_topic_0000001656780128_p197371148163118"></a>设置算子属性数据类型为Bool，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_zh-cn_topic_0000001441184464_row16152579911"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p045625381620"><a name="zh-cn_topic_0000001656780128_p045625381620"></a><a name="zh-cn_topic_0000001656780128_p045625381620"></a>Float</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p1254453017309"><a name="zh-cn_topic_0000001656780128_p1254453017309"></a><a name="zh-cn_topic_0000001656780128_p1254453017309"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p454413012305"><a name="zh-cn_topic_0000001656780128_p454413012305"></a><a name="zh-cn_topic_0000001656780128_p454413012305"></a>设置算子属性数据类型为Float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row1559818378231"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p334545512166"><a name="zh-cn_topic_0000001656780128_p334545512166"></a><a name="zh-cn_topic_0000001656780128_p334545512166"></a>Float</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p1934919344253"><a name="zh-cn_topic_0000001656780128_p1934919344253"></a><a name="zh-cn_topic_0000001656780128_p1934919344253"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p04071414839"><a name="zh-cn_topic_0000001656780128_p04071414839"></a><a name="zh-cn_topic_0000001656780128_p04071414839"></a>设置算子属性数据类型为Float，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row11161541192316"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p124601136173"><a name="zh-cn_topic_0000001656780128_p124601136173"></a><a name="zh-cn_topic_0000001656780128_p124601136173"></a>Int</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p17205733133017"><a name="zh-cn_topic_0000001656780128_p17205733133017"></a><a name="zh-cn_topic_0000001656780128_p17205733133017"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p2020523363016"><a name="zh-cn_topic_0000001656780128_p2020523363016"></a><a name="zh-cn_topic_0000001656780128_p2020523363016"></a>设置算子属性数据类型为Int</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row61311051338"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p271512531715"><a name="zh-cn_topic_0000001656780128_p271512531715"></a><a name="zh-cn_topic_0000001656780128_p271512531715"></a>Int</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p713214510311"><a name="zh-cn_topic_0000001656780128_p713214510311"></a><a name="zh-cn_topic_0000001656780128_p713214510311"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p132411030939"><a name="zh-cn_topic_0000001656780128_p132411030939"></a><a name="zh-cn_topic_0000001656780128_p132411030939"></a>设置算子属性数据类型为Int，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row855612286173"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p1255662841719"><a name="zh-cn_topic_0000001656780128_p1255662841719"></a><a name="zh-cn_topic_0000001656780128_p1255662841719"></a>String</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p11878123519306"><a name="zh-cn_topic_0000001656780128_p11878123519306"></a><a name="zh-cn_topic_0000001656780128_p11878123519306"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p14878193593016"><a name="zh-cn_topic_0000001656780128_p14878193593016"></a><a name="zh-cn_topic_0000001656780128_p14878193593016"></a>设置算子属性数据类型为String</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row1147831122216"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p2478511122211"><a name="zh-cn_topic_0000001656780128_p2478511122211"></a><a name="zh-cn_topic_0000001656780128_p2478511122211"></a>String</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p747881152213"><a name="zh-cn_topic_0000001656780128_p747881152213"></a><a name="zh-cn_topic_0000001656780128_p747881152213"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p547814111225"><a name="zh-cn_topic_0000001656780128_p547814111225"></a><a name="zh-cn_topic_0000001656780128_p547814111225"></a>设置算子属性数据类型为String，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row1910382522211"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p41491428112418"><a name="zh-cn_topic_0000001656780128_p41491428112418"></a><a name="zh-cn_topic_0000001656780128_p41491428112418"></a>ListBool</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p11480104373019"><a name="zh-cn_topic_0000001656780128_p11480104373019"></a><a name="zh-cn_topic_0000001656780128_p11480104373019"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p1448064317306"><a name="zh-cn_topic_0000001656780128_p1448064317306"></a><a name="zh-cn_topic_0000001656780128_p1448064317306"></a>设置算子属性数据类型为ListBool</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row730692212212"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p348153019244"><a name="zh-cn_topic_0000001656780128_p348153019244"></a><a name="zh-cn_topic_0000001656780128_p348153019244"></a>ListBool</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p230720227229"><a name="zh-cn_topic_0000001656780128_p230720227229"></a><a name="zh-cn_topic_0000001656780128_p230720227229"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p1530792214229"><a name="zh-cn_topic_0000001656780128_p1530792214229"></a><a name="zh-cn_topic_0000001656780128_p1530792214229"></a>设置算子属性数据类型为ListBool，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row79541415102215"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p183013613248"><a name="zh-cn_topic_0000001656780128_p183013613248"></a><a name="zh-cn_topic_0000001656780128_p183013613248"></a>ListFloat</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p12930204511309"><a name="zh-cn_topic_0000001656780128_p12930204511309"></a><a name="zh-cn_topic_0000001656780128_p12930204511309"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p193154519302"><a name="zh-cn_topic_0000001656780128_p193154519302"></a><a name="zh-cn_topic_0000001656780128_p193154519302"></a>设置算子属性数据类型为ListFloat</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row228196227"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p12364113852420"><a name="zh-cn_topic_0000001656780128_p12364113852420"></a><a name="zh-cn_topic_0000001656780128_p12364113852420"></a>ListFloat</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p82219112219"><a name="zh-cn_topic_0000001656780128_p82219112219"></a><a name="zh-cn_topic_0000001656780128_p82219112219"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p1455218391332"><a name="zh-cn_topic_0000001656780128_p1455218391332"></a><a name="zh-cn_topic_0000001656780128_p1455218391332"></a>设置算子属性数据类型为ListFloat，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row8424456244"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p136785561242"><a name="zh-cn_topic_0000001656780128_p136785561242"></a><a name="zh-cn_topic_0000001656780128_p136785561242"></a>ListInt</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p865744873011"><a name="zh-cn_topic_0000001656780128_p865744873011"></a><a name="zh-cn_topic_0000001656780128_p865744873011"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p16657114812304"><a name="zh-cn_topic_0000001656780128_p16657114812304"></a><a name="zh-cn_topic_0000001656780128_p16657114812304"></a>设置算子属性数据类型为ListInt</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row17577122518"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p1875713142514"><a name="zh-cn_topic_0000001656780128_p1875713142514"></a><a name="zh-cn_topic_0000001656780128_p1875713142514"></a>ListInt</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p1175714182510"><a name="zh-cn_topic_0000001656780128_p1175714182510"></a><a name="zh-cn_topic_0000001656780128_p1175714182510"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p136389175357"><a name="zh-cn_topic_0000001656780128_p136389175357"></a><a name="zh-cn_topic_0000001656780128_p136389175357"></a>设置算子属性数据类型为ListInt，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row10511155142519"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p381219185254"><a name="zh-cn_topic_0000001656780128_p381219185254"></a><a name="zh-cn_topic_0000001656780128_p381219185254"></a>ListListInt</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p1366115143012"><a name="zh-cn_topic_0000001656780128_p1366115143012"></a><a name="zh-cn_topic_0000001656780128_p1366115143012"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p436685116306"><a name="zh-cn_topic_0000001656780128_p436685116306"></a><a name="zh-cn_topic_0000001656780128_p436685116306"></a>设置算子属性数据类型为ListListInt</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row756154892410"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p193941710257"><a name="zh-cn_topic_0000001656780128_p193941710257"></a><a name="zh-cn_topic_0000001656780128_p193941710257"></a>ListListInt</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p16561104882413"><a name="zh-cn_topic_0000001656780128_p16561104882413"></a><a name="zh-cn_topic_0000001656780128_p16561104882413"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p1471518283520"><a name="zh-cn_topic_0000001656780128_p1471518283520"></a><a name="zh-cn_topic_0000001656780128_p1471518283520"></a>设置算子属性数据类型为ListListInt，并设置属性默认值为value。属性类型设置为OPTIONAL时必须调用该类接口设置默认值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row1652113814816"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p152158154812"><a name="zh-cn_topic_0000001656780128_p152158154812"></a><a name="zh-cn_topic_0000001656780128_p152158154812"></a>Version</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p752117864811"><a name="zh-cn_topic_0000001656780128_p752117864811"></a><a name="zh-cn_topic_0000001656780128_p752117864811"></a>version：配置的版本号</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p252168154812"><a name="zh-cn_topic_0000001656780128_p252168154812"></a><a name="zh-cn_topic_0000001656780128_p252168154812"></a>新增可选属性时，为了保持原有单算子API(aclnnxxx)接口的兼容性，可以通过Version接口配置aclnn接口的版本号，版本号需要从1开始配，且应该连续配置（和<a href="Version.md">可选输入</a>统一编号）。配置后，自动生成的aclnn接口会携带版本号。高版本号的接口会包含低版本号接口的所有参数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row430719366115"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p5182152013136"><a name="zh-cn_topic_0000001656780128_p5182152013136"></a><a name="zh-cn_topic_0000001656780128_p5182152013136"></a>GetName</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p8667522131317"><a name="zh-cn_topic_0000001656780128_p8667522131317"></a><a name="zh-cn_topic_0000001656780128_p8667522131317"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p15307113621112"><a name="zh-cn_topic_0000001656780128_p15307113621112"></a><a name="zh-cn_topic_0000001656780128_p15307113621112"></a>获取属性名称。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001656780128_row125751739111113"><td class="cellrowborder" valign="top" width="16.650000000000002%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001656780128_p2182020131317"><a name="zh-cn_topic_0000001656780128_p2182020131317"></a><a name="zh-cn_topic_0000001656780128_p2182020131317"></a>IsRequired</p>
</td>
<td class="cellrowborder" valign="top" width="20.1%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001656780128_p166762212139"><a name="zh-cn_topic_0000001656780128_p166762212139"></a><a name="zh-cn_topic_0000001656780128_p166762212139"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001656780128_p757503901110"><a name="zh-cn_topic_0000001656780128_p757503901110"></a><a name="zh-cn_topic_0000001656780128_p757503901110"></a>判断算子属性是否为必选，必选返回true，可选返回false。</p>
</td>
</tr>
</tbody>
</table>

