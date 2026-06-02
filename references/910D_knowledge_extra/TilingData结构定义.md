# TilingData结构定义<a name="ZH-CN_TOPIC_0000002554423939"></a>

## 功能说明<a name="zh-cn_topic_0000001576606633_section212607105720"></a>

定义一个TilingData的类，添加所需的成员变量（TilingData字段），用于保存所需TilingData参数。完成该TilingData类的定义后，该类通过继承TilingDef类（用来存放、处理用户自定义Tiling结构体成员变量的基类）提供以下接口：

-   set\_\{field\_name\}接口：用于设置TilingData类的字段值，field\_name为定义TilingData类时添加的字段名。
-   get\_\{field\_name\}接口：用于获取字段名为field\_name的字段值。
-   SaveToBuffer接口：完成TilingData的序列化和保存。
-   GetDataSize接口：获取TilingData的长度。
-   CheckAlignAndGenPlaceHolder：该接口是内部关联接口，用于框架侧检查Tiling结构体中成员变量是否满足字节对齐要求，并对不对齐的变量进行补齐，开发者无需关注。
-   SetDataPtr接口：该接口为预留接口，开发者无需关注。

## 函数原型<a name="zh-cn_topic_0000001576606633_section129451113125413"></a>

-   定义一个TilingData类

    ```
    BEGIN_TILING_DATA_DEF(class_name)
    ```

-   添加通用数据类型的TilingData字段

    ```
    TILING_DATA_FIELD_DEF(data_type, field_name)
    ```

-   添加数组类型的TilingData字段，数组的元素数据类型为通用数据类型

    ```
    TILING_DATA_FIELD_DEF_ARR(arr_type, arr_size, field_name)
    ```

-   添加结构体类型的TilingData字段

    ```
    TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)
    ```

-   定义结束

    ```
    END_TILING_DATA_DEF
    ```

## 参数说明<a name="zh-cn_topic_0000001576606633_section552316288018"></a>

**表 1** **BEGIN\_TILING\_DATA\_DEF**参数说明

<a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p10223674448"><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p10223674448"></a><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p645511218169"><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p645511218169"></a><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p1922337124411"><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p1922337124411"></a><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p2340183613156"><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p2340183613156"></a><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p2340183613156"></a>class_name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p143401361158"><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p143401361158"></a><a name="zh-cn_topic_0000001576606633_zh-cn_topic_0000001389733241_p143401361158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p10538192216213"><a name="zh-cn_topic_0000001576606633_p10538192216213"></a><a name="zh-cn_topic_0000001576606633_p10538192216213"></a>用户定义tiling结构体名，与c++变量命名要求一致</p>
</td>
</tr>
</tbody>
</table>

**表 2** **TILING\_DATA\_FIELD\_DEF**参数说明

<a name="zh-cn_topic_0000001576606633_table396666731"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576606633_row296610614315"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001576606633_p1496666238"><a name="zh-cn_topic_0000001576606633_p1496666238"></a><a name="zh-cn_topic_0000001576606633_p1496666238"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001576606633_p59661261731"><a name="zh-cn_topic_0000001576606633_p59661261731"></a><a name="zh-cn_topic_0000001576606633_p59661261731"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001576606633_p5966136534"><a name="zh-cn_topic_0000001576606633_p5966136534"></a><a name="zh-cn_topic_0000001576606633_p5966136534"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576606633_row109661268319"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p59661461233"><a name="zh-cn_topic_0000001576606633_p59661461233"></a><a name="zh-cn_topic_0000001576606633_p59661461233"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p496626537"><a name="zh-cn_topic_0000001576606633_p496626537"></a><a name="zh-cn_topic_0000001576606633_p496626537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p424615489310"><a name="zh-cn_topic_0000001576606633_p424615489310"></a><a name="zh-cn_topic_0000001576606633_p424615489310"></a>字段的数据类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001576606633_row185910339316"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p1559115331134"><a name="zh-cn_topic_0000001576606633_p1559115331134"></a><a name="zh-cn_topic_0000001576606633_p1559115331134"></a>field_name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p195916331037"><a name="zh-cn_topic_0000001576606633_p195916331037"></a><a name="zh-cn_topic_0000001576606633_p195916331037"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p15913331135"><a name="zh-cn_topic_0000001576606633_p15913331135"></a><a name="zh-cn_topic_0000001576606633_p15913331135"></a>字段名，与c++变量命名要求一致</p>
</td>
</tr>
</tbody>
</table>

**表 3** **TILING\_DATA\_FIELD\_DEF\_ARR**参数说明

<a name="zh-cn_topic_0000001576606633_table164915116419"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576606633_row13490116418"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001576606633_p124919111447"><a name="zh-cn_topic_0000001576606633_p124919111447"></a><a name="zh-cn_topic_0000001576606633_p124919111447"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001576606633_p34913111848"><a name="zh-cn_topic_0000001576606633_p34913111848"></a><a name="zh-cn_topic_0000001576606633_p34913111848"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001576606633_p124913111748"><a name="zh-cn_topic_0000001576606633_p124913111748"></a><a name="zh-cn_topic_0000001576606633_p124913111748"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576606633_row44919112419"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p44701128172914"><a name="zh-cn_topic_0000001576606633_p44701128172914"></a><a name="zh-cn_topic_0000001576606633_p44701128172914"></a>arr_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p2491111848"><a name="zh-cn_topic_0000001576606633_p2491111848"></a><a name="zh-cn_topic_0000001576606633_p2491111848"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p74291753113019"><a name="zh-cn_topic_0000001576606633_p74291753113019"></a><a name="zh-cn_topic_0000001576606633_p74291753113019"></a>数组元素数据类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001576606633_row84971116417"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p8491711042"><a name="zh-cn_topic_0000001576606633_p8491711042"></a><a name="zh-cn_topic_0000001576606633_p8491711042"></a>arr_size</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p1249181116413"><a name="zh-cn_topic_0000001576606633_p1249181116413"></a><a name="zh-cn_topic_0000001576606633_p1249181116413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p169783181544"><a name="zh-cn_topic_0000001576606633_p169783181544"></a><a name="zh-cn_topic_0000001576606633_p169783181544"></a>数组元素个数</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001576606633_row9483162022920"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p1448419208292"><a name="zh-cn_topic_0000001576606633_p1448419208292"></a><a name="zh-cn_topic_0000001576606633_p1448419208292"></a>field_name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p146498486302"><a name="zh-cn_topic_0000001576606633_p146498486302"></a><a name="zh-cn_topic_0000001576606633_p146498486302"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p1048422011297"><a name="zh-cn_topic_0000001576606633_p1048422011297"></a><a name="zh-cn_topic_0000001576606633_p1048422011297"></a>字段名，与c++变量命名要求一致</p>
</td>
</tr>
</tbody>
</table>

**表 4** **TILING\_DATA\_FIELD\_DEF\_STRUCT**参数说明

<a name="zh-cn_topic_0000001576606633_table69741814053"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576606633_row1197414141052"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001576606633_p4974814859"><a name="zh-cn_topic_0000001576606633_p4974814859"></a><a name="zh-cn_topic_0000001576606633_p4974814859"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001576606633_p0974414053"><a name="zh-cn_topic_0000001576606633_p0974414053"></a><a name="zh-cn_topic_0000001576606633_p0974414053"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001576606633_p15974914157"><a name="zh-cn_topic_0000001576606633_p15974914157"></a><a name="zh-cn_topic_0000001576606633_p15974914157"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576606633_row2097412141353"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p19938194863116"><a name="zh-cn_topic_0000001576606633_p19938194863116"></a><a name="zh-cn_topic_0000001576606633_p19938194863116"></a>struct_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p4974714459"><a name="zh-cn_topic_0000001576606633_p4974714459"></a><a name="zh-cn_topic_0000001576606633_p4974714459"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p199749141554"><a name="zh-cn_topic_0000001576606633_p199749141554"></a><a name="zh-cn_topic_0000001576606633_p199749141554"></a>结构体类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001576606633_row79741141253"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576606633_p20562560312"><a name="zh-cn_topic_0000001576606633_p20562560312"></a><a name="zh-cn_topic_0000001576606633_p20562560312"></a>field_name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576606633_p597471415519"><a name="zh-cn_topic_0000001576606633_p597471415519"></a><a name="zh-cn_topic_0000001576606633_p597471415519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576606633_p1097411419512"><a name="zh-cn_topic_0000001576606633_p1097411419512"></a><a name="zh-cn_topic_0000001576606633_p1097411419512"></a>字段名，与c++变量命名要求一致</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001576606633_section65498832"></a>

-   使用时需要包含头文件register/tilingdata\_base.h。
-   TILING\_DATA\_FIELD\_DEF和TILING\_DATA\_FIELD\_DEF\_ARR中定义的变量，仅支持int8\_t, uint8\_t, int16\_t, uint16\_t, int32\_t, uint32\_t, int64\_t, uint64\_t, float数据类型。
-   TILING\_DATA\_FIELD\_DEF\_STRUCT中struct\_type仅支持用BEGIN\_TILING\_DATA\_DEF等定义的tiling结构体，不支持直接使用c++语法定义的结构体类型。
-   用户在host侧设置参数值和使用tiling数据需要使用set\_xxx和get\_xxx接口（xxx请替换为字段名），具体使用方法见调用示例。
-   tiling数据成员需要满足字节对齐要求，即：当前数据成员dataVar位于结构体的偏移offset满足， offset % sizeof\(dataVar\) == 0。
-   tiling结构体是全局属性，需注意应通过结构体名作为全局唯一标记，不同算子若注册同名不同结构tiling结构体则会发生未定义行为。
-   注册中间结构体时，若中间结构体名为struct\_name，则第一个参数固定为struct\_name\#Op。
-   设置TILING\_DATA\_FIELD\_DEF\_ARR定义的字段值时，需注意set\_\{field\_name\}仅传入数组指针并按照宏中定义的数组长度进行赋值，因此，需用户自行保证传入数组指针指向的数组长度不小于宏中定义的数组长度，避免越界访问的问题。

## 调用示例<a name="zh-cn_topic_0000001576606633_section97001499599"></a>

```
#include "register/tilingdata_base.h"

// 定义tilingdata类
namespace optiling {
BEGIN_TILING_DATA_DEF(Matmul)
  TILING_DATA_FIELD_DEF(uint16_t, mmVar);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr);
END_TILING_DATA_DEF;
//注册中间结构体，第一个参数固定为struct_name#Op，第二个参数即struct_name, 如struct_name为Matmul，第一个参数为MatmulOp，第二个参数为Matmul
REGISTER_TILING_DATA_CLASS(MatmulOp, Matmul)      //注册中间结构体

BEGIN_TILING_DATA_DEF(AddCustomTilingData)        // 注册一个tiling类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, blkDim);        // 添加tiling变量类型字段，参与计算核数
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);     // 添加tiling变量类型字段，总计算数据量
  TILING_DATA_FIELD_DEF(uint32_t, splitTile);     // 添加tiling变量类型字段，每个core处理的数据分块计算
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, arrSample);    // 添加tiling数组类型字段
  TILING_DATA_FIELD_DEF_STRUCT(Matmul, mm);             // 添加tiling结构体类型字段
END_TILING_DATA_DEF;                                    // 定义结束
// 注册算子tilingdata类到对应的AddCustom算子
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData) 
}

// host侧设置参数值和使用tiling参数
static void TilingAddInit(AddCustomTilingData *tiling, uint32_t numBlocks)
{
  // 设置参数值
  tiling->set_blkDim(numBlocks);                  // 置值通用数据类型变量numBlocks
  uint16_t arr[] = {10,2,8,2,3,4,5,2,1,2,4,4,5,};
  tiling->set_arrSample(arr);                    // 置值通用数据类型数组变量arrSample，仅会复制arr数据的前三个数据，与TILING_DATA_FIELD_DEF_ARR中arr_size一致
  tiling->mm.set_mmVar(1);                       // 置值嵌套结构体通用数据类型变量mmVar
  tiling->mm.set_mmArr(arr);                     // 置值嵌套结构体通用数据类型数组mmArr
  
  // 使用参数值
  uint32_t useNumBlocks = tiling->get_blkDim();    // 获取通用数据类型变量numBlocks
  uint32_t* arrPoint = tiling->get_arrSample();   // 获取通用数据类型数组变量arrSample
  useNumBlocks = tiling->mm.get_mmVar();           // 获取嵌套结构体通用数据类型变量mmVar
  arrPoint = tiling->mm.get_mmArr();              // 获取嵌套结构体通用数据类型数组mmArr
}
```

