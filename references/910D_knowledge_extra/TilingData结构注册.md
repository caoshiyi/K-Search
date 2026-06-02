# TilingData结构注册<a name="ZH-CN_TOPIC_0000002523343818"></a>

## 功能说明<a name="zh-cn_topic_0000001576728165_section212607105720"></a>

注册定义的TilingData结构体并和自定义算子绑定。具体使用说明请参考[调用示例](#zh-cn_topic_0000001576728165_section97001499599)。

## 函数原型<a name="zh-cn_topic_0000001576728165_section129451113125413"></a>

```
#define REGISTER_TILING_DATA_CLASS(op_type, class_name)
  class op_type##class_name##Helper {
  public:
    op_type##class_name##Helper() {
      CTilingDataClassFactory::RegisterTilingData(#op_type, op_type##class_name##Helper::CreateTilingDataInstance);
    }
    static std::shared_ptr<TilingDef> CreateTilingDataInstance() {
      return std::make_shared<class_name>();
    }
  };
  op_type##class_name##Helper g_tilingdata_##op_type##class_name##helper;
```

## 参数说明<a name="zh-cn_topic_0000001576728165_section552316288018"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p10223674448"><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p10223674448"></a><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p645511218169"><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p645511218169"></a><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p1922337124411"><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p1922337124411"></a><a name="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576728165_zh-cn_topic_0000001389733241_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576728165_p390264532814"><a name="zh-cn_topic_0000001576728165_p390264532814"></a><a name="zh-cn_topic_0000001576728165_p390264532814"></a>op_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576728165_p190218452287"><a name="zh-cn_topic_0000001576728165_p190218452287"></a><a name="zh-cn_topic_0000001576728165_p190218452287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576728165_p1290244542815"><a name="zh-cn_topic_0000001576728165_p1290244542815"></a><a name="zh-cn_topic_0000001576728165_p1290244542815"></a>注册的算子名</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001576728165_row487964220282"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001576728165_p139022045172817"><a name="zh-cn_topic_0000001576728165_p139022045172817"></a><a name="zh-cn_topic_0000001576728165_p139022045172817"></a>struct_name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001576728165_p3902845162817"><a name="zh-cn_topic_0000001576728165_p3902845162817"></a><a name="zh-cn_topic_0000001576728165_p3902845162817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001576728165_p790318455287"><a name="zh-cn_topic_0000001576728165_p790318455287"></a><a name="zh-cn_topic_0000001576728165_p790318455287"></a>tiling结构体名，与c++变量命名要求一致</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001576728165_section65498832"></a>

-   使用时需要包含头文件register/tilingdata\_base.h。
-   中间结构体和定制tilingkey结构体需注意op\_type命名规则，具体见[调用示例](#zh-cn_topic_0000001576728165_section97001499599)。
-   算子定制tilingkey结构体需保证必须注册op\_type默认结构体。
-   tiling结构体是全局属性，需注意应通过结构体名作为全局唯一标记，不同算子若注册同名不同结构tiling结构体则会发生未定义行为。

## 调用示例<a name="zh-cn_topic_0000001576728165_section97001499599"></a>

-   注册算子Tiling结构体

    ```
    #include "register/tilingdata_base.h"
    
    // 定义tilingdata类
    namespace optiling {
    BEGIN_TILING_DATA_DEF(AddCustomTilingData)    // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, blkDim);    // 添加tiling字段，参与计算核数
      TILING_DATA_FIELD_DEF(uint32_t, totalSize); // 添加tiling字段，总计算数据量-输入shape大小
      TILING_DATA_FIELD_DEF(uint32_t, splitTile); // 添加tiling字段，每个core处理的数据分块计算
    END_TILING_DATA_DEF;                          // 定义结束
    // 注册算子tilingdata类到对应的AddCustom算子
    REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData) 
    }
    ```

-   注册中间结构体。当用户有结构体嵌套场景时，嵌套的结构体称为中间结构体。因为一个算子名只能注册一个Tiling结构体，为使得框架能够检测中间结构体信息，需要构造“虚拟算子名”（结构体名+Op）并通过REGISTER\_TILING\_DATA\_CLASS接口注册中间结构体，注册方式如下：

    ```
    BEGIN_TILING_DATA_DEF(Matmul)
      TILING_DATA_FIELD_DEF(uint16_t, mmVar);
      TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr);
    END_TILING_DATA_DEF;
    //注册中间结构体，第一个参数固定为struct_name#Op，第二个参数即struct_name, 如struct_name为Matmul，第一参数为MatmulOp，第二个参数为Matmul
    REGISTER_TILING_DATA_CLASS(MatmulOp, Matmul)      //注册中间结构体
    ```

-   定制tiling\_key注册不同Tiling结构体

    ```
    /*REGISTER_TILING_DATA_CLASS中第一个参数为${op_type} + ‘_’ + tiling_key。若tiling_key未注册匹配的tiling结构体，则会使用默认的结构体。如下面两种方式，tiling_key不指定或者非1情况，tiling结构体为AddStruct；tiling_key等于1的时候，tiling结构体为AddStructSample1*/
    
    // 以op_type为Add为例，默认tiling结构体注册如下
    BEGIN_TILING_DATA_DEF(AddStruct)   
      TILING_DATA_FIELD_DEF(uint16_t, mmVar);   
      TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr); 
    END_TILING_DATA_DEF; 
    REGISTER_TILING_DATA_CLASS(Add, AddStruct) 
    
    // TilingKey等于1时注册结构体如下
    BEGIN_TILING_DATA_DEF(AddStructSample1) 
      TILING_DATA_FIELD_DEF(uint16_t, mmVar);   
      TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr); 
    END_TILING_DATA_DEF; 
    REGISTER_TILING_DATA_CLASS(Add_1, AddStructSample1) 
    ```

