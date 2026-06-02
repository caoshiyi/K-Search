# TilingData结构注册

**页面ID:** atlasascendc_api_07_1006  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1006.html

---

#### 功能说明

注册定义的TilingData结构体并和自定义算子绑定。具体使用说明请参考调用示例。

#### 函数原型

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

#### 参数说明

**表1 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_type | 输入 | 注册的算子名 |
| struct_name | 输入 | tiling结构体名，与c++变量命名要求一致 |

#### 约束说明

- 使用时需要包含头文件register/tilingdata_base.h。
- 中间结构体和定制tilingkey结构体需注意op_type命名规则，具体见调用示例。
- 算子定制tilingkey结构体需保证必须注册op_type默认结构体。
- tiling结构体是全局属性，需注意应通过结构体名作为全局唯一标记，不同算子若注册同名不同结构tiling结构体则会发生未定义行为。

#### 调用示例

- 注册算子Tiling结构体

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

- 注册中间结构体。当用户有结构体嵌套场景时，嵌套的结构体称为中间结构体。因为一个算子名只能注册一个Tiling结构体，为使得框架能够检测中间结构体信息，需要构造“虚拟算子名”（结构体名+Op）并通过REGISTER_TILING_DATA_CLASS接口注册中间结构体，注册方式如下：

```
BEGIN_TILING_DATA_DEF(Matmul)
  TILING_DATA_FIELD_DEF(uint16_t, mmVar);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr);
END_TILING_DATA_DEF;
//注册中间结构体，第一个参数固定为struct_name#Op，第二个参数即struct_name, 如struct_name为Matmul，第一参数为MatmulOp，第二个参数为Matmul
REGISTER_TILING_DATA_CLASS(MatmulOp, Matmul)      //注册中间结构体
```

- 定制tiling_key注册不同Tiling结构体

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
