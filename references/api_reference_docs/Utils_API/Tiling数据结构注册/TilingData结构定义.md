# TilingData结构定义

**页面ID:** atlasascendc_api_07_1005  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1005.html

---

#### 功能说明

定义一个TilingData的类，添加所需的成员变量（TilingData字段），用于保存所需TilingData参数。完成该TilingData类的定义后，该类通过继承TilingDef类（用来存放、处理用户自定义Tiling结构体成员变量的基类）提供以下接口：

- set_{field_name}接口：用于设置TilingData类的字段值，field_name为定义TilingData类时添加的字段名。
- get_{field_name}接口：用于获取字段名为field_name的字段值。
- SaveToBuffer接口：完成TilingData的序列化和保存。
- GetDataSize接口：获取TilingData的长度。
- CheckAlignAndGenPlaceHolder：该接口是内部关联接口，用于框架侧检查Tiling结构体中成员变量是否满足字节对齐要求，并对不对齐的变量进行补齐，开发者无需关注。
- SetDataPtr接口：该接口为预留接口，开发者无需关注。

#### 函数原型

- 定义一个TilingData类

```
BEGIN_TILING_DATA_DEF(class_name)
```

- 添加通用数据类型的TilingData字段

```
TILING_DATA_FIELD_DEF(data_type, field_name)
```

- 添加数组类型的TilingData字段，数组的元素数据类型为通用数据类型

```
TILING_DATA_FIELD_DEF_ARR(arr_type, arr_size, field_name)
```

- 添加结构体类型的TilingData字段

```
TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)
```

- 定义结束

```
END_TILING_DATA_DEF
```

#### 参数说明

**表1 ****BEGIN_TILING_DATA_DEF**参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| class_name | 输入 | 用户定义tiling结构体名，与c++变量命名要求一致 |

**表2 ****TILING_DATA_FIELD_DEF**参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data_type | 输入 | 字段的数据类型 |
| field_name | 输入 | 字段名，与c++变量命名要求一致 |

**表3 ****TILING_DATA_FIELD_DEF_ARR**参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| arr_type | 输入 | 数组元素数据类型 |
| arr_size | 输入 | 数组元素个数 |
| field_name | 输入 | 字段名，与c++变量命名要求一致 |

**表4 ****TILING_DATA_FIELD_DEF_STRUCT**参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| struct_type | 输入 | 结构体类型 |
| field_name | 输入 | 字段名，与c++变量命名要求一致 |

#### 约束说明

- 使用时需要包含头文件register/tilingdata_base.h。
- TILING_DATA_FIELD_DEF和TILING_DATA_FIELD_DEF_ARR中定义的变量，仅支持int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float数据类型。
- TILING_DATA_FIELD_DEF_STRUCT中struct_type仅支持用BEGIN_TILING_DATA_DEF等定义的tiling结构体，不支持直接使用c++语法定义的结构体类型。
- 用户在host侧设置参数值和使用tiling数据需要使用set_xxx和get_xxx接口（xxx请替换为字段名），具体使用方法见调用示例。
- tiling数据成员需要满足字节对齐要求，即：当前数据成员dataVar位于结构体的偏移offset满足， offset % sizeof(dataVar) == 0。
- tiling结构体是全局属性，需注意应通过结构体名作为全局唯一标记，不同算子若注册同名不同结构tiling结构体则会发生未定义行为。
- 注册中间结构体时，若中间结构体名为struct_name，则第一个参数固定为struct_name#Op。
- 设置TILING_DATA_FIELD_DEF_ARR定义的字段值时，需注意set_{field_name}仅传入数组指针并按照宏中定义的数组长度进行赋值，因此，需用户自行保证传入数组指针指向的数组长度不小于宏中定义的数组长度，避免越界访问的问题。

#### 调用示例

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
static void TilingAddInit(AddCustomTilingData *tiling, uint32_t blockDim)
{
  // 设置参数值
  tiling->set_blkDim(blockDim);                  // 置值通用数据类型变量blockDim
  uint16_t arr[] = {10,2,8,2,3,4,5,2,1,2,4,4,5,};
  tiling->set_arrSample(arr);                    // 置值通用数据类型数组变量arrSample，仅会复制arr数据的前三个数据，与TILING_DATA_FIELD_DEF_ARR中arr_size一致
  tiling->mm.set_mmVar(1);                       // 置值嵌套结构体通用数据类型变量mmVar
  tiling->mm.set_mmArr(arr);                     // 置值嵌套结构体通用数据类型数组mmArr

  // 使用参数值
  uint32_t useBlockDim = tiling->get_blkDim();    // 获取通用数据类型变量blockDim
  uint32_t* arrPoint = tiling->get_arrSample();   // 获取通用数据类型数组变量arrSample
  useBlockDim = tiling->mm.get_mmVar();           // 获取嵌套结构体通用数据类型变量mmVar
  arrPoint = tiling->mm.get_mmArr();              // 获取嵌套结构体通用数据类型数组mmArr
}
```
