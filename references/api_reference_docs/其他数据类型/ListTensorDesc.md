# ListTensorDesc

**页面ID:** atlasascendc_api_07_0009  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0009.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

ListTensorDesc用来解析符合以下内存排布格式的数据， 并在kernel侧根据索引获取储存对应数据的地址及shape信息。

<!-- img2text -->
```
下图中，一行为64bit

                    首地址与数据指针的地址偏移量
                           单位为Bytes
                              ←
                    低32位表示dim数
                              ←                           高32位表示索引
                                                          与ptr对应
                                                          →

                           ┌──────────────────────┐
                           │      ptr_offset      │
                           ├────────────┬─────────┤
                           │    dim     │    0    │
                           ├────────────┴─────────┤
                 shape信息  │      shape[0]        │
                    ┌───────┤──────────────────────┤
                    │       │      shape[1]        │
                    │       ├──────────────────────┤
                    │       │         ...          │
                    │       ├──────────────────────┤
                    │       │   shape[dim - 1]     │◄──── TensorDesc
                    └───────┼────────────┬─────────┤
                           │    dim     │    1    │
                           ├────────────┴─────────┤
                           │      shape[0]        │
                           ├──────────────────────┤
                           │      shape[1]        │
                           ├──────────────────────┤
                           │         ...          │
                           ├──────────────────────┤
                           │   shape[dim - 1]     │
                           ├────────────┬─────────┤
                           │    dim     │    2    │
                           ├────────────┴─────────┤
                           │      shape[0]        │
                           ├──────────────────────┤
                           │      shape[1]        │
                           ├──────────────────────┤
                           │         ...          │
                           ├──────────────────────┤
                           │   shape[dim - 1]     │
                           ├────────────┬─────────┤
                           │    dim     │    n    │
                           ├────────────┴─────────┤
                           │      shape[0]        │
                           ├──────────────────────┤
                           │      shape[1]        │
                           ├──────────────────────┤
                           │         ...          │
                           ├──────────────────────┤
                           │   shape[dim - 1]     │
存储数据指针对应的地址 ←      ├──────────────────────┤
                           │        ptr0          │
                           ├──────────────────────┤
                           │        ptr1          │
                           ├──────────────────────┤
                           │         ...          │
                           ├──────────────────────┤
                           │        ptrn          │
                           └──────────────────────┘
                              │
                              │
                              │
                              │
                              │
                              │
                              ↓
                           ptr_offset
```

#### 需要包含的头文件

```
#include "kernel_operator_list_tensor_intf.h"
```

#### 函数原型

```
class ListTensorDesc {
    ListTensorDesc();
    ListTensorDesc(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff);
    void Init(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff);
    template<class T> void GetDesc(TensorDesc<T>& desc, uint32_t index);
    template<class T> T* GetDataPtr(uint32_t index);
    uint32_t GetSize();
}
```

#### 函数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | Tensor中元素的数据类型。 |

**表2 **函数及参数说明

| **函数名称** | **入参说明** | **含义** |
| --- | --- | --- |
| ListTensorDesc | - | 默认构造函数，需配合Init函数使用。 |
| ListTensorDesc | data：待解析数据的首地址 length：待解析内存的长度 shapeSize：数据指针的个数 length和shapeSize仅用于校验，不填写时不进行校验 | ListTensorDesc类的构造函数，用于解析对应的内存排布。 |
| Init | data：待解析数据的首地址 length：待解析内存的长度 shapeSize：数据指针的个数 length和shapeSize仅用于校验，不填写时不进行校验 | 初始化函数，用于解析对应的内存排布。 |
| GetDesc | desc：出参，解析后的Tensor描述信息 index：索引值 | 根据index获得功能说明图中对应的TensorDesc信息。 使用GetDesc前需要先调用TensorDesc.SetShapeAddr为desc指定用于储存shape信息的地址，调用GetDesc后会将shape信息写入该地址。 Atlas 推理系列产品AI Core支持该功能 Atlas 训练系列产品不支持该功能 Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该功能 Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该功能 Atlas 200I/500 A2 推理产品不支持该功能 |
| GetDataPtr | index：索引值 | 根据index获取储存对应数据的地址。 |
| GetSize | - | 获取ListTensor中包含的数据指针的个数。 |

#### 调用示例

示例中待解析的srcGm内存排布如下图所示：

<!-- img2text -->
```text
┌──────────────┐
│      72      │
├──────────────┤
│   3      0   │
├──────────────┤
│      1       │
├──────────────┤
│      2       │
├──────────────┤
│      3       │
├──────────────┤
│   3      1   │
├──────────────┤
│      3       │
├──────────────┤
│      4       │
├──────────────┤
│      5       │
├──────────────┤
│     ptr0     │
├──────────────┤
│     ptr1     │
└──────────────┘
```

```
AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(srcGm)); // srcGm为待解析的gm地址
uint32_t size = listTensorDesc.GetSize();                                       // size = 2
auto dataPtr0 = listTensorDesc.GetDataPtr<int32_t>(0);                          // 获取ptr0
auto dataPtr1 = listTensorDesc.GetDataPtr<int32_t>(1);                          // 获取ptr1

uint64_t buf[100] = {0}; // 示例中Tensor的dim为3, 此处的100表示预留足够大的空间
AscendC::TensorDesc<int32_t> desc;
desc.SetShapeAddr(buf);          // 为desc指定用于储存shape信息的地址
listTensorDesc.GetDesc(desc, 0); // 获取索引0的shape信息

uint64_t dim = desc.GetDim();   // dim = 3
uint64_t idx = desc.GetIndex(); // idx = 0
uint64_t shape[3] = {0};
for (uint32_t i = 0; i < desc.GetDim(); i++)
{
    shape[i] = desc.GetShape(i); // GetShape(0) = 1, GetShape(1) = 2, GetShape(2) = 3
}
auto ptr = desc.GetDataPtr();
```
