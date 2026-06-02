# GET_TILING_DATA_MEMBER

**页面ID:** atlasascendc_api_07_0216  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0216.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

用于获取tiling结构体的成员变量。

#### 函数原型

```
GET_TILING_DATA_MEMBER(struct_name, mem_name, tiling_data, tiling_arg)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| struct_name | 输入 | 指定的结构体名称。 |
| mem_name | 输入 | 指定的成员变量名称。 |
| tiling_data | 输出 | 返回指定Tiling结构体的成员变量。 |
| tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。 |

#### 约束说明

- 本函数需在算子kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;
    if ASCEND_IS_AIV {
        GET_TILING_DATA(tilingData, tiling);   // Vector侧使用算子默认注册的完整结构体
	op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
        op.Process();
    } else {
        GET_TILING_DATA_MEMBER(Add_Struct, tCubeTiling, tCubeTilingVar, tiling); // Cube侧仅使用算子注册结构体的成员变量tCubeTiling
	op.Init(x, y, z, tCubeTilingVar);
	op.Process();
    }
}
```
