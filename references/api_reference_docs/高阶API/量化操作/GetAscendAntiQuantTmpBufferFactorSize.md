# GetAscendAntiQuantTmpBufferFactorSize

**页面ID:** atlasascendc_api_07_10298  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10298.html

---

#### 功能说明

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

- 算子实现需要调用AscendAntiQuant接口，开发者为其预留currBuff大小的空间，利用GetAscendAntiQuantTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

**currentShapeSize = (currBuff - extraBuf) / maxLiveNodeCount / typeSize**

- 算子实现需要调用两个kernel侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值(maxLiveNodeCount、extraBuf)以及当前现有的临时空间，推导单次最大计算元素数量currentShapeSize为：

**currentShapeSize1 = (currBuff - extraBuf1) / maxLiveNodeCount1 / typeSize**

**currentShapeSize2 = (currBuff - extraBuf2) / maxLiveNodeCount2 / typeSize**

**currentShapeSize = min(currentShapeSize1, currentShapeSize2)**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

#### 函数原型

```
void GetAscendAntiQuantTmpBufferFactorSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose, ge::DataType inputDataType, ge::DataType outputDataType, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
```

#### 参数说明

**表1 **参数列表

| 参数名 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| scaleShape | 输入 | 输入scale的shape信息。 |
| isTranspose | 输入 | 是否转置。 |
| inputDataType | 输入 | 输入数据类型。ge::DataType类型，该类型的具体定义请参考DataType。 |
| outputDataType | 输入 | 输出数据类型。ge::DataType类型，该类型的具体定义请参考DataType。 |
| maxLiveNodeCount | 输出 | 最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍。 |
| extraBuf | 输出 | 使用的额外临时空间大小，单位为字节。 |

#### 约束说明

当利用maxLiveNodeCount，extraBuf反推出的currentShapeSize * typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

#### 调用示例

```
std::vector<int64_t> srcDims = { 64, 512 };
auto srcShape = ge::Shape(srcDims);
std::vector<int64_t> scaleDims = { 1, 512 };
auto scaleShape = ge::Shape(scaleDims);
bool isTranspose = false;
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
AscendC::GetAscendAntiQuantTmpBufferFactorSize(srcShape, scaleShape, isTranspose, ge::DT_INT8, ge::DT_BF16, maxLiveNodeCount, extraBuf);
```
