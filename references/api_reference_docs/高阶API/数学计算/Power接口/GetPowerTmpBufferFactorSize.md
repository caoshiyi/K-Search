# GetPowerTmpBufferFactorSize

**页面ID:** atlasascendc_api_07_0522  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0522.html

---

#### 功能说明

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

算子实现需要调用Power接口，开发者为其预留currBuff大小的空间，利用GetPowerTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

**currentShapeSize = (currBuff - extraBuf) / maxLiveNodeCount / typeSize**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

#### 函数原型

```
void GetPowerTmpBufferFactorSize(const bool baseIsTensor, const bool expIsTensor, const bool typeIsInt, const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
```

#### 参数说明

**表1 **参数列表

| 参数名 | 输入/输出 | 功能 |
| --- | --- | --- |
| baseIsTensor | 输入 | 底数是Tensor输入true，否则输入false |
| expIsTensor | 输入 | 指数是Tensor输入true，否则输入false |
| typeIsInt | 输入 | 数据类型是int时输入true，否则输入false |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| maxLiveNodeCount | 输出 | 最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍。 |
| extraBuffer | 输出 | 使用的额外临时空间大小，单位为字节。 |

#### 约束说明

当利用maxLiveNodeCount，extraBuf反推出的currentShapeSize  * typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

#### 调用示例

完整的调用样例请参考更多样例。

```
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
AscendC::GetPowerTmpBufferFactorSize(true, true, true, 4, maxLiveNodeCount, extraBuf);
```
