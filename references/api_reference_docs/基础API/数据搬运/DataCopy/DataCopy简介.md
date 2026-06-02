# DataCopy简介

**页面ID:** atlasascendc_api_07_0102  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0102.html

---

DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。

下表展示了DataCopy各项功能的描述和其通路的支持情况。

| 功能 | 描述 | Local Memory -> Global Memory | Global Memory -> Local Memory | Local Memory -> Local Memory |
| --- | --- | --- | --- | --- |
| 基础数据搬运 | 提供基础的数据搬运能力，数据在传输过程中保持原始格式和内容不变，支持连续和非连续的数据搬运。 | √ | √ | √ |
| 增强数据搬运 | 对数据搬运能力进行增强，相比于基础数据搬运接口，增加了CO1->CO2通路的随路计算。 | √ | √ | √ |
| 切片数据搬运 | 支持数据的切片搬运，提取多维Tensor数据的子集进行搬运。 | √ | √ | × |
| 随路转换ND2NZ搬运 | 支持在数据搬运时进行ND到NZ格式的转换。 | × | √ | √ |
| 随路转换NZ2ND搬运 | 支持在数据搬运时进行NZ到ND格式的转换。 | √ | × | × |
| 随路量化激活搬运 | 支持在数据搬运过程中进行量化和Relu激活等操作，同时支持Local Memory到Global Memory通路NZ到ND格式的转换。 | √ | × | √ |
