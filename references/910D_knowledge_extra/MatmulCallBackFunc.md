# MatmulCallBackFunc<a name="ZH-CN_TOPIC_0000002554423595"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section451064612817"></a>

模板参数MatmulCallBackFunc支持用户定制化Matmul的A矩阵、B矩阵及C矩阵的搬入搬出功能，如非连续搬入或针对搬出设置不同的数据片段间隔等。具体方式为：用户根据实际需要，实现一个或多个自定义的搬运函数，定义Matmul对象时，通过模板参数MatmulCallBackFunc，传入实现的搬运函数的函数指针，传入的函数指针会替换Matmul流程中默认的搬运函数。

MatmulCallBackFunc中包含3个可由用户自定义的回调函数接口，即用户可配置3个函数指针。3个函数指针分别为C矩阵从CO1拷贝到GM、A矩阵从GM拷贝到A1、B矩阵从GM拷贝到B1的回调函数指针。3个函数指针的位置固定，不使用自定义搬运函数的函数指针位置需要设为空指针。各个功能回调函数接口定义及参数释义见[表1 MatmulCallBackFunc回调函数接口及参数说明](#table10989848113111)。每个回调函数实现矩阵搬运中单个基本块（A矩阵基本块baseM \* baseK、B矩阵基本块baseK \* baseN、C矩阵基本块baseM \* baseN）的搬运策略，无法对整块内存空间进行管理。Matmul默认的搬运函数实现单核上单个基本块的搬运，搬运的基本块大小是固定的，在完整的Matmul计算过程中，多次调用搬运函数，对连续排布的基本块按顺序逐个搬运，以搬入A矩阵的过程为例，示意图如下。

**图 1**  Matmul默认搬入A矩阵示意图<a name="fig16926152119357"></a>  
<!-- img2text -->
```text
                    单次搬运1个基本块

                baseK                              baseK
      ┌─────────────────────────┐      ┌─────────────────────────┐
baseM │    1    │    2    │    3    │      │    1    │    2    │    3    │ baseM
      ├─────────┼─────────┼─────────┤      └─────────────────────────┘
      │         │         │         │
      └─────────────────────────┘

                     GM                                  A1

                    ╰────────────────────────────────→
```

**表 1**  MatmulCallBackFunc回调函数接口及参数说明

<a name="table10989848113111"></a>
<table><thead align="left"><tr id="row69897486312"><th class="cellrowborder" valign="top" width="13.84%" id="mcps1.2.4.1.1"><p id="p1698984843110"><a name="p1698984843110"></a><a name="p1698984843110"></a>回调函数功能</p>
</th>
<th class="cellrowborder" valign="top" width="34.23%" id="mcps1.2.4.1.2"><p id="p7989144815314"><a name="p7989144815314"></a><a name="p7989144815314"></a>回调函数接口</p>
</th>
<th class="cellrowborder" valign="top" width="51.93%" id="mcps1.2.4.1.3"><p id="p16989134818316"><a name="p16989134818316"></a><a name="p16989134818316"></a>参数说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row598917483318"><td class="cellrowborder" valign="top" width="13.84%" headers="mcps1.2.4.1.1 "><p id="p4806896321"><a name="p4806896321"></a><a name="p4806896321"></a>可自定义设置不同的搬出数据片段数目等参数，实现将Matmul计算结果从CO1搬出到GM的功能</p>
</td>
<td class="cellrowborder" valign="top" width="34.23%" headers="mcps1.2.4.1.2 "><p id="p18805298329"><a name="p18805298329"></a><a name="p18805298329"></a>void DataCopyOut(const __gm__ void *gm, const LocalTensor&lt;int8_t&gt; &amp;co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)</p>
</td>
<td class="cellrowborder" valign="top" width="51.93%" headers="mcps1.2.4.1.3 "><p id="p68020973217"><a name="p68020973217"></a><a name="p68020973217"></a>gm：输出的GM地址。</p>
<p id="p1731883210186"><a name="p1731883210186"></a><a name="p1731883210186"></a>co1Local: CO1上的计算结果。</p>
<p id="p183915171207"><a name="p183915171207"></a><a name="p183915171207"></a>dataCopyOutParams：Matmul定义的DataCopyOutParams结构体指针，具体定义如下方代码所示，供用户参考使用。</p>
<p id="p136239555189"><a name="p136239555189"></a><a name="p136239555189"></a>tilingPtr: 用户使用<a href="SetUserDefInfo.md">SetUserDefInfo</a>设置的tiling参数地址。</p>
<p id="p109576420196"><a name="p109576420196"></a><a name="p109576420196"></a>dataPtr: 用户使用<a href="SetSelfDefineData.md">SetSelfDefineData</a>设置的计算数据地址。</p>
</td>
</tr>
<tr id="row16990184820318"><td class="cellrowborder" valign="top" width="13.84%" headers="mcps1.2.4.1.1 "><p id="p178021923214"><a name="p178021923214"></a><a name="p178021923214"></a>可自定义左矩阵搬入首地址、搬运块位置、搬运块大小，实现左矩阵从GM搬入L1的功能</p>
</td>
<td class="cellrowborder" valign="top" width="34.23%" headers="mcps1.2.4.1.2 "><p id="p128013953217"><a name="p128013953217"></a><a name="p128013953217"></a>void CopyA1(const LocalTensor&lt;int8_t&gt; &amp;aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)</p>
</td>
<td class="cellrowborder" valign="top" width="51.93%" headers="mcps1.2.4.1.3 "><p id="p114523271558"><a name="p114523271558"></a><a name="p114523271558"></a>aMatrix: 目标L1Buffer地址。</p>
<p id="p669513551251"><a name="p669513551251"></a><a name="p669513551251"></a>gm：左矩阵GM首地址。</p>
<p id="p1145513201868"><a name="p1145513201868"></a><a name="p1145513201868"></a>row、col：搬运块在M、K方向的索引，即在M、K方向上搬运块的序号，序号从0开始。</p>
<p id="p14646749467"><a name="p14646749467"></a><a name="p14646749467"></a>useM、useK：搬运块M、K方向大小，单位为元素个数。通过row、col和useM、useK计算出该搬运块左上角在左矩阵中的地址偏移。</p>
<p id="p119294852"><a name="p119294852"></a><a name="p119294852"></a>tilingPtr: 用户使用<a href="SetUserDefInfo.md">SetUserDefInfo</a>设置的tiling参数地址。</p>
<p id="p109244555"><a name="p109244555"></a><a name="p109244555"></a>dataPtr: 用户使用<a href="SetSelfDefineData.md">SetSelfDefineData</a>设置的计算数据地址。</p>
</td>
</tr>
<tr id="row179905480314"><td class="cellrowborder" valign="top" width="13.84%" headers="mcps1.2.4.1.1 "><p id="p10798991324"><a name="p10798991324"></a><a name="p10798991324"></a>可自定义右矩阵搬入首地址、搬运块位置、搬运块大小，实现右矩阵从GM搬入L1的功能</p>
</td>
<td class="cellrowborder" valign="top" width="34.23%" headers="mcps1.2.4.1.2 "><p id="p479759103215"><a name="p479759103215"></a><a name="p479759103215"></a>void CopyB1(const LocalTensor&lt;int8_t&gt; &amp;bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)</p>
</td>
<td class="cellrowborder" valign="top" width="51.93%" headers="mcps1.2.4.1.3 "><p id="p16739915977"><a name="p16739915977"></a><a name="p16739915977"></a>bMatrix: 目标L1Buffer地址。</p>
<p id="p1273914151477"><a name="p1273914151477"></a><a name="p1273914151477"></a>gm：右矩阵GM首地址。</p>
<p id="p1739815575"><a name="p1739815575"></a><a name="p1739815575"></a>row、col：搬运块在K、N方向的索引，即在K、N方向上搬运块的序号，序号从0开始。</p>
<p id="p273913157717"><a name="p273913157717"></a><a name="p273913157717"></a>useK、useN：搬运块K、N方向大小，单位为元素个数。通过row、col和useK、useN计算出该搬运块左上角在右矩阵中的地址偏移。</p>
<p id="p1673919151979"><a name="p1673919151979"></a><a name="p1673919151979"></a>tilingPtr: 用户使用<a href="SetUserDefInfo.md">SetUserDefInfo</a>设置的tiling参数地址。</p>
<p id="p4739171517713"><a name="p4739171517713"></a><a name="p4739171517713"></a>dataPtr: 用户使用<a href="SetSelfDefineData.md">SetSelfDefineData</a>设置的计算数据地址。</p>
</td>
</tr>
</tbody>
</table>

```
struct DataCopyOutParams {
    uint16_t cBurstNum; //传输数据片段数目
    uint16_t burstLen; //连续传输数据片段长度
    uint16_t srcStride;//源tensor相邻连续数据片段间隔
    uint32_t dstStride; // 目的tensor相邻连续数据片段间隔
    uint16_t oriNSize; // NZ转ND时，源tensorN方向大小
    bool enUnitFlag; // 是否使能UnitFlag
    uint64_t quantScalar; // 量化场景下量化Scalar的值
    uint64_t cbufWorkspaceAddr; //量化场景下量化Tensor地址
}
```

## 约束说明<a name="section1387711715353"></a>

无

## 调用示例<a name="section1311713210369"></a>

完整的使用样例请参考[matmul\_callback样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_callback)。

```
//用户自定义回调函数
void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr);

AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
uint64_t tilingPtr = reinterpret_cast<uint64_t>(tiling);
mm.SetUserDefInfo(tilingPtr); // 设置算子tiling地址，用于回调函数使用
GlobalTensor<SrcT> dataGM; // 保存有回调函数需使用的计算数据的GM
uint64_t dataGMPtr = reinterpret_cast<uint64_t>(dataGM.address_);
mm.SetSelfDefineData(dataGMPtr); // 设置需要的计算数据或在GM上存储的数据地址等信息，用于回调函数使用
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll();
mm.End();
```

