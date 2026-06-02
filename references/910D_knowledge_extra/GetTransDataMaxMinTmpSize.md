# GetTransDataMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002523304818"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧TransData接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。该接口**最大临时空间当前等于最小临时空间**。

## 函数原型<a name="section620mcpsimp"></a>

```
bool GetTransDataMaxMinTmpSize(const platform_ascendc::PlatformAscendC& platform, const ge::Shape& srcShape, const ge::Shape& dstShape,const ge::DataType dataType, const TransDataConfig &config, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p178521516183615"><a name="p178521516183615"></a><a name="p178521516183615"></a>platform</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p68691954161015"><a name="p68691954161015"></a><a name="p68691954161015"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row192692403617"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p327132493617"><a name="p327132493617"></a><a name="p327132493617"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p1927192415365"><a name="p1927192415365"></a><a name="p1927192415365"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p112792483615"><a name="p112792483615"></a><a name="p112792483615"></a>输入源操作数的shape大小，参数取值与TransData接口的params.srcLayout参数中的shape信息保持一致。</p>
</td>
</tr>
<tr id="row82581226123614"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p202588260362"><a name="p202588260362"></a><a name="p202588260362"></a>dstShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p13258132653620"><a name="p13258132653620"></a><a name="p13258132653620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p1125817263360"><a name="p1125817263360"></a><a name="p1125817263360"></a>输出目的操作数的shape大小，参数取值与TransData接口的params.dstLayout参数中的shape信息保持一致。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p170935062414"><a name="p170935062414"></a><a name="p170935062414"></a>输入的数据类型，ge::DataType类型，当前只支持half/float/uint16_t/int16_t数据类型的输入。</p>
</td>
</tr>
<tr id="row0316141211351"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p19316131273512"><a name="p19316131273512"></a><a name="p19316131273512"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p331619124359"><a name="p331619124359"></a><a name="p331619124359"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>数据格式转换的场景，参数取值与TransData接口的config参数保持一致。当前支持的转换场景有：NCDHW -&gt; NDC1HWC0、NDC1HWC0 -&gt; NCDHW、NCDHW -&gt; FRACTAL_Z_3D、FRACTAL_Z_3D -&gt; NCDHW。TransDataConfig类型，具体定义如下。</p>
<a name="screen1119148131414"></a><a name="screen1119148131414"></a><pre class="screen" codetype="Cpp" id="screen1119148131414">struct TransDataConfig {
    DataFormat srcFormat;
    DataFormat dstFormat;
};

enum class DataFormat : uint8_t {
    ND = 0,
    NZ,
    NCHW,
    NC1HWC0,
    NHWC,
    NCDHW,
    NDC1HWC0,
    FRACTAL_Z_3D,
};</pre>
</td>
</tr>
<tr id="row20919154124316"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>TransData接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row124201637134315"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1369580162813"><a name="p1369580162813"></a><a name="p1369580162813"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p176953052810"><a name="p176953052810"></a><a name="p176953052810"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>TransData接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetTransDataMaxMinTmpSize返回值为true/false，true表示成功获取TransData接口内部计算需要的最大和最小临时空间大小；false表示获取失败。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

```
// 输入shape为(1,16,2,4,4)NCDHW转换为输出shape(1,2,1,4,4,16)NDC1HWC0;算子输入的数据类型为half
uint32_t maxSize;
uint32_t minSize;
int32_t n = 1, c = 16, d = 2, h = 4, w = 4, c1 = 1, c0 = 16;
auto ncdhwShape = ge::Shape({ n, c, d, h, w });
auto ndc1hwc0Shape = ge::Shape({ n, d, c1, h, w, c0});
auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
TransDataConfig config = {DataFormat::NCDHW, DataFormat::NDC1HWC0};
bool ret = GetTransDataMaxMinTmpSize(plat, ncdhwShape, ndc1hwc0Shape, ge::DataType::DT_FLOAT16, config, maxSize, minSize);
```

