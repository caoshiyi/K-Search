# LogSoftMax Tiling<a name="ZH-CN_TOPIC_0000002523304340"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧LogSoftMax接口的计算需要开发者预留/申请临时空间，以下接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小，并调用LogSoftMaxTilingFunc函数获取reduceSize，splitSize等参数，作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最大/最小临时空间的接口

    ```
    uint32_t GetLogSoftMaxMaxTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
    ```

    ```
    uint32_t GetLogSoftMaxMinTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
    ```

-   Tiling计算接口

    ```
    void LogSoftMaxTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::LogSoftMaxTiling& softmaxTiling)
    ```

    ```
    void LogSoftMaxTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::LogSoftMaxTiling& softmaxTiling)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  GetLogSoftMaxMaxTmpSize/GetLogSoftMaxMinTmpSize接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row202622311190"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p12634311919"><a name="p12634311919"></a><a name="p12634311919"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p1726311311495"><a name="p1726311311495"></a><a name="p1726311311495"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p681410174421"><a name="p681410174421"></a><a name="p681410174421"></a>是否复用源操作数输入的空间，与LogSoftMax接口一致。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  LogSoftMaxTilingFunc接口参数列表

<a name="table1594020241897"></a>
<table><thead align="left"><tr id="row9940182419919"><th class="cellrowborder" valign="top" width="20.03%" id="mcps1.2.4.1.1"><p id="p1594012414910"><a name="p1594012414910"></a><a name="p1594012414910"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p894010241199"><a name="p894010241199"></a><a name="p894010241199"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p129406241899"><a name="p129406241899"></a><a name="p129406241899"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row1294019241791"><td class="cellrowborder" valign="top" width="20.03%" headers="mcps1.2.4.1.1 "><p id="p14652181315171"><a name="p14652181315171"></a><a name="p14652181315171"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1665261318173"><a name="p1665261318173"></a><a name="p1665261318173"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p3652913101719"><a name="p3652913101719"></a><a name="p3652913101719"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row159411241793"><td class="cellrowborder" valign="top" width="20.03%" headers="mcps1.2.4.1.1 "><p id="p746515195174"><a name="p746515195174"></a><a name="p746515195174"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1065211310173"><a name="p1065211310173"></a><a name="p1065211310173"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p8652171301712"><a name="p8652171301712"></a><a name="p8652171301712"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row69415249915"><td class="cellrowborder" valign="top" width="20.03%" headers="mcps1.2.4.1.1 "><p id="p143823307171"><a name="p143823307171"></a><a name="p143823307171"></a>localWorkSpaceSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p5652171313175"><a name="p5652171313175"></a><a name="p5652171313175"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p1565281315174"><a name="p1565281315174"></a><a name="p1565281315174"></a>输入的临时空间大小。</p>
</td>
</tr>
<tr id="row1599974915171"><td class="cellrowborder" valign="top" width="20.03%" headers="mcps1.2.4.1.1 "><p id="p999916498171"><a name="p999916498171"></a><a name="p999916498171"></a>softmaxTiling</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p499984912172"><a name="p499984912172"></a><a name="p499984912172"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p129991849161719"><a name="p129991849161719"></a><a name="p129991849161719"></a>传递到kernel侧使用的Tiling参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetLogSoftMaxMaxTmpSize/GetLogSoftMaxMinTmpSize接口返回值为最大/最小临时空间。

LogSoftMaxTilingFunc接口无返回值**。**

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
     std::vector<int64_t> srcDims = {outter, inner};
     ge::Shape shape(srcDims);
     const uint32_t tmpsize = AscendC::GetLogSoftMaxMaxTmpSize(shape, dtypesize, false);
     AscendC::LogSoftMaxTilingFunc(shape, dtypesize, tmpsize, tiling.logSoftmaxTilingData);
     ...
}
```

