# CubeResGroupHandle使用说明<a name="ZH-CN_TOPIC_0000002523343606"></a>

CubeResGroupHandle用于在分离模式下对AI Core计算资源分组。分组后，开发者可以对不同的分组指定不同的计算任务。一个AI Core分组可包含多个AIV和AIC，AIV和AIC之间采取Client和Server架构进行任务处理。AIV为Client，每一个Cube计算任务为一个消息，AIV发送消息至消息队列，AIC作为Server，遍历消息队列的消息，根据消息类型及内容执行对应的计算任务。一个CubeResGroupHandle中可以有一个或多个AIC，同一个AIC只能属于一个CubeResGroupHandle，AIV无此限制，即同一个AIV可以属于多个CubeResGroupHandle。

如下图所示，CubeResGroupHandle1中有2个AIC，10个AIV，AIC为Block0和Block1。其中Block0与Queue0、Queue1、Queue2、Queue3、Queue4进行通信，Block1与Queue 5、Queue 6、Queue 7、Queue 8、Queue9进行通信。每一个消息队列对应一个AIV，消息队列的深度固定为4，即一次性最多可以容纳4个消息。CubeResGroupHandle2的消息队列个数为12，表明有12个AIV。CubeResGroupHandle的消息处理顺序如CubeResGroupHandle2中黑色箭头所示。

**图 1**  基于CubeResGroupHandle的AI Core计算资源分组通信示意图<a name="fig1086562617536"></a>  
<!-- img2text -->
```text
┌──────────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ CubeResGroupHandle   │ Block 0 │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│ 1                    ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│                      │         ├─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│                      │         ├─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
├──────────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │ Block 1 │ Queue 5 │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                      │         ├─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 5 │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                      │         ├─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 5 │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                      │         ├─────────┼─────────┼─────────┼─────────┼─────────┤
│                      │         │ Queue 5 │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
└──────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                     │
                                     ↓

┌──────────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┐
│ CubeResGroupHandle2  │ Block 2 │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4  │ Queue 5 │
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4  │ Queue 5 │
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4  │ Queue 5 │
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 0 │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4  │ Queue 5 │
├──────────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │ Block 3 │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │ Queue 10 │ Queue 11│
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │ Queue 10 │ Queue 11│
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │ Queue 10 │ Queue 11│
│                      ├─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│                      │         │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │ Queue 10 │ Queue 11│
└──────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────┴─────────┘
```

说明:
- 上半部分为 CubeResGroupHandle 1：Block 0 对应 Queue 0～Queue 4，Block 1 对应 Queue 5～Queue 9。
- 下半部分为 CubeResGroupHandle2：Block 2 对应 Queue 0～Queue 5，Block 3 对应 Queue 6～Queue 11。
- 每个 Queue 在各自分组中均重复出现 4 行，表示消息队列深度为 4。
- 图中从上半部分指向下半部分的竖向箭头表示 AIV/Queue 可与不同的 CubeResGroupHandle 关联。
- 下半部分 Block 2 区域上方有 3 条手绘斜向/横向箭头线，跨越 Queue 0～Queue 5；由于原图为覆盖在表格上的复杂标注，未在框图中强行复现其精确走线。

基于CubeResGroupHandle实现AI Core计算资源分组步骤如下：

1.  创建AIC上所需要的计算对象类型。
2.  创建通信区域描述[KfcWorkspace](KfcWorkspace.md)，用于记录通信消息Msg的地址分配。
3.  自定义消息结构体，用于通信。
4.  自定义回调计算结构体，根据实际业务场景实现Init函数和Call函数。
5.  创建CubeResGroupHandle。
6.  绑定AIV到CubeResGroupHandle。
7.  收发消息。
8.  AIV退出消息队列。

下文仅提供示例代码片段，更多完整样例请参考[CubeGroup样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/2_features/12_cube_group/CubeGroupCustom)。

1.  <a name="li27691150733"></a>创建AIC上所需要的计算对象类型。

    用户根据实际需求，自定义AIC所需要的计算对象类型，或者高阶API已提供的Matmul类型。例如，创建Matmul类型如下，其中A\_TYPE、B\_TYPE、 C\_TYPE、BIAS\_TYPE、CFG\_NORM等含义请参考[Matmul模板参数](Matmul模板参数.md)。

    ```
    // A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM根据实际需求场景构造
    using MatmulApiType = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, CFG_NORM>;
    ```

2.  创建KfcWorkspace。

    使用[KfcWorkspace](KfcWorkspace.md)管理不同CubeResGrouphandle的消息通信区的划分。

    ```
    // 创建KfcWorkspace对象前，需要对该workspaceGM清零
    KfcWorkspace desc(workspaceGM);
    ```

3.  自定义消息结构体。

    用户需要自行构造消息结构体[CubeMsgBody](#table189051237164018)，用于AIV向AIC发送通信消息。构造的CubeMsgBody必须64字节对齐，该结构体最前面需要定义2字节的CubeGroupMsgHead，使消息收发机制正常运行，CubeGroupMsgHead结构定义请参考[表2](#table77221554135216)。除2字节的CubeGroupMsgHead外，其余参数根据业务需求自行构造。

    **表 1**  CubeMsgBody消息结构体

    <a name="table189051237164018"></a>
    <table><thead align="left"><tr id="row990543774012"><th class="cellrowborder" valign="top" width="19.6%" id="mcps1.2.3.1.1"><p id="p39051637104010"><a name="p39051637104010"></a><a name="p39051637104010"></a>参数名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="80.4%" id="mcps1.2.3.1.2"><p id="p1290623718401"><a name="p1290623718401"></a><a name="p1290623718401"></a>含义</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row19062378407"><td class="cellrowborder" valign="top" width="19.6%" headers="mcps1.2.3.1.1 "><p id="p2906203720407"><a name="p2906203720407"></a><a name="p2906203720407"></a>CubeMsgBody</p>
    </td>
    <td class="cellrowborder" valign="top" width="80.4%" headers="mcps1.2.3.1.2 "><p id="p7906113754018"><a name="p7906113754018"></a><a name="p7906113754018"></a>用户自定义的消息结构体。结构体名称可自定义，结构体大小需要64字节对齐。</p>
    <a name="screen1555172113494"></a><a name="screen1555172113494"></a><pre class="screen" codetype="Cpp" id="screen1555172113494">// 这里提供64B对齐的结构体示例，用户实际使用时，除CubeGroupMsgHead外，其他参数个数及参数类型可自行构造
    struct CubeMsgBody {
       CubeGroupMsgHead head;  // 2B，需放在结构体最前面, 自定义的CubeMsgBody中，CubeGroupMsgHead的变量名需设置为head，否则会编译报错。
       uint8_t funcID;
       uint8_t skipCnt;
       uint32_t value;
       bool isTransA;
       bool isTransB;
       bool isAtomic;
       bool isLast;                 
       int32_t tailM;              
       int32_t tailN;
       int32_t tailK;               
       uint64_t aAddr;
       uint64_t bAddr;
       uint64_t cAddr;
       uint64_t aGap;
       uint64_t bGap;
    }</pre>
    </td>
    </tr>
    </tbody>
    </table>

    **表 2**  CubeGroupMsgHead结构体参数定义

    <a name="table77221554135216"></a>
    <table><thead align="left"><tr id="row1072214548521"><th class="cellrowborder" valign="top" width="11.91%" id="mcps1.2.3.1.1"><p id="p8722115485218"><a name="p8722115485218"></a><a name="p8722115485218"></a>参数名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="88.09%" id="mcps1.2.3.1.2"><p id="p9723754105215"><a name="p9723754105215"></a><a name="p9723754105215"></a>含义</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1572395414524"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.3.1.1 "><p id="p1372335418525"><a name="p1372335418525"></a><a name="p1372335418525"></a>msgState</p>
    </td>
    <td class="cellrowborder" valign="top" width="88.09%" headers="mcps1.2.3.1.2 "><p id="p972385485211"><a name="p972385485211"></a><a name="p972385485211"></a>表明该位置的消息状态。参数取值如下：</p>
    <a name="ul372310546523"></a><a name="ul372310546523"></a><ul id="ul372310546523"><li>CubeMsgState::FREE：表明该位置还未填写消息，可执行<a href="AllocMessage.md">AllocMessage</a>。</li><li>CubeMsgState::VALID：表明该位置已经含有AIV发送的消息，待AIC接收执行。</li><li>CubeMsgState::QUIT：表明该位置的消息为通知AIC有AIV将退出流程。</li><li>CubeMsgState::FAKE：表明该位置的消息为假消息。在消息合并场景，被跳过处理任务的AIV需要发送假消息，消息合并场景请参考<a href="PostFakeMsg.md">PostFakeMsg</a>中的介绍。</li></ul>
    </td>
    </tr>
    <tr id="row472325455211"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.3.1.1 "><p id="p37231654135220"><a name="p37231654135220"></a><a name="p37231654135220"></a>aivID</p>
    </td>
    <td class="cellrowborder" valign="top" width="88.09%" headers="mcps1.2.3.1.2 "><p id="p17723135445214"><a name="p17723135445214"></a><a name="p17723135445214"></a>发送消息的AIV的序号。</p>
    </td>
    </tr>
    </tbody>
    </table>

4.  自定义回调计算结构体，根据实际业务场景实现Init函数和Call函数。

    ```
    template<class MatmulApiCfg, class CubeMsgBody>
    struct NormalCallbackFuncs {
        __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
          // 用户自行实现逻辑
        };
    
        __aicore__ inline static void Init(NormalCallbackFuncs<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM){
           // 用户自行实现逻辑
        };
       
    };
    ```

    计算逻辑结构体的模板参数请参考[表3](#table18865397406)。

    **表 3**  模板参数说明

    <a name="table18865397406"></a>
    <table><thead align="left"><tr id="row888719393401"><th class="cellrowborder" valign="top" width="20.82%" id="mcps1.2.3.1.1"><p id="p18887193944012"><a name="p18887193944012"></a><a name="p18887193944012"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="79.17999999999999%" id="mcps1.2.3.1.2"><p id="p788773984016"><a name="p788773984016"></a><a name="p788773984016"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row48872393405"><td class="cellrowborder" valign="top" width="20.82%" headers="mcps1.2.3.1.1 "><p id="p10887173915407"><a name="p10887173915407"></a><a name="p10887173915407"></a>MatmulApiCfg</p>
    </td>
    <td class="cellrowborder" valign="top" width="79.17999999999999%" headers="mcps1.2.3.1.2 "><p id="p988893994019"><a name="p988893994019"></a><a name="p988893994019"></a>用户自定义的AIC上计算所需要对象的数据类型，参考<a href="#li27691150733">步骤1</a>，该模板参数必须填入。</p>
    </td>
    </tr>
    <tr id="row1888739204012"><td class="cellrowborder" valign="top" width="20.82%" headers="mcps1.2.3.1.1 "><p id="p188883391405"><a name="p188883391405"></a><a name="p188883391405"></a>CubeMsgBody</p>
    </td>
    <td class="cellrowborder" valign="top" width="79.17999999999999%" headers="mcps1.2.3.1.2 "><p id="p1688823914010"><a name="p1688823914010"></a><a name="p1688823914010"></a><a href="#table189051237164018">用户自定义的消息结构体</a>，该模板参数必须填入。</p>
    </td>
    </tr>
    </tbody>
    </table>

    用户自定义回调计算结构体中需要包含固定的Init函数和Call函数，函数原型如下所示。其中，Init函数的参数说明请参考[表4](#zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446)，Call函数的参数说明请参考[表5](#table9997952179)。

    ```
    // 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
    __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &myCallBack, MatmulApiCfg &mm, GM_ADDR tilingGM){
         // 用户自行实现内部逻辑
    }
    ```

    **表 4**  Init函数参数说明

    <a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="19.15%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.04%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="72.81%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>myCallBack</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.4.1.2 "><p id="p19741912147"><a name="p19741912147"></a><a name="p19741912147"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="72.81%" headers="mcps1.2.4.1.3 "><p id="p17175015525"><a name="p17175015525"></a><a name="p17175015525"></a>用户自定义的带<a href="#table18865397406">模板参数</a>的回调计算结构体。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001526206862_row1239183183016"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_p223953193015"><a name="zh-cn_topic_0000001526206862_p223953193015"></a><a name="zh-cn_topic_0000001526206862_p223953193015"></a>mm</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001526206862_p7239938308"><a name="zh-cn_topic_0000001526206862_p7239938308"></a><a name="zh-cn_topic_0000001526206862_p7239938308"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="72.81%" headers="mcps1.2.4.1.3 "><p id="p1117515154210"><a name="p1117515154210"></a><a name="p1117515154210"></a>AIC上计算对象，多为Matmul对象。</p>
    </td>
    </tr>
    <tr id="row9374154371313"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p11374343181311"><a name="p11374343181311"></a><a name="p11374343181311"></a>tilingGM</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.4.1.2 "><p id="p146153901420"><a name="p146153901420"></a><a name="p146153901420"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="72.81%" headers="mcps1.2.4.1.3 "><p id="p517518151222"><a name="p517518151222"></a><a name="p517518151222"></a>用户传入的tiling指针。</p>
    </td>
    </tr>
    </tbody>
    </table>

    ```
    // 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
    __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
            // 用户自行实现内部逻辑
    }
    ```

    **表 5**  Call函数参数说明

    <a name="table9997952179"></a>
    <table><thead align="left"><tr id="row599775171716"><th class="cellrowborder" valign="top" width="18.62813718628137%" id="mcps1.2.4.1.1"><p id="p6997858178"><a name="p6997858178"></a><a name="p6997858178"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.30916908309169%" id="mcps1.2.4.1.2"><p id="p16997165131717"><a name="p16997165131717"></a><a name="p16997165131717"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="73.06269373062693%" id="mcps1.2.4.1.3"><p id="p1799755121718"><a name="p1799755121718"></a><a name="p1799755121718"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row119971056171"><td class="cellrowborder" valign="top" width="18.62813718628137%" headers="mcps1.2.4.1.1 "><p id="p59971755174"><a name="p59971755174"></a><a name="p59971755174"></a>mm</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.30916908309169%" headers="mcps1.2.4.1.2 "><p id="p1599755171712"><a name="p1599755171712"></a><a name="p1599755171712"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.06269373062693%" headers="mcps1.2.4.1.3 "><p id="p39982520173"><a name="p39982520173"></a><a name="p39982520173"></a>AIC上计算对象，多为Matmul对象。</p>
    </td>
    </tr>
    <tr id="row19985531717"><td class="cellrowborder" valign="top" width="18.62813718628137%" headers="mcps1.2.4.1.1 "><p id="p16998135201718"><a name="p16998135201718"></a><a name="p16998135201718"></a>rcvMsg</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.30916908309169%" headers="mcps1.2.4.1.2 "><p id="p49981151174"><a name="p49981151174"></a><a name="p49981151174"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.06269373062693%" headers="mcps1.2.4.1.3 "><p id="p899810515175"><a name="p899810515175"></a><a name="p899810515175"></a>用户自定义的消息结构体指针。</p>
    </td>
    </tr>
    <tr id="row599815501714"><td class="cellrowborder" valign="top" width="18.62813718628137%" headers="mcps1.2.4.1.1 "><p id="p0998165161717"><a name="p0998165161717"></a><a name="p0998165161717"></a>handle</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.30916908309169%" headers="mcps1.2.4.1.2 "><p id="p179981351176"><a name="p179981351176"></a><a name="p179981351176"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.06269373062693%" headers="mcps1.2.4.1.3 "><p id="p89981571714"><a name="p89981571714"></a><a name="p89981571714"></a>分组管理Handle，用户调用其接口进行收发消息，释放消息等。</p>
    </td>
    </tr>
    </tbody>
    </table>

    某算子的回调计算结构体的代码示例如下。

    ```
    // 用户自定义的回调计算逻辑
    template<class MatmulApiCfg, typename CubeMsgBody>
    struct MyCallbackFunc
    {
        template<int32_t funcId>
        __aicore__ inline static typename IsEqual<funcId, 0>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
        {
            GlobalTensor<int64_t> msgGlobal;
            msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
            using SrcAT = typename MatmulApiCfg::AType::T;
            auto skipNum = 0;
            for (int i = 0; i < skipNum + 1; ++i)
            {
                auto tmpId = handle.FreeMessage(rcvMsg + i); // msgPtr process is complete
            }
            handle.SetSkipMsg(skipNum);
        }
        template<int32_t funcId>
        __aicore__ inline static typename IsEqual<funcId, 1>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
        {
            GlobalTensor<int64_t> msgGlobal;
            msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
            using SrcAT = typename MatmulApiCfg::AType::T;
            LocalTensor<SrcAT> tensor_temp;
            auto skipNum = 3;
            auto tmpId = handle.FreeMessage(rcvMsg, CubeMsgState::VALID);
            for (int i = 1; i < skipNum + 1; ++i)
            {
                auto tmpId = handle.FreeMessage(rcvMsg + i, CubeMsgState::FAKE);
            }
            handle.SetSkipMsg(skipNum); // notify the cube not to process
        }
        __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
        {
            if (rcvMsg->funcId == 0)
            {
                CubeGroupCallBack<0> (mm, rcvMsg, handle);
            }
            else if(rcvMsg->funcId == 1)
            {
                CubeGroupCallBack<1> (mm, rcvMsg, handle);
            }
        }
        __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM)
        {
            auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
            auto tempTiling = (uint32_t*)&(foo.tiling);
            for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
            {
                *tempTiling = *tempTilingGM;
            }
            mm.SetSubBlockIdx(0);
            mm.Init(&foo.tiling, GetTPipePtr());
        }
        TCubeTiling tiling;
    };
    ```

5.  <a name="li355132105919"></a>创建CubeResGroupHandle。

    用户使用[CreateCubeResGroup](CreateCubeResGroup.md)接口创建一个或多个CubeResGroupHandle。

    ```
    /* 
     * groupID为用户自定义的CreateCubeResGroup的groupID
     * MatmulApiType为定义好的AIC上计算对象的类型
     * MyCallbackFunc为定义好的自定义回调计算结构体
     * CubeMsgBody为自定义消息结构体
     * desc为用户初始化好的通信区域描述
     * groupID为1，blockStart为0，blockSize为12，msgQueueSize为48，tilingGm为指针，存储了用户在AIC上所需要的tiling信息
    */
    auto handle =  AscendC::CreateCubeResGroup<groupID, MatmulApiType, MyCallbackFunc, CubeMsgBody>(desc, 0, 12, 48, tilingGM);
    ```

6.  绑定AIV到CubeResGroupHandle。

    绑定AIV和消息队列序号。注意：消息队列序号queIdx小于该CubeGroupHandle的消息队列总数，每个AIV需要传入不同的queIdx。handle为[步骤5](#li355132105919)中CreateCubeResGroup创建的CubeResGroupHandle对象。

    ```
    handle.AssignQueue(queIdx);
    ```

7.  AIV发消息。

    用户调用[AllocMessage](AllocMessage.md),  [PostMessage](PostMessage.md)等接口进行消息的收发。其中，调用AllocMessage获取消息结构体指针，通过PostMessage发送消息，在消息合并场景调用[PostFakeMessage](PostFakeMsg.md)发送假消息，示例如下。

    ```
    CubeGroupMsgHead head = {CubeMsgState::VALID, (uint8_t)queIdx};
    CubeMsgBody aCubeMsgBody {head, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    CubeMsgBody bCubeMsgBody {head, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    auto offset = 0;
    if (GetBlockIdx() == 0)
    {
        auto msgPtr = handle.template AllocMessage(); // alloc for queue space
        offset = handle.template PostMessage(msgPtr, bCubeMsgBody); // post true msgPtr
        bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
    }
    else if (GetBlockIdx() < 4)
    {
        auto msgPtr = handle.AllocMessage();
        offset = handle.PostFakeMsg(msgPtr); // post fake msgPtr
        bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
    }
    else
    {
        auto msgPtr = handle.template AllocMessage();
        offset = handle.template PostMessage(msgPtr, aCubeMsgBody);
        bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
    }
    ```

8.  AIV退出消息队列。

    调用AllocMessage获取消息结构体指针后，通过SendQuitMsg发送当前消息队列退出。

    ```
    auto msgPtr = handle.AllocMessage();        // 获取消息空间指针msgPtr
    handle.SetQuit(msgPtr);              // 发送退出消息
    ```

