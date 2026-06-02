# Utils API<a name="ZH-CN_TOPIC_0000002523311596"></a>

Ascend C开发提供了丰富的通用工具类，涵盖标准库、平台信息获取、上下文构建、运行时编译及日志输出等功能，支持开发者高效实现算子开发与性能优化。

-   [C++标准库API](Ascend-C-API列表.md#table99801554584)：提供算法、数学函数、容器函数等C++标准库函数。
-   [平台信息获取API](Ascend-C-API列表.md#table32991747162610)：提供获取平台信息的功能，比如获取硬件平台的核数等信息。
-   [log API](Ascend-C-API列表.md#table1514223372716)：提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC\_CPU\_LOG\_XXX接口来输出相关内容。
-   [调测接口](Ascend-C-API列表.md#table17826192512135)：SIMT VF调试场景下使用的相关接口。

