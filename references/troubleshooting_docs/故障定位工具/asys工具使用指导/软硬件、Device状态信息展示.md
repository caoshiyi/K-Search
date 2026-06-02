# 软硬件、Device状态信息展示

**页面ID:** troubleshooting_0503  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0503.html

---

- **功能说明**

收集安装包版本信息、Device温度、功率等。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **命令****格式**

```
**asys info -r=**"status" **-d=***deviceId*
```

- **参数说明**

  - **r**：必选参数，指定需展示的信息类型，支持如下取值：

    - status：显示device的信息，包含芯片型号、温度、健康状态、CPU和AI Core信息等。
    - software：显示Host的软件信息，包含系统和内核版本、CANN包版本等信息。
    - hardware：显示Host和Device的硬件信息，包括Host的CPU型号&核数、内存容量和硬盘容量，Device的NPU个数&型号，AI CPU/AI Core/AI Vector个数等信息。

  - **d**：可选参数，指定需要展示信息的deviceId，不设置该参数，默认展示device 0的信息, 仅-r=status时有效。

- **使用示例**

```
**asys info -r=**"status" **-d=***0*
```

- **输出示例**各产品型号的输出信息有所不同，请以实际输出信息为准。

```
+----------------------------------+------------------------+
 | Device ID: 0                     | INFORMATION            |
 +==================================+========================+
 | Chip Name                        | Ascend *xxxxxxxxxx*      |
 | Power (W)                        | 1021                   |
 | Temperature (C)                  | 55                     |
 | health                           | Healthy                |
 +--- CPU Information --------------+------------------------+
 | AI CPU Count                     | 6                      |
 | AI CPU Usage (%)                 | 0                      |
 | Control CPU Count                | 1                      |
 | Control CPU Usage (%)            | 1                      |
 | Control CPU Frequency (MHZ)      | 2000                   |
 +--- AI Core Information ----------+------------------------+
 | AI Core Count                    | 20                     |
 | AI Core Usage (%)                | 0                      |
 | AI Core Frequency (MHZ)          | 800                    |
 | AI Core Voltage (MV)             | 900                    |
 +--- Memory Information -----------+------------------------+
 | HBM Total (MB)                   | 65536                  |
 | HBM Used (MB)                    | 3382.52                |
 | HBM Bandwidth Usage (%)          | 0                      |
 | HBM Frequency (MHZ)              | 1600                   |
 +----------------------------------+------------------------+
```
