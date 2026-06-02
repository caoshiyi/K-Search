# EI0012 Execution_Error_SDMA

**页面ID:** atlaserrorcode_15_0348  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/atlaserrorcode_15_0348.html

---

#### Symptom

SDMA memory copy task exception occurred. Remote rank: [%s]. Base information: [%s]. Task information: [%s]. Communicator information: [%s].

#### Possible Cause

1. Network connection exception occurred during the SDMA task execution.
2. The peer process exits abnormally.
3. The input or output memory address is not allocated, the actual allocated size is smaller than the input data size, or the memory is freed before the operator execution is complete.

#### Solution

1. Check whether the network link is abnormal during the execution.
2. Check whether a process in the cluster exits before an error is reported. If yes, locate the cause of the process exit.
3. Check whether the size of the input/output memory passed to the communication operator meets the expectation, and whether the input/output memory or communicator is freed or destroyed before the operator execution is complete.
