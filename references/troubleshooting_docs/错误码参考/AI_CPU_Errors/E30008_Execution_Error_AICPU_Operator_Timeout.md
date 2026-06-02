# E30008 Execution_Error_AICPU_Operator_Timeout

**页面ID:** atlaserrorcode_15_0132  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/atlaserrorcode_15_0132.html

---

#### Symptom

AI CPU operator execution timed out.

#### Possible Cause

1. For a GetNext operator, its preprocessing time may be too long.
2. For a custom operator, it contains an ultra-large loop in the implementation logic or its input and output shapes are too large.
3. The input and output shapes of a built-in operator are too large.

#### Solution

1. For a GetNext operator, check its preprocessing or use the aclrtSetOpExecuteTimeOut interface to adjust the timeout.
2. For a custom operator, ensure that the logic design is proper or modify the shape.
3. If the input and output shapes are too large, modify the shape or use the aclrtSetOpExecuteTimeOut interface to adjust the timeout.
