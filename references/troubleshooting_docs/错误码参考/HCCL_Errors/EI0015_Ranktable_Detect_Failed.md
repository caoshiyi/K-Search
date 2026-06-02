# EI0015 Ranktable_Detect_Failed

**页面ID:** atlaserrorcode_15_0351  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/atlaserrorcode_15_0351.html

---

#### Symptom

Failed to collect cluster information of the communicator based on rootInfo detection. Reason: %s.

#### Solution

1. Check whether all ranks in the communicator have delivered the communicator creation interface.
2. Check the connectivity between the host networks of all nodes and the server node.
3. Check whether the HCCL_SOCKET_IFNAME environment variable of all nodes is correctly configured.
4. Increase the timeout by configuring the HCCL_CONNECT_TIMEOUT environment variable.
