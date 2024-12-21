# import ipaddress

# class Iptables:
#     def __init__(self):
#         # Initialize an empty dictionary to store chains and their rules
#         self.chains = {}

#     def insert_rule(self, chain, ip_cidr, action):
#         # Insert rule at the beginning of the chain
#         if chain not in self.chains:
#             self.chains[chain] = []
#         self.chains[chain].insert(0, (ip_cidr, action))

#     def append_rule(self, chain, ip_cidr, action):
#         # Append rule at the end of the chain
#         if chain not in self.chains:
#             self.chains[chain] = []
#         self.chains[chain].append((ip_cidr, action))

#     def delete_rule(self, chain, ip_cidr):
#         # Remove the rule from the chain
#         if chain in self.chains:
#             self.chains[chain] = [rule for rule in self.chains[chain] if rule[0] != ip_cidr]

#     def goto_chain(self, chain_from, ip_cidr, chain_to):
#         # Insert a Goto rule at the end of the chain (3 elements: ip_cidr, 'G', chain_to)
#         if chain_from not in self.chains:
#             self.chains[chain_from] = []
#         self.chains[chain_from].append((ip_cidr, "G", chain_to))

#     def match_ip(self, chain, ip):
#         while chain in self.chains:
#             for rule in self.chains[chain]:
#                 rule_ip_cidr = rule[0]
#                 action = rule[1]

#                 # Convert to an ip_network object for CIDR comparison
#                 network = ipaddress.ip_network(rule_ip_cidr, strict=False)
                
#                 if ip in network:
#                     if action == "A" or action == "R":
#                         # Return Accept or Reject if matched
#                         return action
#                     elif action == "G" and len(rule) == 3:
#                         # Jump to the next chain using Goto
#                         chain = rule[2]
#                         break  # Immediately break and match in the new chain
#             else:
#                 # If no match is found in the current chain, return 'U' (Unknown)
#                 return 'U'
#         return 'U'  # If chain does not exist or no match found

# def process_iptables_commands(n, commands):
#     iptables = Iptables()
#     results = []

#     for command in commands:
#         parts = command.split()

#         if parts[0] == 'I':
#             # Insert rule
#             chain_name, ip_cidr, action = parts[1], parts[2], parts[3]
#             iptables.insert_rule(chain_name, ip_cidr, action)
        
#         elif parts[0] == 'A':
#             # Append rule
#             chain_name, ip_cidr, action = parts[1], parts[2], parts[3]
#             iptables.append_rule(chain_name, ip_cidr, action)

#         elif parts[0] == 'D':
#             # Delete rule
#             chain_name, ip_cidr = parts[1], parts[2]
#             iptables.delete_rule(chain_name, ip_cidr)

#         elif parts[0] == 'G':
#             # Goto chain
#             chain_name_from, ip_cidr, chain_name_to = parts[1], parts[2], parts[4]
#             iptables.goto_chain(chain_name_from, ip_cidr, chain_name_to)

#         elif parts[0] == 'M':
#             # Match IP
#             ip = ipaddress.ip_address(parts[1])
#             # Always start matching from the 'c0' chain
#             result = iptables.match_ip('c0', ip)
#             results.append(result)
    
#     return results
# # Example test case:
# # n = 7
# # commands = [
# #     "A c0 10.1.0.0/24 A",        # Rule 1: Accept 10.1.0.0/24 in c0
# #     "A c0 192.168.0.0/16 R",     # Rule 2: Reject 192.168.0.0/16 in c0
# #     "A c1 172.16.0.0/12 A",      # Rule 3: Accept 172.16.0.0/12 in c1
# #     "A c0 172.16.0.0/12 G c1",   # Rule 4: Goto c1 for 172.16.0.0/12 in c0
# #     "M 10.1.0.5",                # Query 1: Match IP 10.1.0.5 (Accept from rule 1)
# #     "M 192.168.1.1",             # Query 2: Match IP 192.168.1.1 (Reject from rule 2)
# #     "M 172.16.1.1"               # Query 3: Match IP 172.16.1.1 (Goto c1, then Accept from rule 3)
# # ]

# n = 2
# commands = [
#     "A c0 192.168.1.0/24 R",
#     "M 192.168.1.20"
# ]
# # Process the commands
# output = process_iptables_commands(n, commands)
# for res in output:
#     print(res)


'''
某在线游戏运营商对旗下的游戏后台数据做了快照，从中可以得到用户的登陆时间信息，希望据此快速得到历史最大的用户同时在线人数。
算法输入：每行英文逗号分隔的：一用户上线时间（简化为整数，不小于0）,下线时间（简化为整数，小于10000）；空行结束，如：
0,8
12,15
4,9
6,9
14,20
8,10
10,20

算法输出：最大同时在线人数，如：4
'''

def solve():
    ls = []
    N = int(input())
    for _ in range(N):
        ls.append(list(map(int, input().split(','))))
    ls.sort(key=lambda x: x[0])

    result = 1
    count = 1
    for i in range(1, N):
        if ls[i][0] <= ls[i - 1][1]:
            count += 1
            ls[i][1] = min(ls[i][1], ls[i - 1][1])
            result = max(result, count)
        else:
            count = 1
    return result



"""
【累加序列】
给定一个字符串数字，编写一个算法来判断组成它的数字可以形成累加序列。
一个有效的累加序列必须**至少**包含3个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
说明：字符串只包含`'0'-'9'`的字符，累加序列里的数不会以0开头，所以不会出现`1, 2, 03`或者`1, 02, 3`的情况。
**Example1:**
```
Input: "112358"
Output: true
Explanation: The digits can form an additive sequence: 1, 1, 2, 3, 5, 8. 
             1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
```
**Example 2:**
```
Input: "199100199"
Output: true
Explanation: The additive sequence is: 1, 99, 100, 199. 
             1 + 99 = 100, 99 + 100 = 199
```
**Constraints:**
`1 <= num.length <= 35`
"""
def solve():
    s = input()
    s_ls = list(s)
    n = len(s)

    # startSec = 1
    # endSec = 1
    for startSec in range(1, n - 1):
        for endSec in range(startSec, n - 1):
            if s[startSec] == '0' and startSec != endSec:
                break
            # num1 = int(s[:startSec])
            # num2 = int(s[startSec:endSec + 1])
            # total = num1 + num2
            # total_s = str(total)
            # if endSec + len(total_s) >= n:
            #     break
            if isValid(s, startSec, endSec):
                return True
                
    return False

def isValid(s, startSec, endSec):
    n = len(s)
    startFirst, endFirst = 0, startSec - 1
    while startSec <= n - 1:
        num1 = int(s[startFirst:endFirst + 1])
        num2 = int(s[startSec:endSec + 1])
        num3 = num1 + num2
        num3_s = str(num3)
        startThird = endSec + 1
        endThird = endSec + len(num3_s)
        print(num1, num2, num3)
        print(startThird, endThird)
        if endThird >= n or int(s[startThird:endThird + 1]) != num3:
            break
        if endThird == n - 1:
            return True
        startFirst, endFirst = startSec, endSec
        startSec, endSec = startThird, endThird
    return False

if __name__ == "__main__":
    # print(solve())
    import numpy as np

    # a = np.array([-1,-1,1,2,3,1,4,5,6])
    # b = np.where(a==-1)[0]
    # # print(np.split(a, b))

    # a = np.array([3,2,3,4,3])
    # print(len(np.where(a == 2)[0]))


    A = np.array([1,2,3,-1,5,4,-1,6,9,7,8,10,-1])
    B = np.array([9,7,8,-1,5,4,6,-1,1,2,3,-1,2])

    ids = np.where(B==-1)[0]
    index = np.where(ids==7)[0]
    assert len(index) == 1
    index = index[0]
    xs = np.where(A==-1)[0]
    x = xs[index]
    # print(x)
    p1, p2 = np.random.randint(0, 10, 2)
    print(p1, p2)