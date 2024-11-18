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
    print(solve())
