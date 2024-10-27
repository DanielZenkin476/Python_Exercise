from itertools import filterfalse
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        st = s.strip()
        if st == "" : return 0
        st = s.lstrip()  # remove whitespaces if present
        curr_char = st[0]
        curr_val = ord(curr_char)
        sign = 1
        leading_zero = False
        res = 0
        if curr_val == 45 or curr_val == 43: # 44-43 = 1 for plus , 44-45 = -1 for minus, the read next
            sign = 44-curr_val
            st = st[1:]
            if st == "": return 0
            curr_char = st[0]
            curr_val = ord(curr_char)
        while(48<=curr_val<=57):
            if curr_val ==48 and leading_zero == False:
                pass
            else:
                leading_zero = True
                curr_int = int(curr_char)
                res = res*10 + curr_int
            st = st[1:]
            if len(st) == 0: break
            curr_char = st[0]
            curr_val = ord(curr_char)
            if res*sign < -2**(31):
                res = 2 **(31)
                break
            if res*sign >(2 ** 31 - 1):
                res = 2 **31 -1
                break
        res = res*sign
        if res  < -2 ** (31):
            res = -2 ** (31)
        if res >(2 ** 31 - 1):
            res = 2 **31 -1
        return (res)

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0 : return False # always not true if negative
        if x == 0 : return True
        # reverse int, chek if coimp is 0
        for_num= x
        num = 0
        while (for_num!= 0):
            num *= 10
            num+= for_num%10
            for_num = for_num // 10
        if num-x == 0:
            return True
        else : return False

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n == 1 or n == 0:
            return 0
        max_mass = 0
        for i in range(0,n):
            if max_mass > height[i]* (n-i-1): pass
            else:
                for j in range(i+1,n):
                    if max_mass > height[i] * (n-i-1): break
                    max_mass = max(max_mass,min(height[i],height[j])*(j-i))
        return  max_mass

    def maxArea2(self,height):
        """
                :type height: List[int]
                :rtype: int
                """
        n = len(height)
        if n == 1 or n == 0:
            return 0
        i = 0
        j = n-1
        max_area = 0
        while (i <j):
            min_piller = min(height[i],height[j])
            max_area = max(max_area,min_piller* (j-i) )
            if min_piller == height[i]: i+=1
            else: j-=1
        return max_area

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        roman_dict = ["I","V","X","L","C","D","M"]
        roman_val = [1,5,10,50,100,500,1000]
        #            0 1  2  3  4   5   6 even need to check, odd not
        st = ''
        i = len(roman_val)-1
        while(num>0):
            if num>= roman_val[i] or i == 0:
                st+= roman_dict[i]
                num-= roman_val[i]
            else:
                if i%2 == 0 and i != 0 :
                    if num >= roman_val[i]-roman_val[i-2] : #for exemp num >= 1000-100 = 900
                        st+= roman_dict[i-2]+roman_dict[i]
                        num-= roman_val[i]-roman_val[i-2]
                if i%2 == 1 :
                    if num >= roman_val[i]-roman_val[i-1] : #for exemp num >= 500-100 = 900
                        st+= roman_dict[i-1]+roman_dict[i]
                        num-= roman_val[i]-roman_val[i-1]
                i-=1
        return(st)

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        i = len(s)-1
        curr_sym = ""
        roman_dict = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        flags = [False,False,False]# I,X,C
        res = 0
        while(i>=0):
            curr_sym = s[i]
            if (flags[0] and s[i]=='I') or (flags[1] and s[i]=='X') or (flags[2] and s[i]=='C') :
                res-= roman_dict[s[i]]
                i-=1
                flags = [False, False, False]  # I,X,C
            else:
                res+= roman_dict[s[i]]
                if s[i] == 'V' or s[i] == 'X':
                    flags = [True, False, False]
                elif s[i] == 'L' or s[i] == 'C':
                    flags = [False, True, False]
                elif s[i] == 'D' or s[i] == 'M':
                    flags = [False, False, True]
                i-=1
        return res

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""
        if len(strs) == 1:
            return strs[0]
        #strs = sorted(strs,key = len)
        st = strs[0]
        max_st = ""
        for j in range(0,len(st)):
            curr_sub = st[0:j+1]
            in_strs = True
            for s in strs:
                try:
                    if curr_sub != s[0:j + 1]:
                        in_strs = False
                        break
                except:
                    break
            if in_strs == False: break
            max_st = curr_sub
        return max_st

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        if n<3:
            return []
        if n == 3 :
            if nums[0] + nums[1] + nums[2] == 0: return [nums]
            else: return []
        res = []
        for i in range(0,n):
            for j in range(0,n):
                if j!= i:
                    for k in range(0,n):
                        if k!= j and k!= i:
                            if nums[k]+nums[j]+nums[i] == 0:
                                flag = True
                                temp_lst = [nums[k],nums[j],nums[i]]
                                temp_lst.sort()
                                if temp_lst not in res: res.append(temp_lst)
        return res

    def threesum_2(self,nums):
        n = len(nums)
        if n < 3:
            return []
        if n == 3:
            if nums[0] + nums[1] + nums[2] == 0:
                return [nums]
            else:
                return []
        res = []
        arr = sorted(nums)
        i = 0
        j = 1
        k = n - 1
        while (i < k):
            j = i + 1
            k = n - 1
            while j < k:
                solu = [arr[i], arr[j], arr[k]]
                sum = solu[0] + solu[1] + solu[2]
                if sum == 0:
                    if solu not in res:
                        res.append(solu)
                    j += 1
                if sum < 0:
                    prev = arr[j]
                    try:
                        while sum - prev + arr[j] < 0:
                            j += 1
                    except:
                        break  # all out of range
                if sum > 0:
                    prev = arr[k]
                    try:
                        while sum - prev + arr[k] > 0:
                            k -= 1
                    except:
                        break  # all out of range
            i += 1
        return res

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """















sol = Solution()
print(sol.threesum_2([-4,-2,1,-5,-4,-4,4,-2,0,4,0,-2,3,1,-5,0]))
#print("0:",ord("0"),"1:",ord("1"),"9:",ord("9"),)








