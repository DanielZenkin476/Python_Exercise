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
        st = ''
        i = len(roman_val)-1
        while(num>0):
            if num>= roman_val[i]:
                st+= roman_dict[i]
                num-= roman_val[i]
            else:
                i-=1
        return(st)












sol = Solution()
print(sol.intToRoman(6))
#print("0:",ord("0"),"1:",ord("1"),"9:",ord("9"),)








