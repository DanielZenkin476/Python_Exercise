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
        roman_dict = {1:"I",5:"V",10:"X",50:"L",100:"C",500:"D",1000:"M",}
        st = ''
        while num >= 1000:
            st+= roman_dict[1000]
            num-=1000
        if num >= 900:
            st+=roman_dict[100]+roman_dict[1000]
            num-=900

        while num >= 500:
            st += roman_dict[500]
            num -= 500
        if num >= 400:
            st+=roman_dict[100]+roman_dict[500]
            num -= 400

        while num >= 100:
            st += roman_dict[100]
            num -= 100
        if num >= 90:
            st += roman_dict[10] + roman_dict[100]
            num -= 90

        while (num >= 50):
            st += roman_dict[50]
            num -= 50
        if num >= 40:
            st += roman_dict[10] + roman_dict[50]
            num -= 40

        while num >= 10:
            st += roman_dict[10]
            num -= 10
        if num >= 9:
            st += roman_dict[1] + roman_dict[10]
            num -= 9

        while (num >= 5):
            st += roman_dict[5]
            num -= 5
        if num >= 4:
            st += roman_dict[1] + roman_dict[5]
            num -= 4

        while num >= 1:
            st += roman_dict[1]
            num -= 1

        return st









sol = Solution()
print(sol.intToRoman(100))
#print("0:",ord("0"),"1:",ord("1"),"9:",ord("9"),)








