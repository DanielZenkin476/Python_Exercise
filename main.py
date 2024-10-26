from itertools import filterfalse


class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        st = s.lstrip()  # remove whitespaces if present
        curr_char = st[0]
        curr_val = ord(curr_char)
        sign = 1
        leading_zero = False
        val = 0
        mult = 0
        if curr_val == 45 or curr_val == 43: # 44-43 = 1 for plus , 44-45 = -1 for minus, the read next
            sign = 44-curr_val
            st = st[1:]
            curr_char = st[0]
            curr_val = ord(curr_char)
        print(st)
        print(sign)








sol = Solution()
sol.myAtoi("  -Test ss2   d    ")







