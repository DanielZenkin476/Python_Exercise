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
        res = 0
        mult = 0
        if curr_val == 45 or curr_val == 43: # 44-43 = 1 for plus , 44-45 = -1 for minus, the read next
            sign = 44-curr_val
            st = st[1:]
            curr_char = st[0]
            curr_val = ord(curr_char)
        loop_count = 0
        while(48<=curr_val<=57):
            print(res)
            if curr_val ==48 and leading_zero == False:
                st = st[1:]
                curr_char = st[0]
                curr_val = ord(curr_char)
                pass
            else:
                if curr_val == 48 :
                    res = res * 10
                else:
                    print('curr:',curr_char)
                    res += int(curr_char)/(10**mult)
                    mult+=1
                st = st[1:]
                curr_char = st[0]
                curr_val = ord(curr_char)
                loop_count+=1
        res = sign * int(res*(10**loop_count-1))
        print(res)










sol = Solution()
sol.myAtoi("  -123213    ")
print("0:",ord("0"),"1:",ord("1"),"9:",ord("9"),)






